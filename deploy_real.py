import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union
import torch
from io import BytesIO
import draccus
import json_numpy
import numpy as np
import torch
import uvicorn
import msgpack

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from PIL import Image
from transformers import AutoTokenizer

from go1.internvl.model.go1 import GO1Model, GO1ModelConfig
from go1.internvl.train.constants import IMG_END_TOKEN
from go1.internvl.train.dataset import build_transform, dynamic_preprocess, preprocess_internvl2_5

json_numpy.patch()


def normalize(data, stats):
    """
    使用统计信息对数据进行归一化处理
    
    Args:
        data: 需要归一化的数据
        stats: 统计信息（均值和标准差）
        
    Returns:
        归一化后的数据
    """
    return (data - stats["mean"]) / (stats["std"] + 1e-6)


def unnormalize(data, stats):
    """
    使用统计信息对数据进行反归一化处理
    
    Args:
        data: 需要反归一化的数据
        stats: 统计信息（均值和标准差）
        
    Returns:
        反归一化后的数据
    """
    return data * stats["std"] + stats["mean"]


def get_stats_tensor(stats_json):
    """
    从JSON统计信息创建张量
    
    Args:
        stats_json: JSON格式的统计信息
        
    Returns:
        包含统计信息张量的字典
    """
    stats_tensor = {}

    stats_tensor["state"]={}
    stats_tensor["action"]={}

    stats_tensor["state"]["mean"] = torch.from_numpy(np.array(stats_json["observation.state"]["mean"]))
    stats_tensor["state"]["std"] = torch.from_numpy(np.array(stats_json["observation.state"]["std"]))
    stats_tensor["action"]["mean"] = torch.from_numpy(np.array(stats_json["action"]["mean"]))
    stats_tensor["action"]["std"] = torch.from_numpy(np.array(stats_json["action"]["std"]))


def multi_image_get_item(
    raw_target: Dict[str, Any],
    img_transform,
    text_tokenizer,
    num_image_token,
    cam_keys: list[str] = [
        "cam_head_color",
        "cam_hand_right_color",
        "cam_hand_left_color",
    ],
    dynamic_image_size=False,
    use_thumbnail=False,
    min_dynamic_patch=1,
    max_dynamic_patch=6,
    image_size=448,
):
    """
    处理多图像输入并生成模型输入项
    
    Args:
        raw_target: 原始目标数据，包含图像、文本指令等信息
        img_transform: 图像变换函数，用于图像预处理
        text_tokenizer: 文本分词器，用于处理文本数据
        num_image_token: 每个图像使用的token数量
        cam_keys: 相机键列表，指定使用的相机视角
        dynamic_image_size: 是否使用动态图像大小
        use_thumbnail: 是否使用缩略图
        min_dynamic_patch: 最小动态patch数
        max_dynamic_patch: 最大动态patch数
        image_size: 图像大小
        
    Returns:
        包含模型输入的字典
    """
    # 初始化图像列表和分块信息
    images, num_tiles = [], []
    num_image = 0
    
    # 遍历所有相机视角，处理图像数据
    for cam_key in cam_keys:
        # 检查当前视角图像是否存在
        if cam_key in raw_target:
            num_image += 1
            # 根据是否使用动态图像大小选择处理方式
            if dynamic_image_size:
                # 动态预处理图像，可能将单个图像分割为多个patch
                image = dynamic_preprocess(
                    raw_target[cam_key],
                    min_num=min_dynamic_patch,
                    max_num=max_dynamic_patch,
                    image_size=image_size,
                    use_thumbnail=use_thumbnail,
                )
                # 将处理后的图像添加到列表中
                images += image
                # 记录当前视角图像的分块数量
                num_tiles.append(len(image))
            else:
                # 直接添加原始图像
                images.append(raw_target[cam_key])
                # 每个视角图像计为1个分块
                num_tiles.append(1)

    # 对所有图像应用变换处理（如归一化、尺寸调整等）
    pixel_values = [img_transform(image) for image in images]
    # 将图像张量堆叠为批次
    pixel_values = torch.stack(pixel_values)
    # 获取图像块总数
    num_patches = pixel_values.size(0)

    # 计算每个视角的图像token数量
    num_image_tokens = [num_image_token * num_tile for num_tile in num_tiles]
    # 获取文本目标（如果不存在则为空字符串）
    ntp_target = raw_target.get("ntp_target", "")
    # 构建对话格式数据，包含人类指令和模型回复
    conversation = [
        {"from": "human", "value": f"{'<image>'*num_image}{raw_target['final_prompt']}"},
        {"from": "gpt", "value": ntp_target},
    ]
    # 使用InternVL2.5预处理函数处理对话数据
    ret = preprocess_internvl2_5(
        "internvl2_5",
        [conversation],
        text_tokenizer,
        num_image_tokens,
        num_image=num_image,
        group_by_length=True,
    )

    # 为打包数据集计算position_ids，用于标识每个token在序列中的位置
    position_ids = ret["attention_mask"].long().cumsum(-1) - 1
    position_ids.masked_fill_(ret["attention_mask"] == 0, 1)
    # 获取图像结束token的ID
    image_end_token_id = text_tokenizer.convert_tokens_to_ids(IMG_END_TOKEN)
    # 确保图像token没有被截断
    assert (ret["input_ids"][0] == image_end_token_id).sum() == num_image, "image tokens are truncated"

    # 创建最终返回字典，包含模型所需的所有输入数据
    final_ret = dict(
        # 文本输入ID序列
        input_ids=ret["input_ids"][0],
        # 标签序列，用于训练时计算损失
        labels=ret["labels"][0],
        # 注意力掩码，标识有效token位置
        attention_mask=ret["attention_mask"][0],
        # 位置ID，标识每个token的位置信息
        position_ids=position_ids[0],
        # 图像像素值，模型视觉编码器的输入
        pixel_values=pixel_values,
        # 图像标志，标识哪些输入是图像
        image_flags=torch.tensor([1] * num_patches, dtype=torch.long),
    )
    return final_ret


class GO1Infer:
    """
    GO1模型推理类，用于加载模型并执行推理
    """
    def __init__(
        self,
        model_path: Union[str, Path],
        data_stats_path: Union[str, Path] = None,
    ) -> Path:
        """
        初始化GO1推理模型
        
        Args:
            model_path: 模型路径
            data_stats_path: 数据统计信息路径
        """
        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

        self.config = GO1ModelConfig.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            ignore_mismatched_sizes=False,
        )
        self.image_size = self.config.force_image_size
        self.num_image_token: int = int(
            (self.image_size // self.config.vision_config.patch_size) ** 2 * (self.config.downsample_ratio**2)
        )
        self.dynamic_image_size = self.config.dynamic_image_size

        self.go1 = GO1Model.from_pretrained(model_path, config=self.config)
        self.go1.to(torch.bfloat16).to(self.device)

        self.img_transform = build_transform(
            is_train=False, input_size=self.image_size, pad2square=self.config.pad2square
        )
        self.text_tokenizer = AutoTokenizer.from_pretrained(
            model_path, add_eos_token=False, trust_remote_code=True, use_fast=False
        )

        self.norm = getattr(self.config, "norm", False)  # 如果配置中没有 norm 属性，则默认为 False
        if self.norm:
            assert data_stats_path is not None, "data_stats_path must be provided when norm is True"
            with open(data_stats_path, "rb") as f:
                self.data_stats = get_stats_tensor(json.load(f))

    def predict_action(self, inputs: Dict[str, Any]) -> str:
        """
        预测动作
        
        Args:
            inputs: 模型输入数据
            
        Returns:
            预测的动作
        """
        # print("开始推理...")
        pixel_values = inputs["pixel_values"]
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        position_ids = inputs["position_ids"]
        image_flags = inputs["image_flags"]
        ctrl_freqs = inputs["ctrl_freqs"]

        state = inputs["state"]
        # 如果需要，对状态进行归一化
        if self.norm:
            state = normalize(state, self.data_stats["state"])

        start_time = time.time()
        device = self.device
        with torch.no_grad():
            action = self.go1(
                pixel_values=pixel_values.to(dtype=torch.bfloat16, device=device),
                input_ids=input_ids.to(device).unsqueeze(0),
                attention_mask=attention_mask.to(device).unsqueeze(0),
                position_ids=position_ids.to(device).unsqueeze(0),
                image_flags=image_flags.to(device),
                state=state.to(dtype=torch.bfloat16, device=device).unsqueeze(0),
                ctrl_freqs=ctrl_freqs.to(dtype=torch.bfloat16, device=device).unsqueeze(0),
            )
        print(f"Model inference time: {(time.time() - start_time)*1000:.3f} ms")
        outputs = action[1][0].float().cpu()

        # 如果需要，对动作进行反归一化
        if self.norm:
            outputs = unnormalize(outputs, self.data_stats["action"])

        outputs = outputs.numpy()

        return outputs

    def inference(self, payload: Dict[str, Any]):
        """
        执行推理
        
        Args:
            payload: 输入数据负载
            需要对这个函数进行修改将回传的数据键进行修改
            
        Returns:
            推理结果
        """
        if "base_rgb_images" in payload:
            payload["cam_head_color"] = Image.fromarray(payload['base_rgb_images'])
        if "right" in payload:
            payload["cam_hand_right_color"] = Image.fromarray(payload["right"])
        if "low_rgb_images" in payload:
            payload["cam_hand_left_color"] = Image.fromarray(payload["low_rgb_images"])

        prompt = 'pick up the big workpiece.'
        print(f"获取的提示: {prompt}")
        payload["final_prompt"] = f"What action should the robot take to {prompt}?"

        inputs = multi_image_get_item(
            raw_target=payload,
            img_transform=self.img_transform,
            text_tokenizer=self.text_tokenizer,
            num_image_token=self.num_image_token,
            dynamic_image_size=self.dynamic_image_size,
            use_thumbnail=self.config.use_thumbnail,
            min_dynamic_patch=self.config.min_dynamic_patch,
            max_dynamic_patch=self.config.max_dynamic_patch,
            image_size=self.image_size,
        )

        inputs["state"] = torch.from_numpy(payload["state"]).unsqueeze(0)
        inputs["ctrl_freqs"] = torch.tensor([30])

        # for k in inputs:
        #     if torch is not None and isinstance(inputs[k], torch.Tensor):
        #         print(f"inputs[{k}] =", inputs[k].shape)

        return self.predict_action(inputs)


def to_serializable(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_serializable(v) for v in obj]
    return obj

def decompress_image(compressed_data: bytes, format: str, expected_shape: tuple) -> np.ndarray:
    """解压缩图像数据"""
    img = Image.open(BytesIO(compressed_data))
    
    if format == 'jpeg':
        # RGB图像
        img_array = np.array(img.convert('RGB'))
    elif format == 'png':
        # 深度图 (uint16)
        img_array = np.array(img)
        # 确保是正确的形状
        if len(img_array.shape) == 2:
            # 已经是2D数组
            pass
        else:
            # 可能需要转换
            img_array = img_array[:, :, 0] if img_array.shape[2] > 1 else img_array.squeeze()
    else:
        raise ValueError(f"不支持的格式: {format}")
    
    return img_array


def decode_observation(obs_packed: dict) -> dict:
    """解码客户端发送的observation（支持压缩格式）"""

    try:
        # 检查是否是压缩格式（新格式有'format'字段）
        base_rgb_info = obs_packed.get(b'base_rgb_images') or obs_packed.get('base_rgb_images')
        
        if isinstance(base_rgb_info, dict) and ('format' in base_rgb_info or b'format' in base_rgb_info):
            # 新格式：压缩的图像数据
            print("  📦 检测到压缩格式")
            
            # 解压缩RGB图像
            base_rgb = decompress_image(
                base_rgb_info.get(b'data') or base_rgb_info.get('data'),
                (base_rgb_info.get(b'format') or base_rgb_info.get('format')).decode() if isinstance(base_rgb_info.get(b'format') or base_rgb_info.get('format'), bytes) else base_rgb_info.get('format'),
                tuple(base_rgb_info.get(b'shape') or base_rgb_info.get('shape'))
            )
            
            low_rgb_info = obs_packed.get(b'low_rgb_images') or obs_packed.get('low_rgb_images')
            low_rgb = decompress_image(
                low_rgb_info.get(b'data') or low_rgb_info.get('data'),
                (low_rgb_info.get(b'format') or low_rgb_info.get('format')).decode() if isinstance(low_rgb_info.get(b'format') or low_rgb_info.get('format'), bytes) else low_rgb_info.get('format'),
                tuple(low_rgb_info.get(b'shape') or low_rgb_info.get('shape'))
            )
            
            
            print(f"  ✓ 解压完成 - base_rgb: {base_rgb.shape}")
            
        else:
            # 旧格式：原始字节数据（向后兼容）
            print("  📦 检测到原始格式")
            
            # 处理旧格式的数据
            base_rgb_data = base_rgb_info.get(b'data') or base_rgb_info.get('data')
            base_rgb_shape = tuple(base_rgb_info.get(b'shape') or base_rgb_info.get('shape'))
            base_rgb = np.frombuffer(base_rgb_data, dtype=np.uint8).reshape(base_rgb_shape)
            
            low_rgb_info = obs_packed.get(b'low_rgb_images') or obs_packed.get('low_rgb_images')
            low_rgb_data = low_rgb_info.get(b'data') or low_rgb_info.get('data')
            low_rgb_shape = tuple(low_rgb_info.get(b'shape') or low_rgb_info.get('shape'))
            low_rgb = np.frombuffer(low_rgb_data, dtype=np.uint8).reshape(low_rgb_shape)
        
        
        # 获取state
        state = np.array(obs_packed.get(b'state') or obs_packed.get('state'))
        
        # 构造标准格式的observation
        observation_raw = {
            'state': state,
            'base_rgb_images': base_rgb,
            'low_rgb_images': low_rgb,
            'ctrl_freqs': 30,  
            'instruction': (obs_packed.get(b'instruction') or obs_packed.get('instruction'))
        }
        
        return observation_raw
        
    except Exception as e:
        print(f"❌ 解码observation失败: {e}")
        import traceback
        traceback.print_exc()
        raise


class GO1Server:
    def __init__(self, model_path: Union[str, Path], data_stats_path: Optional[Union[str, Path]] = None) -> None:
        self.model = GO1Infer(model_path=model_path, data_stats_path=data_stats_path)
        self.app = FastAPI(title="GO1 WS Server", version="1.0.0")
        self._register_ws()

    def _register_ws(self) -> None:
        @self.app.websocket("/ws") #将下面的函数注册为WebSocket端点
        async def ws_endpoint(websocket: WebSocket):
            await websocket.accept()  # 等待并接受来自客户端的WebSocket连接请求
            try:
                while True:
                    msg = await websocket.receive()
                    # 仅接受二进制帧
                    data_bytes: Optional[bytes] = msg.get("bytes")
                    if data_bytes is None:
                        await websocket.send_text('{"error":"expect binary msgpack frame"}')
                        continue
                    try:
                        payload = msgpack.unpackb(data_bytes, raw=False, use_list=True)  # 客户端未设置 use_bin_type 也兼容
                        print(f"接收到的 payload keys: {list(payload.keys())}")
                    except Exception as e:
                        await websocket.send_text(f'{{"error":"msgpack unpack failed: {e}"}}')
                        continue
                    try:
                        payload = decode_observation(payload)

                    except Exception as e:
                        await websocket.send_text(f'{{"error":"reconstruct failed: {e}"}}')
                        continue
                    try:
                        result = self.model.inference(payload)
                    except Exception as e:
                        await websocket.send_text(f'{{"error":"inference failed: {e}"}}')
                        continue
                    out_bytes = msgpack.packb(to_serializable(result), use_bin_type=True)
                    await websocket.send(out_bytes)
            except WebSocketDisconnect:
                return
            except Exception as e:
                try:
                    await websocket.send_text(f'{{"error":"server error: {e}"}}')
                finally:
                    await websocket.close(code=1011)

    def run(self, host: str = "0.0.0.0", port: int = 8000, ws_max_mb: int = 64) -> None:
        # WHY: 图像帧较大，放宽上限（单位字节）
        uvicorn.run(
            self.app,
            host=host,
            port=port,
            ws="websockets",
            ws_max_size=ws_max_mb * 1024 * 1024,
        )

if __name__ == "__main__":
    # GO1Server("/home/vipuser/Desktop/pick_place_go1_air_4/","/home/vipuser/Desktop/AgiBot-World/fuwei/dataset_stats.json").run(host="0.0.0.0", port=8800)
    GO1Server("/root/.cache/huggingface/hub/models--MartinB7--go1_air_pick_place_air_6/snapshots/7dc976f98a04e51816aaa4a64c0c6248dc8171ba/","/home/vipuser/Desktop/stats.json").run(host="0.0.0.0", port=8800)