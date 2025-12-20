# banana_nodes_async.py
import torch
import torch.nn.functional as F
import os
import tempfile
from typing import Any, Dict, Optional, Union, List, Tuple, TYPE_CHECKING
from PIL import Image
import requests
import json
import numpy as np
from io import BytesIO
import traceback
import folder_paths 
import re
import threading
from datetime import datetime
import secrets 
import pandas as pd
import concurrent.futures

from . import get_api_key

if TYPE_CHECKING:
    from PIL import Image

# --- 工具函数 ---
def download_image(url: str, timeout: int = 60) -> Optional[Image.Image]:
    """从URL下载图像并返回PIL.Image对象"""
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=timeout)
        response.raise_for_status()
        return Image.open(BytesIO(response.content))
    except Exception as e:
        print(f"图像下载失败: {str(e)}")
        return None

def safe_pil_to_rgb(image: Image.Image) -> Image.Image:
    """安全地将任何PIL图像转换为RGB模式，处理透明度。"""
    if image.mode == 'RGBA':
        background = Image.new("RGB", image.size, (255, 255, 255))
        background.paste(image, mask=image.split()[3])
        return background
    elif image.mode == 'RGB':
        return image
    else:
        return image.convert('RGB')

def tensor_to_pil(tensor: torch.Tensor) -> List[Image.Image]:
    """将torch张量 (B, H, W, C) 转换为PIL图像列表"""
    if not isinstance(tensor, torch.Tensor): return []
    images = []
    for i in range(tensor.shape[0]):
        img_np = (torch.clamp(tensor[i], 0, 1).cpu().numpy() * 255).astype(np.uint8)
        images.append(Image.fromarray(img_np, 'RGB' if img_np.shape[-1] == 3 else 'RGBA'))
    return images

def safe_pil_batch_to_tensor(pil_images: List[Image.Image]) -> torch.Tensor:
    """将PIL图像列表安全地转换为ComfyUI图像张量，自动处理不同尺寸。"""
    if not pil_images:
        return torch.empty((0, 1, 1, 3), dtype=torch.float32)

    tensors = []
    max_h = 0
    max_w = 0
    
    for pil_image in pil_images:
        if pil_image is None: continue
        try:
            pil_image = safe_pil_to_rgb(pil_image)
            img_array = np.array(pil_image).astype(np.float32) / 255.0
            
            if len(img_array.shape) == 2: 
                img_array = np.stack((img_array,)*3, axis=-1)
            elif img_array.shape[2] == 4: 
                img_array = img_array[:,:,:3]

            tensor = torch.from_numpy(img_array)[None,]
            if tensor.shape[1] > max_h: max_h = tensor.shape[1]
            if tensor.shape[2] > max_w: max_w = tensor.shape[2]
            tensors.append(tensor)
        except Exception as e:
            print(f"Warning: 跳过损坏的图像: {e}")
            continue

    if not tensors: return torch.empty((0, 1, 1, 3), dtype=torch.float32)
    
    padded_tensors = []
    for tensor in tensors:
        b, h, w, c = tensor.shape
        if h == 0 or w == 0 or c != 3:
            continue
        if h == max_h and w == max_w:
            padded_tensors.append(tensor)
            continue
        
        tensor_chw = tensor.permute(0, 3, 1, 2)
        pad_w = max_w - w
        pad_h = max_h - h
        padding = (0, pad_w, 0, pad_h) 
        padded_tensor_chw = F.pad(tensor_chw, padding, "constant", 0)
        padded_tensor_hwc = padded_tensor_chw.permute(0, 2, 3, 1)
        padded_tensors.append(padded_tensor_hwc)

    if not padded_tensors:
         return torch.empty((0, 1, 1, 3), dtype=torch.float32)

    try:
        return torch.cat(padded_tensors, dim=0)
    except Exception as e:
        print(f"Error: 最终张量合并失败: {e}")
        traceback.print_exc()
        return padded_tensors[0]

def format_error_message(error: Exception) -> str:
    return f"{type(error).__name__}: {str(error)}"

# --- 上传功能 ---
def get_upload_token_zh(api_key: str, data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    url = "https://grsai.dakka.com.cn/client/resource/newUploadTokenZH"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    response = requests.post(url=url, headers=headers, json=data or {}, timeout=30)
    response.raise_for_status()
    return response.json()

def upload_file_zh(file_path: str, api_key: str) -> str:
    if not file_path or not api_key: 
        return ""
    if not os.path.exists(file_path): 
        raise FileNotFoundError(f"文件不存在: {file_path}")
    
    file_extension = os.path.splitext(file_path)[1].lstrip(".") or "png"
    result = get_upload_token_zh(api_key, {"sux": file_extension})
    token, key, url, domain = (
        result["data"]["token"], 
        result["data"]["key"], 
        result["data"]["url"], 
        result["data"]["domain"]
    )
    
    with open(file_path, "rb") as file:
        upload_response = requests.post(
            url=url, 
            data={"token": token, "key": key}, 
            files={"file": file}, 
            timeout=120
        )
        upload_response.raise_for_status()
    
    return f"{domain}/{key}"

def upload_image_tensor(image_tensor: torch.Tensor, api_key: str, index: int) -> Optional[str]:
    try:
        pil_image = tensor_to_pil(image_tensor)[0]
        rgb_pil = safe_pil_to_rgb(pil_image)
        with tempfile.NamedTemporaryFile(suffix=f"_{index}.png", delete=False) as temp_file:
            rgb_pil.save(temp_file, "PNG")
            temp_path = temp_file.name
        image_url = upload_file_zh(temp_path, api_key)
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        return image_url if image_url else None
    except Exception as e:
        print(f"图像上传失败: {e}")
        return None

# --- API 客户端 ---
class GrsaiAPIError(Exception): pass

class GrsaiAPI:
    def __init__(self, api_key: str):
        if not api_key or not api_key.strip(): raise GrsaiAPIError("API密钥不能为空")
        self.api_key = api_key
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json; charset=utf-8", "User-Agent": "ComfyUI-Nkxx/1.0", "Authorization": f"Bearer {self.api_key}"})

    def _make_request(self, method: str, endpoint: str, data: Optional[Dict] = None, timeout: int = 300) -> Dict[str, Any]:
        url = f"https://grsai.dakka.com.cn{endpoint}"
        response = self.session.request(method, url, json=data, timeout=timeout)
        response.raise_for_status()
        text = response.text
        json_data = text[6:] if text.startswith("data: ") else text
        return json.loads(json_data)

# --- 配置 ---
SUPPORTED_ASPECT_RATIOS = ["auto", "1:1", "16:9", "9:16", "4:3", "3:4", "3:2", "2:3", "5:4", "4:5", "21:9"]
# 模型列表常量
SUPPORTED_MODELS_ASYNC = ["nano-banana-fast", "nano-banana-pro", "nano-banana-pro-vt"]

# --- 节点基类 ---
class _GrsaiNodeBase:
    FUNCTION = "execute"
    @classmethod
    def IS_CHANGED(cls, **kwargs): return float("NaN")

    def _create_error_result(self, error_message: str, is_text_output: bool = False):
        print(f"节点执行错误: {error_message}")
        if is_text_output:
            return {"ui": {"string": [error_message]}, "result": (None, error_message)}
        image_out = torch.zeros((1, 1, 1, 3), dtype=torch.float32)
        return {"ui": {"string": [error_message]}, "result": (image_out, f"失败: {error_message}")}

    def _get_credits_balance(self, api_key: str) -> int:
        try:
            url = f"https://grsai.dakka.com.cn/client/common/getCredits?apikey={api_key}"
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data.get("code") == 0 and "data" in data and "credits" in data["data"]:
                    return int(data["data"]["credits"])
        except Exception:
            pass
        return -1
        
    def _handle_image_uploads(self, images_in: List[Optional[torch.Tensor]], api_key: str):
        uploaded_urls = []
        if not any(img is not None for img in images_in): 
            return uploaded_urls
        try:
            for i, image_tensor in enumerate(images_in):
                if image_tensor is None: continue
                image_url = upload_image_tensor(image_tensor, api_key, i)
                if image_url:
                    uploaded_urls.append(image_url)
            return uploaded_urls
        except Exception as e:
            raise Exception(f"图像上传失败: {format_error_message(e)}")

    def _make_api_request_with_error_handling(self, api_client, endpoint, payload):
        try:
            response = api_client._make_request("POST", endpoint, data=payload)
            if response is None: raise GrsaiAPIError("API 返回了空响应 (None)")
            if not isinstance(response, dict): raise GrsaiAPIError(f"API 返回了无效的数据类型: {type(response)}")
            code = response.get("code")
            if code is not None and code != 0:
                msg = response.get("msg", "未知错误")
                raise GrsaiAPIError(f"API 错误码 {code}: {msg}")
            data = response.get("data")
            if data is None: raise GrsaiAPIError(f"API 响应中没有 'data' 字段")
            task_id = data.get("id")
            if not task_id: raise GrsaiAPIError(f"API 未返回有效的任务ID")
            return response
        except requests.exceptions.RequestException as e:
            raise GrsaiAPIError(f"网络请求失败: {str(e)}")
        except json.JSONDecodeError as e:
            raise GrsaiAPIError(f"JSON 解析失败: {str(e)}")
        except Exception as e:
            raise GrsaiAPIError(f"未知错误: {str(e)}")

# --- 异步任务管理 ---
BANANA_TASK_FILE = os.path.join(folder_paths.get_temp_directory(), "banana_task_history.json")
MAX_BANANA_HISTORY_DOWNLOADED = 5
banana_task_lock = threading.Lock()

ZERO_WIDTH_CHARS = ["\u200b", "\u200c", "\u200d", "\ufeff", "\u180e", "\u200e", "\u200f"]

def _read_banana_tasks():
    if not os.path.exists(BANANA_TASK_FILE): return {}
    try:
        with open(BANANA_TASK_FILE, 'r', encoding='utf-8') as f:
            tasks = json.load(f)
        return tasks if isinstance(tasks, dict) else {}
    except (json.JSONDecodeError, IOError):
        return {}

def _write_banana_tasks(tasks):
    try:
        downloaded_tasks = [(task_id, info) for task_id, info in tasks.items() if info.get("status") == "downloaded"]
        if len(downloaded_tasks) > MAX_BANANA_HISTORY_DOWNLOADED:
            downloaded_tasks.sort(key=lambda x: x[1].get("submitted_at", "1970-01-01 00:00:00"))
            tasks_to_remove = downloaded_tasks[:-MAX_BANANA_HISTORY_DOWNLOADED]
            for task_id, _ in tasks_to_remove:
                if task_id in tasks: del tasks[task_id]
        
        with open(BANANA_TASK_FILE, 'w', encoding='utf-8') as f:
            json.dump(tasks, f, indent=4, ensure_ascii=False, sort_keys=True)
    except IOError as e:
        print(f"[Banana Task Manager] 写入任务文件失败: {e}")

def _get_next_task_number(is_batch=False):
    """
    根据任务类型获取下一个序号。
    is_batch=False: 查找 "任务X"
    is_batch=True:  查找 "批量任务X"
    """
    tasks = _read_banana_tasks()
    existing_nums = []
    
    prefix = "批量任务" if is_batch else "任务"
    
    for task_key in tasks.keys():
        if task_key.startswith(prefix):
            # 确保严格匹配前缀，防止 "批量任务" 匹配到 "任务"
            if not is_batch and task_key.startswith("批量任务"):
                continue
                
            try:
                # 提取数字部分
                num_str = task_key[len(prefix):]
                if num_str.isdigit():
                    existing_nums.append(int(num_str))
            except ValueError:
                continue
                
    return max(existing_nums, default=0) + 1

# --- 节点 1: NanoBanana 异步提交 (单任务) ---
class NanoBananaAsyncSubmit(_GrsaiNodeBase):
    CATEGORY = "Nkxx/图像"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
                    "prompt": ("STRING", {"multiline": True, "default": "一只可爱的小猫"}),
                    "model": (SUPPORTED_MODELS_ASYNC, {"default": "nano-banana-fast"}),
                    "image_size": (["默认", "1K", "2K", "4K"], {"default": "默认"}),
                    "concurrency": ("INT", {"default": 1, "min": 1, "max": 10, "step": 1}),
                    "aspect_ratio": (SUPPORTED_ASPECT_RATIOS, {"default": "auto"}),
                }, "optional": {
                    "api_key": ("STRING", {"multiline": False, "default": "", "placeholder": "留空则使用 __init__.py 中的配置"}),
                    "image_1": ("IMAGE",), "image_2": ("IMAGE",),
                    "image_3": ("IMAGE",), "image_4": ("IMAGE",),
                    "image_5": ("IMAGE",), "image_6": ("IMAGE",),
                    "image_7": ("IMAGE",), "image_8": ("IMAGE",),
                    "image_9": ("IMAGE",), "image_10": ("IMAGE",),
                    "image_11": ("IMAGE",), "image_12": ("IMAGE",),
                    "image_13": ("IMAGE",), "image_14": ("IMAGE",),		
                }}
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    FUNCTION = "submit"
    
    @classmethod
    def IS_CHANGED(cls, **kwargs): return float("NaN")
    
    def submit(self, prompt: str, model: str, image_size: str, concurrency: int, aspect_ratio: str, api_key: str = "", **kwargs):
        final_api_key = get_api_key(api_key)
        if not final_api_key: 
            return {"ui": {"string": ["API Key 不能为空。"]}, "result": ("API Key 不能为空。",)}
        
        images_in = [kwargs.get(f"image_{i}") for i in range(1, 15)]
        try:
            uploaded_urls = self._handle_image_uploads(images_in, final_api_key)
            task_num = _get_next_task_number(is_batch=False)
            task_id = f"任务{task_num}"
            api_client = GrsaiAPI(api_key=final_api_key)
            subtasks = []

            final_image_size = None
            # 支持 Pro 系列模型设置分辨率
            if model in ["nano-banana-pro", "nano-banana-pro-vt"]:
                if image_size == "默认": final_image_size = "2K"
                else: final_image_size = image_size
            
            for i in range(concurrency):
                payload = {
                    "model": model,
                    "prompt": f"{prompt}{secrets.choice(ZERO_WIDTH_CHARS) * i}",
                    "aspectRatio": aspect_ratio,
                    "urls": uploaded_urls,
                    "webHook": "-1",
                    "shutProgress": True
                }
                if final_image_size: payload["imageSize"] = final_image_size
                response = self._make_api_request_with_error_handling(api_client, "/v1/draw/nano-banana", payload)
                subtasks.append({
                    "api_task_id": response["data"]["id"],
                    "status": "running",
                    "image_url": None,
                    "progress": 0,
                    "failure_reason": None
                })
            
            with banana_task_lock:
                tasks = _read_banana_tasks()
                tasks[task_id] = {
                    "type": "normal", # 标记为普通任务
                    "prompt": prompt,
                    "model": model,
                    "submitted_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "status": "running",
                    "aspect_ratio": aspect_ratio,
                    "concurrency": concurrency,
                    "subtasks": subtasks
                }
                _write_banana_tasks(tasks)
            
            credits = self._get_credits_balance(final_api_key)
            status_msg = f"任务提交成功 | {task_id} | 模型: {model} | 子任务数: {concurrency} | 积分: {credits if credits >= 0 else 'N/A'}"
            return {"ui": {"string": [status_msg]}, "result": (status_msg,)}
        except Exception as e:
            error_msg = f"提交失败: {format_error_message(e)}"
            print(f"节点执行错误: {error_msg}")
            return {"ui": {"string": [error_msg]}, "result": (error_msg,)}

# --- 节点 2: NanoBanana 异步批量提交 (新增) ---
class NanoBananaAsyncBatchSubmit(_GrsaiNodeBase):
    CATEGORY = "Nkxx/图像"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
                "file_path": ("STRING", {"default": "", "placeholder": "拖拽 CSV/Excel 文件至此"}),
                "column_name": ("STRING", {"default": "prompt"}),
                "prompt_prefix": ("STRING", {"multiline": True, "default": ""}),
                "model": (SUPPORTED_MODELS_ASYNC, {"default": "nano-banana-fast"}),
                "image_size": (["默认", "1K", "2K", "4K"], {"default": "默认"}),
                "aspect_ratio": (SUPPORTED_ASPECT_RATIOS, {"default": "auto"}),
                "executions_per_prompt": ("INT", {"default": 1, "min": 1, "max": 10, "step": 1, "label": "单提示词执行次数"}),
            }, "optional": {
                "api_key": ("STRING", {"multiline": False, "default": "", "placeholder": "留空则使用 __init__.py 中的配置"}),
                "image_1": ("IMAGE",), "image_2": ("IMAGE",),
                "image_3": ("IMAGE",), "image_4": ("IMAGE",),
            }}
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    FUNCTION = "submit_batch"
    
    @classmethod
    def IS_CHANGED(cls, **kwargs): return float("NaN")
    
    def submit_batch(self, file_path: str, column_name: str, prompt_prefix: str, model: str, image_size: str, aspect_ratio: str, executions_per_prompt: int, api_key: str = "", **kwargs):
        final_api_key = get_api_key(api_key)
        if not final_api_key: 
            return {"ui": {"string": ["API Key 不能为空。"]}, "result": ("API Key 不能为空。",)}
        
        # 1. 验证和读取文件
        if not file_path or not os.path.exists(file_path):
            return self._create_error_result("文件路径为空或文件不存在。", is_text_output=True)
        
        try:
            if file_path.lower().endswith('.csv'): df = pd.read_csv(file_path, encoding='utf-8')
            elif file_path.lower().endswith(('.xls', '.xlsx')): df = pd.read_excel(file_path)
            else: return self._create_error_result("仅支持 .csv, .xls, .xlsx 文件。", is_text_output=True)
        except Exception as e: return self._create_error_result(f"读取文件失败: {format_error_message(e)}", is_text_output=True)
        
        if column_name not in df.columns: return self._create_error_result(f"列 '{column_name}' 不存在。", is_text_output=True)
        
        base_prompts = [f"{prompt_prefix}{p}" for p in df[column_name].dropna().astype(str).tolist()]
        if not base_prompts: return self._create_error_result(f"列 '{column_name}' 中未找到有效 prompt。", is_text_output=True)
        
        # 扩展 Prompt 列表 (单词多次执行)
        prompts = [p for p in base_prompts for _ in range(max(1, executions_per_prompt))]
        total_tasks = len(prompts)

        # 2. 处理图片上传 (批量节点只支持4张)
        images_in = [kwargs.get(f"image_{i}") for i in range(1, 5)]
        try:
            uploaded_urls = self._handle_image_uploads(images_in, final_api_key)
        except Exception as e:
            return self._create_error_result(str(e), is_text_output=True)

        # 3. 准备参数
        final_image_size = None
        if model in ["nano-banana-pro", "nano-banana-pro-vt"]:
            if image_size == "默认": final_image_size = "2K"
            else: final_image_size = image_size
        
        api_client = GrsaiAPI(api_key=final_api_key)
        subtasks = []
        errors = []

        # 4. 并发提交 API 任务
        # 虽然是异步节点，但我们需要快速将所有请求发给服务器拿到 task_id
        def submit_single_req(p_idx, prompt_text):
            try:
                # 添加零宽字符防止去重缓存 (如果有的话)
                final_prompt = f"{prompt_text}{secrets.choice(ZERO_WIDTH_CHARS) * (p_idx % 10)}"
                payload = {
                    "model": model,
                    "prompt": final_prompt,
                    "aspectRatio": aspect_ratio,
                    "urls": uploaded_urls,
                    "webHook": "-1",
                    "shutProgress": True
                }
                if final_image_size: payload["imageSize"] = final_image_size
                response = self._make_api_request_with_error_handling(api_client, "/v1/draw/nano-banana", payload)
                return {
                    "api_task_id": response["data"]["id"],
                    "status": "running",
                    "image_url": None,
                    "progress": 0,
                    "failure_reason": None,
                    "original_prompt": prompt_text
                }
            except Exception as e:
                return {"error": str(e)}

        # 使用线程池加快提交速度
        print(f"[Banana Async Batch] 开始提交 {total_tasks} 个任务...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            future_to_idx = {executor.submit(submit_single_req, i, p): i for i, p in enumerate(prompts)}
            for future in concurrent.futures.as_completed(future_to_idx):
                res = future.result()
                if "error" in res:
                    errors.append(res["error"])
                else:
                    subtasks.append(res)
        
        if not subtasks:
             return self._create_error_result(f"所有任务提交均失败。错误示例: {errors[0] if errors else '未知'}", is_text_output=True)

        # 5. 写入任务记录
        task_num = _get_next_task_number(is_batch=True)
        task_id = f"批量任务{task_num}"
        
        with banana_task_lock:
            tasks = _read_banana_tasks()
            tasks[task_id] = {
                "type": "batch", # 标记为批量任务
                "prompt": f"批量文件: {os.path.basename(file_path)}", # 批量任务不需要详细prompt，记个文件名
                "model": model,
                "submitted_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "status": "running",
                "aspect_ratio": aspect_ratio,
                "concurrency": total_tasks, # 这里concurrency代表总子任务数
                "subtasks": subtasks
            }
            _write_banana_tasks(tasks)

        credits = self._get_credits_balance(final_api_key)
        status_msg = f"批量提交完成 | {task_id} | 成功提交: {len(subtasks)}/{total_tasks} | 积分: {credits if credits >= 0 else 'N/A'}"
        if errors:
            status_msg += f" | 提交失败数: {len(errors)}"
            
        return {"ui": {"string": [status_msg]}, "result": (status_msg,)}


# --- 节点 3: NanoBanana 异步查询下载 (通用) ---
class NanoBananaAsyncQuery(_GrsaiNodeBase):
    CATEGORY = "Nkxx/图像"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {"optional": {
                    "api_key": ("STRING", {"multiline": False, "default": "", "placeholder": "留空则使用 __init__.py 中的配置"}),
                }}
    
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "status")
    FUNCTION = "query_and_download"
    
    @classmethod
    def IS_CHANGED(cls, **kwargs): return float("NaN")

    def _generate_display_status(self, tasks: Dict) -> str:
        all_tasks_status = []
        
        # 中文状态映射表
        status_map = {
            "running": "运行中",
            "pending": "排队中",
            "succeeded": "已完成", # 指服务端完成，等待下载
            "failed": "失败",
            "downloaded": "已下载"
        }

        # 按提交时间倒序显示
        for tid, tinfo in sorted(tasks.items(), key=lambda x: x[1].get('submitted_at', ''), reverse=True):
            raw_status = tinfo.get('status', 'running')
            task_type = tinfo.get('type', 'normal') # 默认为 normal 兼容旧数据
            concurrency = tinfo.get('concurrency', 0)
            
            # 计算子任务进度
            subtasks = tinfo.get('subtasks', [])
            total_sub = len(subtasks)
            
            # 统计成功/失败数
            success_sub = sum(1 for s in subtasks if s.get('status') == 'succeeded')
            fail_sub = sum(1 for s in subtasks if s.get('status') == 'failed')
            done_sub = success_sub + fail_sub
            
            if raw_status == 'pending': raw_status = 'running'
            
            # 获取中文状态
            display_str = status_map.get(raw_status, raw_status).upper()
            
            if task_type == 'batch':
                # 批量任务：显示详细进度 (成功/失败)
                # 格式: [运行中] 批量任务1 - 进度: 10/50 (成功:8 失败:2)
                info_text = f"进度: {done_sub}/{total_sub}"
                if done_sub > 0:
                    info_text += f" (成功: {success_sub} | 失败: {fail_sub})"
                all_tasks_status.append(f"[{display_str}] {tid} - {info_text}")
            else:
                # 普通任务：显示Prompt缩略
                prompt_full = tinfo.get('prompt', '')
                prompt_snippet = prompt_full[:15] + "..." if len(prompt_full) > 15 else prompt_full
                all_tasks_status.append(f"[{display_str}] {tid} ({prompt_snippet}) - {concurrency}个子任务")
        
        if not all_tasks_status:
            return "当前无任务记录"
        return "\n".join(all_tasks_status)
    
    def query_and_download(self, api_key: str = ""):
        final_api_key = get_api_key(api_key)
        if not final_api_key: 
            return self._create_error_result("API Key 不能为空。")
        
        with banana_task_lock:
            tasks = _read_banana_tasks()
        
        if not tasks:
            return self._create_error_result("当前没有任务记录。", is_text_output=True)

        api_client = GrsaiAPI(api_key=final_api_key)
        
        # =================================================
        # 第一步：刷新所有未“完结”的任务状态
        # =================================================
        active_tasks = []
        for tid, tinfo in tasks.items():
            if tinfo.get("status") not in ["succeeded", "failed", "downloaded"]:
                active_tasks.append((tid, tinfo))
        
        tasks_updated = False
        
        if active_tasks:
            print(f"[Banana Query] 正在刷新 {len(active_tasks)} 个活跃任务的状态...")
            
            for tid, tinfo in active_tasks:
                subtasks = tinfo.get("subtasks", [])
                if not subtasks: continue
                
                any_sub_running = False 
                any_sub_succeeded = False 
                subtask_updated = False
                
                # 遍历所有子任务查询状态
                # 对于批量任务，子任务可能很多，这里依然逐个查询（API限制通常较宽，但如果非常多可能需要优化，暂保持逐个）
                for subtask in subtasks:
                    current_status = subtask.get("status", "running")
                    if current_status in ["succeeded", "failed"]:
                        if current_status == "succeeded":
                            any_sub_succeeded = True
                        continue 
                        
                    api_task_id = subtask.get("api_task_id")
                    if not api_task_id: 
                        current_status = "failed"
                    else:
                        try:
                            response = api_client._make_request("POST", "/v1/draw/result", data={"id": api_task_id})
                            query_data = response.get("data", {})
                            
                            if not query_data:
                                current_status = "running" 
                            else:
                                api_status = query_data.get("status", "running")
                                if api_status == "pending":
                                    current_status = "running"
                                else:
                                    current_status = api_status
                                    
                                if query_data.get("results") and len(query_data["results"]) > 0:
                                    subtask["image_url"] = query_data["results"][0].get("url")
                                    
                                if current_status == "failed":
                                    subtask["failure_reason"] = query_data.get("failure_reason", "未知错误")
                                    
                        except Exception as e:
                            # print(f"[Banana Query] 查询子任务 {api_task_id} 异常: {e}")
                            current_status = "running"

                    if subtask.get("status") != current_status:
                        subtask["status"] = current_status
                        subtask_updated = True
                    
                    if current_status == "running":
                        any_sub_running = True
                    elif current_status == "succeeded":
                        any_sub_succeeded = True
                
                old_status = tinfo.get("status")
                
                if any_sub_running:
                    final_main_status = "running"
                else:
                    if any_sub_succeeded:
                        final_main_status = "succeeded"
                    else:
                        final_main_status = "failed"
                
                if final_main_status != old_status or subtask_updated:
                    tinfo["status"] = final_main_status
                    tinfo["subtasks"] = subtasks
                    tasks_updated = True

        if tasks_updated:
            with banana_task_lock:
                current_snapshot = _read_banana_tasks()
                for tid, tinfo in tasks.items():
                    if tid in current_snapshot:
                        current_snapshot[tid] = tinfo
                _write_banana_tasks(current_snapshot)
                tasks = current_snapshot 

        # =================================================
        # 第二步：下载 (succeeded 状态可能包含部分成功)
        # =================================================
        succeeded_candidates = []
        for tid, tinfo in tasks.items():
            if tinfo.get("status") == "succeeded":
                succeeded_candidates.append((tid, tinfo))
        
        status_display_text = self._generate_display_status(tasks)
        
        if not succeeded_candidates:
            return {"ui": {"string": [status_display_text]}, "result": (torch.zeros((1, 1, 1, 3), dtype=torch.float32), status_display_text)}
        
        # 按提交时间排序，下载最早的一个 (无论是普通任务还是批量任务)
        succeeded_candidates.sort(key=lambda x: x[1].get("submitted_at", "2999-01-01"))
        target_tid, target_tinfo = succeeded_candidates[0]
        
        try:
            pil_images = []
            subtasks = target_tinfo.get("subtasks", [])
            
            # 并发下载图片，因为批量任务可能有几十张图
            download_futures = {}
            with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                for sub in subtasks:
                    if sub.get("status") == "succeeded" and sub.get("image_url"):
                        future = executor.submit(download_image, sub.get("image_url"))
                        download_futures[future] = sub
                
                for future in concurrent.futures.as_completed(download_futures):
                    img = future.result()
                    if img:
                        pil_images.append(img)
            
            if not pil_images:
                 return self._create_error_result(f"{target_tid} 图片下载失败 (部分成功任务)。", is_text_output=True)
            
            with banana_task_lock:
                current_tasks = _read_banana_tasks()
                if target_tid in current_tasks:
                    current_tasks[target_tid]["status"] = "downloaded"
                    _write_banana_tasks(current_tasks)
                    status_display_text = self._generate_display_status(current_tasks)

            # 统计
            total_subtasks = len(subtasks)
            success_count = len(pil_images)
            fail_count = total_subtasks - success_count
            
            credits = self._get_credits_balance(final_api_key)
            credits_str = str(credits) if credits >= 0 else 'N/A'

            if fail_count > 0:
                result_title = "部分成功"
                count_info = f"成功: {success_count} | 失败: {fail_count}"
            else:
                result_title = "下载成功"
                count_info = f"共 {success_count} 张"

            final_msg = f"{result_title}: {target_tid} | {count_info} | 积分: {credits_str}\n{status_display_text}"
            
            return {"ui": {"string": [final_msg]}, "result": (safe_pil_batch_to_tensor(pil_images), final_msg)}
            
        except Exception as e:
            return self._create_error_result(f"下载过程出错: {e}", is_text_output=True)

# --- 节点注册 ---
NODE_CLASS_MAPPINGS = {
    "NanoBananaAsyncSubmit": NanoBananaAsyncSubmit,
    "NanoBananaAsyncBatchSubmit": NanoBananaAsyncBatchSubmit,
    "NanoBananaAsyncQuery": NanoBananaAsyncQuery,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NanoBananaAsyncSubmit": "🍌 Nano Banana 异步提交",
    "NanoBananaAsyncBatchSubmit": "🍌 Nano Banana 异步批量提交 (CSV/Excel)",
    "NanoBananaAsyncQuery": "🍌 Nano Banana 异步查询下载",
}