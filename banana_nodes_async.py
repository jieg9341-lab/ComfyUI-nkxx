import torch
import torch.nn.functional as F
import os
import tempfile
from typing import Any, Dict, Optional, Union, List, Tuple, TYPE_CHECKING
from PIL import Image
import pandas as pd
import requests
import concurrent.futures
import json
import base64
import numpy as np
from io import BytesIO
import time
import traceback
import folder_paths # 确保 folder_paths 被导入
import re
import threading
from datetime import datetime
import secrets # 确保 secrets 被导入

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

def pil_to_tensor(pil_images: Union[Image.Image, List[Image.Image]]) -> torch.Tensor:
    """将PIL图像或列表转换为ComfyUI图像张量"""
    if not isinstance(pil_images, list): pil_images = [pil_images]
    tensors = []
    for pil_image in pil_images:
        img_array = np.array(pil_image).astype(np.float32) / 255.0
        tensor = torch.from_numpy(img_array)[None,]
        tensors.append(tensor)
    if not tensors: return torch.empty((0, 1, 1, 3), dtype=torch.float32)
    return torch.cat(tensors, dim=0)

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
            # 确保是 RGB
            pil_image = safe_pil_to_rgb(pil_image)
            
            img_array = np.array(pil_image).astype(np.float32) / 255.0
            
            # 确保3通道
            if len(img_array.shape) == 2: # 灰度图
                img_array = np.stack((img_array,)*3, axis=-1)
            elif img_array.shape[2] == 4: # RGBA
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
            print(f"Warning: 跳过无效尺寸的张量: shape {tensor.shape}")
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
         print("Warning: 填充后没有有效的张量。")
         return torch.empty((0, 1, 1, 3), dtype=torch.float32)

    try:
        return torch.cat(padded_tensors, dim=0)
    except Exception as e:
        print(f"Error: 最终张量合并失败: {e}")
        traceback.print_exc()
        return padded_tensors[0]

def format_error_message(error: Exception) -> str:
    """格式化错误消息"""
    return f"{type(error).__name__}: {str(error)}"

def sanitize_filename(text: str, max_length: int = 100) -> str:
    """清理字符串，使其成为有效的文件名。"""
    sanitized = re.sub(r'[\\/*?:"<>|]', '_', text)
    sanitized = re.sub(r'[\s_]+', '_', sanitized)
    sanitized = sanitized.strip('_')
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length]
    return sanitized

# --- 上传功能 ---
def get_upload_token_zh(api_key: str, data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    url = "https://grsai.dakka.com.cn/client/resource/newUploadTokenZH"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    response = requests.post(url=url, headers=headers, json=data or {}, timeout=30)
    response.raise_for_status()
    return response.json()

def upload_file_zh(file_path: str, api_key: str) -> str:
    """上传文件并返回URL，接收api_key参数"""
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
    """上传单个图像张量并返回URL"""
    try:
        pil_image = tensor_to_pil(image_tensor)[0]
        rgb_pil = safe_pil_to_rgb(pil_image)
        with tempfile.NamedTemporaryFile(suffix=f"_{index}.png", delete=False) as temp_file:
            rgb_pil.save(temp_file, "PNG")
            temp_path = temp_file.name
        
        # 上传并获取URL
        image_url = upload_file_zh(temp_path, api_key)
        
        # 清理临时文件
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
        # *** 修复: 笔误 _response -> response ***
        response.raise_for_status()
        text = response.text
        json_data = text[6:] if text.startswith("data: ") else text
        return json.loads(json_data)

# --- 配置 ---
SUPPORTED_ASPECT_RATIOS = ["auto", "1:1", "16:9", "9:16", "4:3", "3:4", "3:2", "2:3", "5:4", "4:5", "21:9"]

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
        """处理图像上传，返回有效的URL列表"""
        uploaded_urls = []
        if not any(img is not None for img in images_in): 
            return uploaded_urls
        
        try:
            # *** 关键修复：移除并发，使用普通循环 ***
            # 避免 ThreadPoolExecutor 导致的死锁
            for i, image_tensor in enumerate(images_in):
                if image_tensor is None: 
                    continue
                
                # 上传图像并获取URL
                image_url = upload_image_tensor(image_tensor, api_key, i)
                if image_url:
                    uploaded_urls.append(image_url)
            
            return uploaded_urls
            
        except Exception as e:
            raise Exception(f"图像上传失败: {format_error_message(e)}")

    def _cleanup_temp_files(self, temp_files: List[str]):
        for path in temp_files:
            if os.path.exists(path): os.unlink(path)

    def _make_api_request_with_error_handling(self, api_client, endpoint, payload):
        """封装 API 请求，提供更详细的错误信息"""
        try:
            response = api_client._make_request("POST", endpoint, data=payload)
            
            if response is None:
                raise GrsaiAPIError("API 返回了空响应 (None)")
            
            if not isinstance(response, dict):
                raise GrsaiAPIError(f"API 返回了无效的数据类型: {type(response)}, 内容: {response}")
            
            code = response.get("code")
            if code is not None and code != 0:
                msg = response.get("msg", "未知错误")
                raise GrsaiAPIError(f"API 错误码 {code}: {msg}")
            
            data = response.get("data")
            if data is None:
                raise GrsaiAPIError(f"API 响应中没有 'data' 字段。完整响应: {response}")
            
            task_id = data.get("id")
            if not task_id:
                raise GrsaiAPIError(f"API 未返回有效的任务ID。data 内容: {data}")
            
            return response
            
        except requests.exceptions.RequestException as e:
            raise GrsaiAPIError(f"网络请求失败: {str(e)}")
        except json.JSONDecodeError as e:
            raise GrsaiAPIError(f"JSON 解析失败: {str(e)}")
        except Exception as e:
            raise GrsaiAPIError(f"未知错误: {str(e)}")

# --- 异步任务管理 ---
BANANA_TASK_FILE = os.path.join(folder_paths.get_temp_directory(), "banana_task_history.json")
MAX_BANANA_HISTORY_DOWNLOADED = 5 # 最多保留5条 'downloaded' 记录
banana_task_lock = threading.Lock()

# 零宽字符，确保并发提交时 prompt 唯一
ZERO_WIDTH_CHARS = [
    "\u200b", "\u200c", "\u200d", "\ufeff",
    "\u180e", "\u200e", "\u200f",
]

def _read_banana_tasks():
    """读取任务历史文件"""
    if not os.path.exists(BANANA_TASK_FILE): return {}
    try:
        with open(BANANA_TASK_FILE, 'r', encoding='utf-8') as f:
            tasks = json.load(f)
        return tasks if isinstance(tasks, dict) else {}
    except (json.JSONDecodeError, IOError):
        return {}

def _write_banana_tasks(tasks):
    """写入任务历史文件, 并清理旧的 'downloaded' 任务"""
    try:
        downloaded_tasks = [
            (task_id, info) for task_id, info in tasks.items()
            if info.get("status") == "downloaded"
        ]
        
        if len(downloaded_tasks) > MAX_BANANA_HISTORY_DOWNLOADED:
            downloaded_tasks.sort(key=lambda x: x[1].get("submitted_at", "1970-01-01 00:00:00"))
            tasks_to_remove = downloaded_tasks[:-MAX_BANANA_HISTORY_DOWNLOADED]
            
            print(f"[Banana Task Manager] 清理 {len(tasks_to_remove)} 个旧的 'downloaded' 任务。")
            for task_id, _ in tasks_to_remove:
                if task_id in tasks:
                    del tasks[task_id]
        
        with open(BANANA_TASK_FILE, 'w', encoding='utf-8') as f:
            json.dump(tasks, f, indent=4, ensure_ascii=False, sort_keys=True)
    except IOError as e:
        print(f"[Banana Task Manager] 写入任务文件失败: {e}")


def _get_next_task_number():
    """获取下一个任务编号"""
    tasks = _read_banana_tasks()
    existing_nums = []
    for task_key in tasks.keys():
        if task_key.startswith("任务"):
            try:
                num = int(task_key[2:])
                existing_nums.append(num)
            except ValueError:
                continue
    return max(existing_nums, default=0) + 1

# --- 节点 1: NanoBanana 异步提交 ---
class NanoBananaAsyncSubmit(_GrsaiNodeBase):
    CATEGORY = "Nkxx/图像"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
                    "prompt": ("STRING", {"multiline": True, "default": "一只可爱的小猫"}),
                    "concurrency": ("INT", {"default": 1, "min": 1, "max": 10, "step": 1}),
                    "aspect_ratio": (SUPPORTED_ASPECT_RATIOS, {"default": "auto"}),
                }, "optional": {
                    "api_key": ("STRING", {"multiline": False, "default": "", "placeholder": "留空则使用 __init__.py 中的配置"}),
                    "image_1": ("IMAGE",), "image_2": ("IMAGE",),
                    "image_3": ("IMAGE",), "image_4": ("IMAGE",),
                }}
    
    # *** 变更: 只有一个输出 ***
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    FUNCTION = "submit"
    
    @classmethod
    def IS_CHANGED(cls, **kwargs): return float("NaN")
    
    def submit(self, prompt: str, concurrency: int, aspect_ratio: str, api_key: str = "", **kwargs):
        final_api_key = get_api_key(api_key)
        if not final_api_key: 
            # *** 变更: 自定义错误返回以匹配单个输出 ***
            error_msg = "API Key 不能为空。"
            print(f"节点执行错误: {error_msg}")
            return {"ui": {"string": [error_msg]}, "result": (error_msg,)}
        
        images_in = [kwargs.get(f"image_{i}") for i in range(1, 5)]
        
        try:
            # (阻塞) 上传图片并获取URL
            uploaded_urls = self._handle_image_uploads(images_in, final_api_key)
            
            task_num = _get_next_task_number()
            task_id = f"任务{task_num}"
            
            api_client = GrsaiAPI(api_key=final_api_key)
            subtasks = []
            
            # (阻塞) 提交多个子任务
            for i in range(concurrency):
                payload = {
                    "model": "nano-banana-fast",
                    "prompt": f"{prompt}{secrets.choice(ZERO_WIDTH_CHARS) * i}", # 添加唯一字符
                    "aspectRatio": aspect_ratio,
                    "urls": uploaded_urls,
                    "webHook": "-1",
                    "shutProgress": True
                }
                
                # (阻塞) 使用新的错误处理方法
                response = self._make_api_request_with_error_handling(
                    api_client, "/v1/draw/nano-banana", payload
                )
                
                api_task_id = response["data"]["id"]
                
                subtasks.append({
                    "api_task_id": api_task_id,
                    "status": "pending",
                    "image_url": None,
                    "progress": 0,
                    "failure_reason": None
                })
            
            # (快速) 保存任务记录
            with banana_task_lock:
                tasks = _read_banana_tasks()
                tasks[task_id] = {
                    "prompt": prompt, # 保存 prompt
                    "submitted_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "status": "pending",
                    "aspect_ratio": aspect_ratio,
                    "concurrency": concurrency,
                    "subtasks": subtasks
                }
                _write_banana_tasks(tasks)
            
            credits = self._get_credits_balance(final_api_key)
            status_msg = f"任务提交成功 | {task_id} | 子任务数: {concurrency} | 积分: {credits if credits >= 0 else 'N/A'}"
            # *** 变更: 单个输出返回 ***
            return {"ui": {"string": [status_msg]}, "result": (status_msg,)}
            
        except GrsaiAPIError as e:
            # *** 变更: 自定义错误返回 ***
            error_msg = str(e)
            print(f"节点执行错误: {error_msg}")
            return {"ui": {"string": [error_msg]}, "result": (error_msg,)}
        except Exception as e:
            # *** 变更: 自定义错误返回 ***
            error_msg = f"提交失败: {format_error_message(e)}"
            print(f"节点执行错误: {error_msg}")
            return {"ui": {"string": [error_msg]}, "result": (error_msg,)}

# --- 节点 2: NanoBanana 异步查询下载 ---
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
    
    def query_and_download(self, api_key: str = ""):
        final_api_key = get_api_key(api_key)
        if not final_api_key: 
            return self._create_error_result("API Key 不能为空。")
        
        with banana_task_lock:
            tasks = _read_banana_tasks()
        
        if not tasks:
            return self._create_error_result("当前没有任务记录。", is_text_output=True)
        
        # 查找第一个已完成 (succeeded) 且未下载的任务
        task_to_download = None
        task_info_to_download = None
        
        for task_id, task_info in sorted(tasks.items()):
            if task_info.get("status") == "succeeded" and task_info.get("status") != "downloaded":
                subtasks = task_info.get("subtasks", [])
                all_succeeded = all(subtask.get("status") == "succeeded" for subtask in subtasks)
                if subtasks and all_succeeded:
                    task_to_download = task_id
                    task_info_to_download = task_info
                    # print(f"[Banana Query] 发现可下载任务: {task_id}") # 已注释
                    break
        
        # 如果没有可下载的，选择最早未完成的
        if not task_to_download:
            pending_tasks = []
            for tid, tinfo in tasks.items():
                if tinfo.get("status") in ["pending", "running"]:
                    pending_tasks.append((tid, tinfo))
            
            if pending_tasks:
                # 优先处理最早提交的 pending/running 任务
                task_id, task_info = min(pending_tasks, key=lambda x: x[1].get("submitted_at", ""))
                print(f"[Banana Query] 更新最早未完成任务: {task_id}")
            else:
                # 如果没有 pending/running，也没有可下载的，说明都下载完了
                task_id_for_status = list(tasks.keys())[-1] if tasks else "N/A"
                print(f"[Banana Query] 没有待处理或可下载的任务。")
                all_tasks_status = []
                for tid, tinfo in sorted(tasks.items(), key=lambda x: x[1].get('submitted_at', ''), reverse=True):
                    status = tinfo.get('status', 'N/A')
                    concurrency = tinfo.get('concurrency', 1)
                    prompt_full = tinfo.get('prompt', '')
                    prompt_snippet = prompt_full[:15]
                    if len(prompt_full) > 15: prompt_snippet += "..."
                    all_tasks_status.append(f"[{status}] {tid} ({prompt_snippet}) - {concurrency}个子任务")
                status_msg = "没有待处理或可下载的任务。\n所有任务:\n" + "\n".join(all_tasks_status)
                return {"ui": {"string": [status_msg]}, "result": (torch.zeros((1, 1, 1, 3), dtype=torch.float32), status_msg)}
        else:
            task_id = task_to_download
            task_info = task_info_to_download
        
        subtasks = task_info.get("subtasks", [])
        
        if not subtasks:
            return self._create_error_result(f"{task_id} 无子任务记录。", is_text_output=True)
        
        try:
            api_client = GrsaiAPI(api_key=final_api_key)
            
            print(f"[Banana Query] 开始查询任务 {task_id} 的 {len(subtasks)} 个子任务")
            
            all_succeeded = True
            all_failed = True
            any_running = False
            pending_count = 0
            failed_count = 0
            succeeded_count = 0
            
            # (阻塞) 查询所有子任务状态
            for i, subtask in enumerate(subtasks):
                # 如果子任务已经成功，跳过查询
                if subtask.get("status") == "succeeded":
                    succeeded_count += 1
                    all_failed = False
                    continue

                api_task_id = subtask.get("api_task_id")
                if not api_task_id:
                    print(f"[Banana Query] 子任务{i}缺少api_task_id")
                    all_succeeded = False
                    continue
                
                # print(f"[Banana Query] 查询子任务{i+1}: {api_task_id}") # 已注释
                response = api_client._make_request("POST", "/v1/draw/result", 
                                                    data={"id": api_task_id})
                
                # print(f"[Banana Query] API响应: {json.dumps(response)[:200]}...") # 已注释
                
                query_data = response.get("data", {})
                if not query_data:
                    if response.get("code") == -22:
                        print(f"[Banana Query] 任务 {api_task_id} API 尚未找到 (code -22)，将重试。")
                        status = "pending"
                        progress = 0
                        image_url = None
                    else:
                        raise GrsaiAPIError(f"API 响应中没有 'data' 字段。响应: {response}")
                else:
                    status = query_data.get("status", "")
                    progress = query_data.get("progress", 0)
                    image_url = None
                    if query_data.get("results") and len(query_data["results"]) > 0:
                        image_url = query_data["results"][0].get("url")
                
                # print(f"[Banana Query] 子任务{i+1}状态: {status}, 进度: {progress}%, URL: {image_url[:50] if image_url else 'None'}") # 已注释
                
                subtask["status"] = status
                subtask["progress"] = progress
                
                if status == "succeeded":
                    subtask["image_url"] = image_url
                    succeeded_count += 1
                    all_failed = False
                    # print(f"[Banana Query] 子任务{i+1}成功") # 已注释
                elif status == "failed":
                    subtask["failure_reason"] = query_data.get("failure_reason", "未知错误")
                    failed_count += 1
                    all_succeeded = False
                    # print(f"[Banana Query] 子任务{i+1}失败") # 已注释
                elif status == "running":
                    any_running = True
                    all_succeeded = False
                    all_failed = False
                    # print(f"[Banana Query] 子任务{i+1}运行中: {progress}%") # 已注释
                else: # pending
                    pending_count += 1
                    all_succeeded = False
                    all_failed = False
                    # print(f"[Banana Query] 子任务{i+1}等待中") # 已注释
            
            print(f"[Banana Query] 任务{task_id}汇总 - 成功: {succeeded_count}, 失败: {failed_count}, 等待: {pending_count}, 运行中: {any_running}")
            
            # (快速) 更新主任务状态
            with banana_task_lock:
                tasks = _read_banana_tasks()
                if task_id not in tasks: # 检查任务是否已被删除
                    return self._create_error_result(f"任务 {task_id} 在写入时丢失。")

                if all_succeeded:
                    tasks[task_id]["status"] = "succeeded"
                    # print(f"[Banana Query] 任务{task_id}状态更新为: succeeded") # 已注释
                elif all_failed:
                    tasks[task_id]["status"] = "failed"
                    # print(f"[Banana Query] 任务{task_id}状态更新为: failed") # 已注释
                elif any_running:
                    tasks[task_id]["status"] = "running"
                    # print(f"[Banana Query] 任务{task_id}状态更新为: running") # 已注释
                else:
                    tasks[task_id]["status"] = "pending"
                    # print(f"[Banana Query] 任务{task_id}状态更新为: pending") # 已注释
                
                tasks[task_id]["subtasks"] = subtasks
                _write_banana_tasks(tasks) # 写入时会触发清理
                # print(f"[Banana Query] 任务状态已写入文件") # 已注释
            
            all_tasks_status = []
            for tid, tinfo in sorted(tasks.items(), key=lambda x: x[1].get('submitted_at', ''), reverse=True):
                status = tinfo.get('status', 'N/A')
                concurrency = tinfo.get('concurrency', 1)
                prompt_full = tinfo.get('prompt', '')
                prompt_snippet = prompt_full[:15]
                if len(prompt_full) > 15:
                    prompt_snippet += "..."
                
                if status == 'running':
                    subtasks_list = tinfo.get('subtasks', [])
                    if subtasks_list:
                        valid_subtasks = [s for s in subtasks_list if 'progress' in s]
                        if valid_subtasks:
                            avg_progress = sum(s.get('progress', 0) for s in valid_subtasks) / len(valid_subtasks)
                            status_str = f"running {int(avg_progress)}%"
                        else:
                            status_str = "running 0%"
                    else:
                        status_str = "running"
                else:
                    status_str = status
                
                all_tasks_status.append(f"[{status_str}] {tid} ({prompt_snippet}) - {concurrency}个子任务")
            
            # (阻塞) 如果所有子任务都成功，批量下载
            if all_succeeded:
                print(f"[Banana Query] 任务{task_id}全部成功，开始下载图片")
                pil_images = []
                for i, subtask in enumerate(subtasks):
                    image_url = subtask.get("image_url")
                    if image_url:
                        # print(f"[Banana Query] 下载子任务{i+1}图片: {image_url[:50]}...") # 已注释
                        pil_image = download_image(image_url)
                        if pil_image:
                            pil_images.append(pil_image)
                            # print(f"[Banana Query] 子任务{i+1}图片下载成功") # 已注释
                        else:
                            print(f"[Banana Query] 子任务{i+1}图片下载失败")
                    else:
                        print(f"[Banana Query] 子任务{i+1}无图片URL")
                
                if not pil_images:
                    return self._create_error_result(f"{task_id} 所有图片下载失败。", is_text_output=True)
                
                # (快速) 标记为已下载
                with banana_task_lock:
                    tasks = _read_banana_tasks()
                    tasks[task_id]["status"] = "downloaded"
                    _write_banana_tasks(tasks) # 写入时会触发清理
                    # print(f"[Banana Query] 任务{task_id}标记为已下载") # 已注释
                
                credits = self._get_credits_balance(final_api_key)
                # 更新 all_tasks_status 以反映 "downloaded"
                for i, task_str in enumerate(all_tasks_status):
                    if task_str.startswith(f"[succeeded] {task_id}"):
                        all_tasks_status[i] = task_str.replace("[succeeded]", "[downloaded]", 1)
                        break
                
                status_msg = f"下载成功: {task_id} | 子任务: {len(subtasks)}个 | 积分: {credits if credits >= 0 else 'N/A'}\n所有任务:\n" + "\n".join(all_tasks_status)
                return {"ui": {"string": [status_msg]}, "result": (safe_pil_batch_to_tensor(pil_images), status_msg)}
            
            elif all_failed:
                status_msg = f"任务失败: {task_id} | 成功: {succeeded_count} | 失败: {failed_count} | 等待: {pending_count}\n所有任务:\n" + "\n".join(all_tasks_status)
                return {"ui": {"string": [status_msg]}, "result": (torch.zeros((1, 1, 1, 3), dtype=torch.float32), status_msg)}
            
            else:
                status_msg = f"任务进行中: {task_id} | 成功: {succeeded_count} | 失败: {failed_count} | 等待: {pending_count}\n所有任务:\n" + "\n".join(all_tasks_status)
                return {"ui": {"string": [status_msg]}, "result": (torch.zeros((1, 1, 1, 3), dtype=torch.float32), status_msg)}
                
        except Exception as e:
            error_msg = f"查询失败: {format_error_message(e)}"
            print(f"[Banana Query] {error_msg}")
            traceback.print_exc()
            return self._create_error_result(error_msg, is_text_output=True)

# --- 节点注册 ---
NODE_CLASS_MAPPINGS = {
    "NanoBananaAsyncSubmit": NanoBananaAsyncSubmit,
    "NanoBananaAsyncQuery": NanoBananaAsyncQuery,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NanoBananaAsyncSubmit": "🍌 Nano Banana 异步提交",
    "NanoBananaAsyncQuery": "🍌 Nano Banana 异步查询下载",
}