# banana_nodes.py 
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
import folder_paths
import re

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

# --- 安全打包函数 ---
def safe_pil_batch_to_tensor(pil_images: List[Image.Image]) -> torch.Tensor:
    """将PIL图像列表安全地转换为ComfyUI图像张量，自动处理不同尺寸。"""
    if not pil_images:
        return torch.empty((0, 1, 1, 3), dtype=torch.float32)

    tensors = []
    max_h = 0
    max_w = 0
    
    # Pass 1: 转换并找到最大尺寸
    for pil_image in pil_images:
        if pil_image is None: continue
        try:
            img_array = np.array(pil_image).astype(np.float32) / 255.0
            tensor = torch.from_numpy(img_array)[None,] # (1, H, W, C)
            if tensor.shape[1] > max_h: max_h = tensor.shape[1]
            if tensor.shape[2] > max_w: max_w = tensor.shape[2]
            tensors.append(tensor)
        except Exception as e:
            print(f"Warning: 跳过损坏的图像: {e}")
            continue

    if not tensors: return torch.empty((0, 1, 1, 3), dtype=torch.float32)
    
    # Pass 2: 填充至最大尺寸
    padded_tensors = []
    for tensor in tensors:
        b, h, w, c = tensor.shape
        
        if h == 0 or w == 0 or c < 3:
            print(f"Warning: 跳过无效尺寸的张量: shape {tensor.shape}")
            continue
            
        if h == max_h and w == max_w:
            padded_tensors.append(tensor)
            continue
        
        # (B, H, W, C) -> (B, C, H, W)
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

def upload_file_zh(file_path: str = "") -> str:
    api_key = os.getenv("GRSAI_KEY")
    if not file_path or not api_key: return ""
    if not os.path.exists(file_path): raise FileNotFoundError(f"文件不存在: {file_path}")
    file_extension = os.path.splitext(file_path)[1].lstrip(".") or "png"
    result = get_upload_token_zh(api_key, {"sux": file_extension})
    token, key, url, domain = (result["data"]["token"], result["data"]["key"], result["data"]["url"], result["data"]["domain"])
    with open(file_path, "rb") as file:
        upload_response = requests.post(url=url, data={"token": token, "key": key}, files={"file": file}, timeout=120)
        upload_response.raise_for_status()
        return f"{domain}/{key}"

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

    def nano_banana_generate_image(self, prompt: str, model: str, urls: List[str], aspectRatio: str, imageSize: str = "1K") -> Tuple[List[Image.Image], List[str], List[str]]:
        payload = {"model": model, "prompt": prompt, "urls": urls, "shutProgress": True, "aspectRatio": aspectRatio}
        
        # 仅在模型为 pro 或 pro-vt 时添加 imageSize 参数
        if model in ["nano-banana-pro", "nano-banana-pro-vt"]:
            payload["imageSize"] = imageSize

        response = self._make_request("POST", "/v1/draw/nano-banana", data=payload)
        if response.get("status") != "succeeded":
            raise GrsaiAPIError(f"图像生成失败: {response.get('error', '未知错误')}")
        resultsUrls = [r["url"] for r in response.get("results", []) if "url" in r]
        if not resultsUrls: raise GrsaiAPIError("API未返回有效的图像URL")
        
        pil_images, image_urls, errors = [], [], []
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(resultsUrls)) as executor:
            for img, url in executor.map(lambda u: (download_image(u), u), resultsUrls):
                if img:
                    pil_images.append(img)
                    image_urls.append(url)
                else:
                    errors.append(f"下载失败: {url}")
        return pil_images, image_urls, errors

# --- 配置 ---
SUPPORTED_ASPECT_RATIOS = ["auto", "1:1", "16:9", "9:16", "4:3", "3:4", "3:2", "2:3", "5:4", "4:5", "21:9"]
# 更新模型列表
SUPPORTED_MODELS = ["nano-banana-fast", "nano-banana-pro", "nano-banana-pro-vt"]

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
        
    def _handle_image_uploads(self, images_in: List[Optional[torch.Tensor]]):
        uploaded_urls, temp_files = [], []
        if not any(img is not None for img in images_in): return uploaded_urls, temp_files
        try:
            for i, image_tensor in enumerate(images_in):
                if image_tensor is None: continue
                pil_image = tensor_to_pil(image_tensor)[0]
                rgb_pil = safe_pil_to_rgb(pil_image)
                with tempfile.NamedTemporaryFile(suffix=f"_{i}.png", delete=False) as temp_file:
                    rgb_pil.save(temp_file, "PNG"); temp_files.append(temp_file.name)
            
            for path in temp_files:
                uploaded_urls.append(upload_file_zh(path))
            
            if not uploaded_urls: return {"error": "提供了输入图像，但无法处理或上传。"}, temp_files
            return uploaded_urls, temp_files
        except Exception as e:
            return {"error": f"图像上传失败: {format_error_message(e)}"}, temp_files

    def _cleanup_temp_files(self, temp_files: List[str]):
        for path in temp_files:
            if os.path.exists(path): os.unlink(path)

# --- 节点 1: GrsaiNanoBanana (普通节点：8张图) ---
class GrsaiNanoBanana(_GrsaiNodeBase):
    CATEGORY = "Nkxx/图像"
    @classmethod
    def INPUT_TYPES(cls):
        inputs = {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": "一只可爱的小猫"}),
                "model": (SUPPORTED_MODELS, {"default": "nano-banana-fast"}),
                "concurrency": ("INT", {"default": 1, "min": 1, "max": 10, "step": 1}),
                "aspect_ratio": (SUPPORTED_ASPECT_RATIOS, {"default": "auto"}),
            },
            "optional": {
                "api_key": ("STRING", {"multiline": False, "default": "", "placeholder": "留空则使用 __init__.py 中的配置"}),
            }
        }
        for i in range(1, 9):
            inputs["optional"][f"image_{i}"] = ("IMAGE",)
        return inputs

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "status")
    
    def execute(self, prompt: str, model: str, concurrency: int, aspect_ratio: str, api_key: str = "", **kwargs):
        final_api_key = get_api_key(api_key)
        if not final_api_key: return self._create_error_result("API Key 不能为空。")
        os.environ["GRSAI_KEY"] = final_api_key
        
        images_in = [kwargs.get(f"image_{i}") for i in range(1, 9)]
        uploaded_urls, temp_files = self._handle_image_uploads(images_in)
        if isinstance(uploaded_urls, dict):
            self._cleanup_temp_files(temp_files)
            return self._create_error_result(uploaded_urls["error"])

        try:
            api_client = GrsaiAPI(api_key=final_api_key)
            all_pil_images, all_errors = [], []
            
            def submit_task(_):
                try:
                    target_size = "2K" if model in ["nano-banana-pro", "nano-banana-pro-vt"] else "1K"
                    pils, _, errs = api_client.nano_banana_generate_image(
                        prompt, model, uploaded_urls, aspect_ratio, imageSize=target_size
                    )
                    return (pils, errs)
                except Exception as e:
                    return ([], [format_error_message(e)])

            with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
                results = executor.map(submit_task, range(concurrency))
                for pils, errs in results:
                    if pils: all_pil_images.extend(pils)
                    if errs: all_errors.extend(errs)
            
            if not all_pil_images: return self._create_error_result(f"所有图像生成均失败: {'; '.join(all_errors)}")
            credits = self._get_credits_balance(final_api_key)
            status = f"成功: {len(all_pil_images)} | 失败: {len(all_errors)} | 积分: {credits if credits >=0 else 'N/A'}"
            return {"ui": {"string": [status]}, "result": (pil_to_tensor(all_pil_images), status)}
        except Exception as e:
            return self._create_error_result(f"执行时发生意外错误: {format_error_message(e)}")
        finally:
            self._cleanup_temp_files(temp_files)


# --- 节点: GrsaiNanoBananaPro (Pro节点：14张图) ---
class GrsaiNanoBananaPro(_GrsaiNodeBase):
    CATEGORY = "Nkxx/图像"
    @classmethod
    def INPUT_TYPES(cls):
        inputs = {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": "一只可爱的小猫"}),
                "model": (["nano-banana-pro", "nano-banana-pro-vt"], {"default": "nano-banana-pro"}),
                "concurrency": ("INT", {"default": 1, "min": 1, "max": 10, "step": 1}),
                "aspect_ratio": (SUPPORTED_ASPECT_RATIOS, {"default": "auto"}),
                "image_size": (["1K", "2K", "4K"], {"default": "2K"}),
            },
            "optional": {
                "api_key": ("STRING", {"multiline": False, "default": "", "placeholder": "留空则使用 __init__.py 中的配置"}),
            }
        }
        for i in range(1, 15):
            inputs["optional"][f"image_{i}"] = ("IMAGE",)
        return inputs

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "status")
    
    def execute(self, prompt: str, model: str, concurrency: int, aspect_ratio: str, image_size: str, api_key: str = "", **kwargs):
        final_api_key = get_api_key(api_key)
        if not final_api_key: return self._create_error_result("API Key 不能为空。")
        os.environ["GRSAI_KEY"] = final_api_key
        
        images_in = [kwargs.get(f"image_{i}") for i in range(1, 15)]
        uploaded_urls, temp_files = self._handle_image_uploads(images_in)
        if isinstance(uploaded_urls, dict):
            self._cleanup_temp_files(temp_files)
            return self._create_error_result(uploaded_urls["error"])

        try:
            api_client = GrsaiAPI(api_key=final_api_key)
            all_pil_images, all_errors = [], []
            
            def submit_task(_):
                try:
                    # 使用传入的 model 参数，而不是硬编码
                    pils, _, errs = api_client.nano_banana_generate_image(
                        prompt, model, uploaded_urls, aspect_ratio, imageSize=image_size
                    )
                    return (pils, errs)
                except Exception as e:
                    return ([], [format_error_message(e)])

            with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
                results = executor.map(submit_task, range(concurrency))
                for pils, errs in results:
                    if pils: all_pil_images.extend(pils)
                    if errs: all_errors.extend(errs)
            
            if not all_pil_images: return self._create_error_result(f"所有图像生成均失败: {'; '.join(all_errors)}")
            credits = self._get_credits_balance(final_api_key)
            status = f"成功: {len(all_pil_images)} | 失败: {len(all_errors)} | 积分: {credits if credits >=0 else 'N/A'}"
            return {"ui": {"string": [status]}, "result": (pil_to_tensor(all_pil_images), status)}
        except Exception as e:
            return self._create_error_result(f"执行时发生意外错误: {format_error_message(e)}")
        finally:
            self._cleanup_temp_files(temp_files)


# --- 节点 2: GrsaiNanoBananaBatch (批量，支持模型选择和分辨率) ---
class GrsaiNanoBananaBatch(GrsaiNanoBanana):
    CATEGORY = "Nkxx/图像"
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
                "file_path": ("STRING", {"default": "", "placeholder": "拖拽 CSV/Excel 文件至此"}),
                "column_name": ("STRING", {"default": "prompt"}),
                "prompt_prefix": ("STRING", {"multiline": True, "default": ""}),
                "model": (SUPPORTED_MODELS, {"default": "nano-banana-fast"}),
                "concurrency": ("INT", {"default": 10, "min": 1, "max": 50, "step": 1}),
                "max_count": ("INT", {"default": 50, "min": 1, "max": 100}),
                "aspect_ratio": (SUPPORTED_ASPECT_RATIOS, {"default": "auto"}),
                "image_size": (["默认", "1K", "2K", "4K"], {"default": "默认"}),
                "executions_per_prompt": ("INT", {"default": 1, "min": 1, "max": 10, "step": 1, "label": "单提示词执行次数"}),
            }, "optional": {
                "api_key": ("STRING", {"multiline": False, "default": "", "placeholder": "留空则使用 __init__.py 中的配置"}),
                "image_1": ("IMAGE",), "image_2": ("IMAGE",),
                "image_3": ("IMAGE",), "image_4": ("IMAGE",),
            }}

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images_batch", "status")

    def execute(self, file_path: str, column_name: str, prompt_prefix: str, model: str, concurrency: int, max_count: int, aspect_ratio: str, image_size: str, executions_per_prompt: int, api_key: str = "", **kwargs):
        final_api_key = get_api_key(api_key)
        if not final_api_key: return self._create_error_result("API Key 不能为空。")
        os.environ["GRSAI_KEY"] = final_api_key

        if not file_path or not os.path.exists(file_path):
            return self._create_error_result("文件路径为空或文件不存在。")
        
        try:
            if file_path.lower().endswith('.csv'): df = pd.read_csv(file_path, encoding='utf-8')
            elif file_path.lower().endswith(('.xls', '.xlsx')): df = pd.read_excel(file_path)
            else: return self._create_error_result("仅支持 .csv, .xls, .xlsx 文件。")
        except Exception as e: return self._create_error_result(f"读取文件失败: {format_error_message(e)}")
        if column_name not in df.columns: return self._create_error_result(f"列 '{column_name}' 不存在。")
        
        base_prompts = [f"{prompt_prefix}{p}" for p in df[column_name].dropna().astype(str).tolist()[:max_count]]
        if not base_prompts: return self._create_error_result(f"列 '{column_name}' 中未找到有效 prompt。")

        prompts = [p for p in base_prompts for _ in range(max(1, executions_per_prompt))]

        images_in = [kwargs.get(f"image_{i}") for i in range(1, 5)]
        uploaded_urls, temp_files = self._handle_image_uploads(images_in)
        if isinstance(uploaded_urls, dict):
            self._cleanup_temp_files(temp_files)
            return self._create_error_result(uploaded_urls["error"])
        
        if image_size == "默认":
            final_image_size = "2K" if model in ["nano-banana-pro", "nano-banana-pro-vt"] else "1K"
        else:
            final_image_size = image_size

        try:
            api_client, all_pil_images, errors = GrsaiAPI(api_key=final_api_key), [], []
            
            def submit_task(prompt):
                try:
                    pils, _, errs = api_client.nano_banana_generate_image(
                        prompt, model, uploaded_urls, aspect_ratio, imageSize=final_image_size
                    )
                    return (pils, errs)
                except Exception as e:
                    return ([], [format_error_message(e)])

            with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
                results = executor.map(submit_task, prompts)
                for pils, errs in results:
                    if pils: all_pil_images.extend(pils)
                    if errs: errors.extend(errs)
            
            if not all_pil_images: return self._create_error_result(f"所有图像生成均失败: {'; '.join(errors)}")
            credits = self._get_credits_balance(final_api_key)
            if executions_per_prompt == 1:
                task_info = f"{len(prompts)}个总任务"
            else:
                task_info = f"{len(base_prompts)}个Prompt x {executions_per_prompt}次 = {len(prompts)}个总任务"
            status = f"批量完成 | {task_info} | 成功: {len(all_pil_images)} | 失败: {len(errors)} | 积分: {credits if credits >=0 else 'N/A'}"
            return {"ui": {"string": [status]}, "result": (safe_pil_batch_to_tensor(all_pil_images), status)}
        except Exception as e:
            return self._create_error_result(f"批量执行时发生意外错误: {format_error_message(e)}")
        finally:
            self._cleanup_temp_files(temp_files)

# --- 节点 3: GrsaiNanoBananaSaveWithPrompt (命名细化版) ---
class GrsaiNanoBananaSaveWithPrompt(_GrsaiNodeBase):
    CATEGORY = "Nkxx/图像"
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
                "file_path": ("STRING", {"default": "", "placeholder": "拖拽 CSV/Excel 文件至此"}),
                "column_name": ("STRING", {"default": "prompt"}),
                "prompt_prefix": ("STRING", {"multiline": True, "default": ""}),
                "filename_prefix": ("STRING", {"default": "GrsaiBanana"}),
                "rm_prompt_prefix": ("STRING", {"multiline": False, "default": ""}),
                "rm_prompt_suffix": ("STRING", {"multiline": False, "default": ""}),
                "model": (SUPPORTED_MODELS, {"default": "nano-banana-fast"}), 
                "concurrency": ("INT", {"default": 10, "min": 1, "max": 50, "step": 1}),
                "max_count": ("INT", {"default": 50, "min": 1, "max": 100}),
                "aspect_ratio": (SUPPORTED_ASPECT_RATIOS, {"default": "auto"}),
                "image_size": (["默认", "1K", "2K", "4K"], {"default": "默认"}), 
                "ui_display_mode": (["保存 (Saved)", "预览 (Preview)"], {"default": "保存 (Saved)"}),
            }, "optional": { "api_key": ("STRING", {"multiline": False, "default": "", "placeholder": "留空则使用 __init__.py 中的配置"}),
                "image_1": ("IMAGE",), "image_2": ("IMAGE",), "image_3": ("IMAGE",), "image_4": ("IMAGE",),
            }}

    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("image", "status", "filenames")

    def execute(self, file_path: str, column_name: str, prompt_prefix: str, filename_prefix: str, 
                  rm_prompt_prefix: str, rm_prompt_suffix: str, 
                  model: str, concurrency: int, max_count: int, aspect_ratio: str, image_size: str, 
                  ui_display_mode: str, 
                  api_key: str = "", **kwargs):
        final_api_key = get_api_key(api_key)
        if not final_api_key: return self._create_error_result("API Key 不能为空。")
        os.environ["GRSAI_KEY"] = final_api_key

        if not file_path or not os.path.exists(file_path):
            return self._create_error_result("文件路径为空或文件不存在。")
        
        try:
            if file_path.lower().endswith('.csv'): df = pd.read_csv(file_path, encoding='utf-8')
            elif file_path.lower().endswith(('.xls', '.xlsx')): df = pd.read_excel(file_path)
            else: return self._create_error_result("仅支持 .csv, .xls, .xlsx 文件。")
        except Exception as e: return self._create_error_result(f"读取文件失败: {format_error_message(e)}")
        if column_name not in df.columns: return self._create_error_result(f"列 '{column_name}' 不存在。")
        
        prompts = [f"{prompt_prefix}{p}" for p in df[column_name].dropna().astype(str).tolist()[:max_count]]
        if not prompts: return self._create_error_result(f"列 '{column_name}' 中未找到有效 prompt。")

        images_in = [kwargs.get(f"image_{i}") for i in range(1, 5)]
        uploaded_urls, temp_files = self._handle_image_uploads(images_in)
        if isinstance(uploaded_urls, dict):
            self._cleanup_temp_files(temp_files)
            return self._create_error_result(uploaded_urls["error"])
        
        if image_size == "默认":
            final_image_size = "2K" if model in ["nano-banana-pro", "nano-banana-pro-vt"] else "1K"
        else:
            final_image_size = image_size

        try:
            api_client = GrsaiAPI(api_key=final_api_key)
            all_pil_images, ui_image_info, saved_filenames, errors, saved_count = [], [], [], [], 0
            
            def submit_task(prompt):
                try:
                    pils, _, errs = api_client.nano_banana_generate_image(
                        prompt, model, uploaded_urls, aspect_ratio, imageSize=final_image_size
                    )
                    return (pils, errs)
                except Exception as e:
                    return ([], [format_error_message(e)])
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
                main_output_dir = folder_paths.get_output_directory()
                temp_dir = folder_paths.get_temp_directory() 

                results = executor.map(submit_task, prompts)
                
                for original_prompt, (pils, errs) in zip(prompts, results):
                    if errs: errors.extend(errs)
                    if pils:
                        prompt_for_filename = original_prompt
                        if rm_prompt_prefix and prompt_for_filename.startswith(rm_prompt_prefix):
                            prompt_for_filename = prompt_for_filename[len(rm_prompt_prefix):]
                        if rm_prompt_suffix and prompt_for_filename.endswith(rm_prompt_suffix):
                            prompt_for_filename = prompt_for_filename[:-len(rm_prompt_suffix)]
                        base_filename = sanitize_filename(prompt_for_filename)
                        subfolder, actual_prefix = os.path.split(filename_prefix)

                        for pil_image in pils:
                            rgb_pil = safe_pil_to_rgb(pil_image)
                            all_pil_images.append(rgb_pil) 
                            
                            filename_no_ext = f"{actual_prefix}_{base_filename}"
                            extension = ".png"
                            final_output_dir = os.path.join(main_output_dir, subfolder)
                            os.makedirs(final_output_dir, exist_ok=True)
                            
                            final_path = os.path.join(final_output_dir, filename_no_ext + extension)
                            counter = 1
                            while os.path.exists(final_path):
                                final_path = os.path.join(final_output_dir, f"{filename_no_ext} ({counter}){extension}")
                                counter += 1
                            
                            saved_filenames.append(final_path)

                            if ui_display_mode == "保存 (Saved)":
                                rgb_pil.save(final_path, "PNG", compress_level=4)
                                saved_count += 1
                                ui_filename = os.path.relpath(final_path, main_output_dir)
                                ui_image_info.append({"filename": ui_filename, "subfolder": subfolder, "type": "output"})
                            
                            else: # "预览 (Preview)"
                                os.makedirs(temp_dir, exist_ok=True)
                                temp_filename = os.path.basename(final_path)
                                temp_path = os.path.join(temp_dir, temp_filename)
                                temp_counter = 1
                                while os.path.exists(temp_path):
                                    base, ext = os.path.splitext(temp_path)
                                    base = re.sub(r' \(\d+\)$', '', base)
                                    temp_path = os.path.join(temp_dir, f"{base} ({temp_counter}){ext}")
                                    temp_counter += 1
                                rgb_pil.save(temp_path, "PNG", compress_level=4)
                                ui_image_info.append({"filename": os.path.basename(temp_path), "subfolder": "", "type": "temp"})
            
            if not all_pil_images:
                return self._create_error_result(f"所有图像生成均失败。错误: {'; '.join(errors)}")
            credits = self._get_credits_balance(final_api_key)
            if ui_display_mode == "保存 (Saved)":
                status = f"批量完成 | 总Prompt: {len(prompts)} | 成功保存: {saved_count} | 失败: {len(errors)} | 积分: {credits if credits >=0 else 'N/A'}"
            else:
                status = f"批量预览 | 总Prompt: {len(prompts)} | 成功预览: {len(all_pil_images)} | 失败: {len(errors)} | 积分: {credits if credits >=0 else 'N/A'}"
            return {
                "ui": {"string": [status], "images": ui_image_info},
                "result": (safe_pil_batch_to_tensor(all_pil_images), status, "\n".join(saved_filenames))
            }
        except Exception as e:
            traceback.print_exc()
            return self._create_error_result(f"执行时发生意外错误: {format_error_message(e)}")
        finally:
            self._cleanup_temp_files(temp_files)
            
# --- 节点 4: GrsaiLLMWriter ---
class GrsaiLLMWriter(_GrsaiNodeBase):
    CATEGORY = "Nkxx/语言模型"
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
                "model": (["gemini-3-flash", "gemini-3-pro", "gemini-2.5-flash", "gemini-2.5-flash-lite"], {"default": "gemini-3-flash"}),
                "main_prompt": ("STRING", {"multiline": True, "default": "请为我生成5条关于“夏日海滩”的Midjourney绘画prompt"}),
                "system_prompt": ("STRING", {"default": "You are a helpful assistant.", "multiline": True}),
                "output_filename": ("STRING", {"default": "generated_prompts.csv"}),
                "column_name": ("STRING", {"default": "prompt"}),
            }, "optional": {
                "api_key": ("STRING", {"multiline": False, "default": "", "placeholder": "留空则使用 __init__.py 中的配置"}),
                "image_1": ("IMAGE",), "image_2": ("IMAGE",), "image_3": ("IMAGE",),
            }}

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("file_path", "status")

    def llm_api_call(self, api_key, model, messages):
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        
        headers = {
            "Content-Type": "application/json", 
            "Authorization": f"Bearer {api_key}",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"
        }
        
        payload = {"model": model, "messages": messages, "stream": False}
        url = "https://grsai.dakka.com.cn/v1/chat/completions" 
        
        session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(max_retries=3)
        session.mount('https://', adapter)
        
        try:
            response = session.post(url, headers=headers, json=payload, timeout=180, verify=False)
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"].strip(), None
        except Exception as e:
            return None, str(e)

    def execute(self, model: str, main_prompt: str, system_prompt: str, output_filename: str, column_name: str, api_key: str = "", **kwargs):
        final_api_key = get_api_key(api_key)
        if not final_api_key: return self._create_error_result("API Key 不能为空。", is_text_output=True)
        
        messages = []
        if system_prompt.strip(): messages.append({"role": "system", "content": system_prompt.strip()})
        
        user_content_list = [{"type": "text", "text": main_prompt.strip()}]
        images_in = [kwargs.get(f"image_{i}") for i in range(1, 4)]
        if any(img is not None for img in images_in):
            for image_tensor in images_in:
                if image_tensor is None: continue
                pil_img = tensor_to_pil(image_tensor)[0]
                rgb_pil = safe_pil_to_rgb(pil_img)
                buffered = BytesIO(); rgb_pil.save(buffered, format="JPEG")
                img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
                user_content_list.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}})
        messages.append({"role": "user", "content": user_content_list})
        
        try:
            llm_response, error = self.llm_api_call(final_api_key, model, messages)
            if error: return self._create_error_result(f"LLM API 调用失败: {error}", is_text_output=True)
            
            try:
                json_match = re.search(r'\[.*\]', llm_response, re.DOTALL)
                parsed_list = json.loads(json_match.group()) if json_match else [l.strip() for l in llm_response.split('\n') if l.strip()]
            except (json.JSONDecodeError, TypeError):
                 parsed_list = [line.strip() for line in llm_response.split('\n') if llm_response.strip()]

            parsed_list = [item for item in parsed_list if item]
            if not parsed_list or not isinstance(parsed_list, list): return self._create_error_result(f"LLM未返回有效列表。收到: {llm_response}", is_text_output=True)
            
            df = pd.DataFrame(parsed_list, columns=[column_name.strip()])
            
            output_dir = folder_paths.get_output_directory()
            os.makedirs(output_dir, exist_ok=True)
            full_path = os.path.join(output_dir, output_filename.strip())

            if full_path.lower().endswith('.csv'): df.to_csv(full_path, index=False, encoding='utf-8-sig')
            elif full_path.lower().endswith(('.xls', '.xlsx')): df.to_excel(full_path, index=False)
            else: return self._create_error_result("文件名必须以 .csv, .xls, 或 .xlsx 结尾。", is_text_output=True)
            
            status = f"成功生成 {len(parsed_list)} 条记录并写入: {output_filename}"
            return {"ui": {"string": [status]}, "result": (full_path, status)}
        except Exception as e:
            traceback.print_exc()
            return self._create_error_result(f"执行或写入文件失败: {format_error_message(e)}", is_text_output=True)

# --- 智能保存节点 ---
class SaveImageByFilename(_GrsaiNodeBase):
    CATEGORY = "Nkxx/图像"
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "filenames": ("STRING", {"multiline": True}),
            },
            "optional": {
                "output_path": ("STRING", {"default": ""}),
                "suffix": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ()

    def execute(self, images, filenames, output_path="", suffix=""):
        pil_images = tensor_to_pil(images)
        paths = [p.strip() for p in filenames.strip().split('\n') if p.strip()]

        if not paths:
            print("Warning: SaveImageByFilename 节点收到空的 'filenames' 输入。")
            return {"ui": {"string": ["错误: 文件名列表为空"]}}
            
        if len(pil_images) != len(paths):
            print(f"Warning: 图像数量 ({len(pil_images)}) 与文件名数量 ({len(paths)}) 不匹配。将只保存较少数量的图像。")

        main_output_dir = folder_paths.get_output_directory()
        saved_count = 0
        ui_image_info = []

        for i, pil_image in enumerate(pil_images):
            if i >= len(paths): break 

            original_path = paths[i]
            
            if output_path:
                final_save_dir = os.path.join(main_output_dir, output_path)
                original_filename = os.path.basename(original_path)
                base, ext = os.path.splitext(original_filename)
            else:
                final_save_dir = os.path.dirname(original_path)
                original_filename = os.path.basename(original_path)
                base, ext = os.path.splitext(original_filename)
            
            if not ext.lower() == ".png": ext = ".png"

            os.makedirs(final_save_dir, exist_ok=True)
            
            new_filename_no_ext = f"{base}{suffix}"
            new_path = os.path.join(final_save_dir, new_filename_no_ext + ext)
            
            counter = 1
            while os.path.exists(new_path):
                new_path = os.path.join(final_save_dir, f"{new_filename_no_ext} ({counter}){ext}")
                counter += 1

            pil_image.save(new_path, "PNG", compress_level=4)
            saved_count += 1
            
            ui_filename = os.path.relpath(new_path, main_output_dir)
            ui_subfolder = os.path.dirname(ui_filename)
            ui_image_info.append({"filename": os.path.basename(ui_filename), "subfolder": ui_subfolder, "type": "output"})

        return {"ui": {"images": ui_image_info, "string": [f"成功保存 {saved_count} 张高清图"]}}

# --- 节点 5: GrsaiNanoBananaBatchDir (全功能增强版：文件夹+CSV+重试+积分熔断+UI优化) ---
class GrsaiNanoBananaBatchDir(_GrsaiNodeBase):
    CATEGORY = "Nkxx/图像"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
                "directory_path": ("STRING", {"default": "", "placeholder": "输入图片文件夹路径"}),
                "csv_file_path": ("STRING", {"default": "", "placeholder": "填写 CSV/Excel 文件路径至此"}),
                "column_name": ("STRING", {"default": "prompt", "label": "CSV列名"}),
                "model": (SUPPORTED_MODELS, {"default": "nano-banana-fast"}),
                "max_concurrent_files": ("INT", {"default": 3, "min": 1, "max": 100, "step": 1, "label": "同时处理文件数"}),
                "aspect_ratio": (SUPPORTED_ASPECT_RATIOS, {"default": "auto"}),
                "image_size": (["默认", "1K", "2K", "4K"], {"default": "默认"}),
                "executions_per_prompt": ("INT", {"default": 1, "min": 1, "max": 10, "step": 1, "label": "单条Prompt执行次数"}),
            }, "optional": {
                "fixed_prompt": ("STRING", {"multiline": True, "default": "", "placeholder": "备用：如果不用CSV，则在此处填提示词跑所有图"}),
                "api_key": ("STRING", {"default": "", "placeholder": "留空则使用 __init__.py 中的配置"}),
            }}

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images_preview_15", "status_report")

    # --- 辅助：检查图片是否损坏 ---
    def _is_valid_image(self, file_path):
        try:
            if os.path.getsize(file_path) == 0: return False
            with Image.open(file_path) as img:
                img.verify() # 快速校验文件头
            return True
        except:
            return False

    def process_and_save_single_file(self, image_idx, file_path, prompts, output_dir, model, aspect_ratio, image_size, final_api_key, need_preview):
        # [防线1] 坏图预检
        if not self._is_valid_image(file_path):
            return 0, [], [f"文件损坏或非图片: {os.path.basename(file_path)}"]

        base_filename = os.path.splitext(os.path.basename(file_path))[0]
        base_filename = sanitize_filename(base_filename) 

        try:
            # 上传
            uploaded_url = upload_file_zh(file_path)
            if not uploaded_url:
                return 0, [], [f"{base_filename} 上传失败"]
            
            banana_refs = [uploaded_url]
            api_client = GrsaiAPI(api_key=final_api_key)
            inner_workers = min(len(prompts), 5) 
            
            file_success_count = 0
            file_pils = []
            file_errors = []

            def strict_task_with_retry(prompt_idx, p_text):
                # [防线2] 内部重试机制 (Max 3次)
                last_error = ""
                for attempt in range(3):
                    try:
                        pils, _, errs = api_client.nano_banana_generate_image(
                            p_text, model, banana_refs, aspect_ratio, imageSize=image_size
                        )
                        if not pils:
                            if errs: last_error = errs[0]
                            time.sleep(1) # 稍微冷却
                            continue # 重试

                        saved_local = 0
                        saved_pils = []
                        
                        for j, img in enumerate(pils):
                            prefix = f"Img{image_idx+1:03d}_{base_filename}"
                            
                            # === [修改开始] 智能命名逻辑 ===
                            # 如果单次生成多于1张，才加后缀 _1, _2
                            if len(pils) > 1:
                                suffix = f"P{prompt_idx+1:03d}_{j+1}"
                            else:
                                suffix = f"P{prompt_idx+1:03d}" # 只有1张时，去掉 _1
                            # === [修改结束] ===

                            save_name = f"{prefix}_{suffix}.png"
                            save_path = os.path.join(output_dir, save_name)
                            
                            # 处理同名文件（防止覆盖）
                            counter = 1
                            while os.path.exists(save_path):
                                save_path = os.path.join(output_dir, f"{prefix}_{suffix}_{counter}.png")
                                counter += 1
                            
                            try:
                                img.save(save_path, "PNG", compress_level=4)
                                saved_local += 1
                                saved_pils.append(img)
                            except Exception as save_err:
                                print(f"保存失败: {save_err}")

                        return saved_local, saved_pils, [] # 成功返回
                    except Exception as e:
                        # [紧急防线] 如果是积分耗尽，直接返回特殊标记，不重试
                        err_str = str(e).lower()
                        if "credit" in err_str or "balance" in err_str:
                            return 0, [], [f"CRITICAL_NO_CREDITS: {str(e)}"]

                        last_error = str(e)
                        time.sleep(1)
                
                # 3次都失败
                return 0, [], [f"RetryFailed: {last_error}"]

            with concurrent.futures.ThreadPoolExecutor(max_workers=inner_workers) as inner_exc:
                futures = [inner_exc.submit(strict_task_with_retry, i, p) for i, p in enumerate(prompts)]
                for f in concurrent.futures.as_completed(futures):
                    cnt, ret_pils, errs = f.result()
                    file_success_count += cnt
                    if errs: file_errors.append(errs[0])
                    if need_preview and len(file_pils) < 15:
                        file_pils.extend(ret_pils)

            if file_success_count == 0 and file_errors:
                return 0, [], [f"{base_filename}: {file_errors[0]}"]
            
            return file_success_count, file_pils, []

        except Exception as e:
            traceback.print_exc()
            return 0, [], [f"{base_filename} 异常: {str(e)[:50]}"]

    def execute(self, directory_path, csv_file_path, column_name, model, max_concurrent_files, aspect_ratio, image_size, executions_per_prompt, fixed_prompt="", api_key=""):
        final_api_key = get_api_key(api_key)
        if not final_api_key: return self._create_error_result("API Key 不能为空。")
        os.environ["GRSAI_KEY"] = final_api_key

        # 1. 验证
        if not directory_path or not os.path.exists(directory_path):
            return self._create_error_result("文件夹路径不存在")

        valid_exts = {'.png', '.jpg', '.jpeg', '.webp', '.bmp'}
        image_files = [os.path.join(directory_path, f) for f in os.listdir(directory_path) 
                       if os.path.splitext(f)[1].lower() in valid_exts]
        image_files.sort()

        if not image_files: return self._create_error_result("文件夹内无有效图片")

        # 2. 准备 Prompts
        base_prompts = []
        source_mode = "Fixed"
        if csv_file_path and os.path.exists(csv_file_path):
            try:
                if csv_file_path.lower().endswith('.csv'): 
                    df = pd.read_csv(csv_file_path, encoding='utf-8')
                elif csv_file_path.lower().endswith(('.xls', '.xlsx')): 
                    df = pd.read_excel(csv_file_path)
                else: 
                    df = pd.DataFrame()
                if column_name in df.columns:
                    base_prompts = df[column_name].dropna().astype(str).tolist()
                    source_mode = "CSV"
            except Exception: pass

        if not base_prompts:
            if not fixed_prompt.strip():
                return self._create_error_result("必须提供有效 CSV 或 固定提示词。")
            base_prompts = [fixed_prompt]
            source_mode = "Fixed Prompt"

        final_prompts = [p for p in base_prompts for _ in range(executions_per_prompt)]

        # 3. 输出路径
        import datetime
        base_output = folder_paths.get_output_directory()
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        batch_output_dir = os.path.join(base_output, f"Banana_Batch_{timestamp}")
        os.makedirs(batch_output_dir, exist_ok=True)

        final_image_size = image_size
        if image_size == "默认":
            final_image_size = "2K" if model in ["nano-banana-pro", "nano-banana-pro-vt"] else "1K"

        # 4. 执行 (含熔断机制)
        failed_list = []   
        total_success_imgs = 0
        preview_pil_images = []
        all_file_logs = [] 
        
        all_file_logs.append(f"=== Banana Batch Log ({timestamp}) ===")
        all_file_logs.append(f"Source: {source_mode}, Prompts: {len(base_prompts)}, Images: {len(image_files)}")
        
        # [防线3] 熔断计数器
        consecutive_failures = 0
        MAX_FAILURES = 10 
        ABORT_FLAG = False

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_concurrent_files) as executor:
            futures_map = {} # {future: idx}
            
            for idx, f in enumerate(image_files):
                if ABORT_FLAG: break
                
                need_preview = (len(preview_pil_images) < 15)
                future = executor.submit(
                    self.process_and_save_single_file, 
                    idx, f, final_prompts, batch_output_dir, model, aspect_ratio, final_image_size, final_api_key, need_preview
                )
                futures_map[future] = os.path.basename(f)
            
            for future in concurrent.futures.as_completed(futures_map):
                fname = futures_map[future]
                
                if ABORT_FLAG:
                    future.cancel()
                    continue

                try:
                    count, pils, errs = future.result()
                    if count > 0:
                        total_success_imgs += count
                        consecutive_failures = 0 # 成功则重置计数器
                        all_file_logs.append(f"[√] {fname} 生成 {count} 张")
                        if pils and len(preview_pil_images) < 15:
                            remain = 15 - len(preview_pil_images)
                            preview_pil_images.extend(pils[:remain])
                    else:
                        consecutive_failures += 1
                        err_msg = errs[0] if errs else "0 output"
                        all_file_logs.append(f"[X] {fname} {err_msg}")
                        failed_list.append(f"{fname} -> {err_msg}")
                        
                        # [紧急熔断] 如果是积分问题，直接触发熔断
                        if "CRITICAL_NO_CREDITS" in err_msg:
                            consecutive_failures = MAX_FAILURES
                            print("🚨 错误：积分耗尽，立即终止任务！")
                        
                        # 检查熔断
                        if consecutive_failures >= MAX_FAILURES:
                            ABORT_FLAG = True
                            fail_msg = "🚨 触发熔断保护：连续失败过多次或积分耗尽，停止后续任务。"
                            all_file_logs.append(fail_msg)
                            failed_list.append("BATCH ABORTED (Circuit Breaker)")
                            print(fail_msg)
                            for f_pending in futures_map:
                                f_pending.cancel()
                            
                except Exception as e:
                    failed_list.append(f"{fname} -> 错误: {str(e)[:50]}")

        # 5. 报告 (UI 逻辑修正版)
        cred = self._get_credits_balance(final_api_key)
        total_tasks = len(image_files) * len(final_prompts)
        
        ui_summary = [
            "📢 【重要提示】预览图如果含黑边仅为兼容显示，实际保存的原图无黑边！",
            f"✅ 批量任务完成" if not ABORT_FLAG else "❌ 任务异常终止 (熔断)",
            f"📁 输入: {os.path.basename(directory_path)}",
            f"📄 Prompts: {len(base_prompts)}",
            f"🔢 总任务: {total_tasks}",
            f"🖼️ 成功: {total_success_imgs}",
            f"❌ 失败: {len(failed_list)}",
            f"💰 积分: {cred}",
            f"💾 路径: {batch_output_dir}",
            f"⚠️ 预览仅显示前 15 张，全量图请查看文件夹。"
        ]
        
        if failed_list:
            ui_summary.append("\n--- 失败记录 (前5个) ---")
            for fail in failed_list[:5]:
                ui_summary.append(f"• {fail}")
            if len(failed_list) > 5:
                ui_summary.append(f"...以及其他 {len(failed_list)-5} 个文件")
            
            log_path = os.path.join(batch_output_dir, "batch_error_log.txt")
            with open(log_path, "w", encoding="utf-8") as f:
                f.write("\n".join(all_file_logs))
                f.write("\n\n=== 失败详情 ===\n")
                f.write("\n".join(failed_list))
        else:
            ui_summary.append("\n✨ 完美！所有文件均处理成功。")
        
        final_ui_text = "\n".join(ui_summary)
        print(f"Batch Finished. Success: {total_success_imgs}, Failed: {len(failed_list)}")
        
        if not preview_pil_images:
            final_tensor = torch.zeros((1, 1, 1, 3), dtype=torch.float32)
        else:
            final_tensor = safe_pil_batch_to_tensor(preview_pil_images)
        
        return {"ui": {"string": [final_ui_text]}, "result": (final_tensor, final_ui_text)}

# --- 节点注册 ---
NODE_CLASS_MAPPINGS = {
    "GrsaiNanoBanana": GrsaiNanoBanana,
    "GrsaiNanoBananaBatch": GrsaiNanoBananaBatch,
    "GrsaiNanoBananaSaveWithPrompt": GrsaiNanoBananaSaveWithPrompt,
    "GrsaiNanoBananaPro": GrsaiNanoBananaPro, 
    "GrsaiLLMWriter": GrsaiLLMWriter,
    "SaveImageByFilename": SaveImageByFilename,
    "GrsaiNanoBananaBatchDir": GrsaiNanoBananaBatchDir,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GrsaiNanoBanana": "🍌 Grsai Nano Banana",
    "GrsaiNanoBananaBatch": "🍌 Grsai Nano Banana Batch (CSV/Excel)",
    "GrsaiNanoBananaSaveWithPrompt": "🍌 Grsai Nano Banana 命名细化版",
    "GrsaiNanoBananaPro": "🍌 Grsai Nano Banana Pro (2K/4K)", 
    "GrsaiLLMWriter": "✍️ Grsai LLM/VLM Writer",
    "SaveImageByFilename": "💾 Save Image By Filename (智能保存)",
    "GrsaiNanoBananaBatchDir": "🍌📂 Grsai 文件夹批量处理(CSV)",
}