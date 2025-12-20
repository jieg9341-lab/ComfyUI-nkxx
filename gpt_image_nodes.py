# gptimage_nodes.py
import torch
import torch.nn.functional as F
import os
import requests
import concurrent.futures
import json
import numpy as np
import time
import traceback
from io import BytesIO
from typing import Any, Dict, Optional, Union, List, Tuple
from PIL import Image

# 尝试从当前包导入 get_api_key
try:
    from . import get_api_key
except ImportError:
    def get_api_key(key_in_node):
        return key_in_node or os.getenv("GRSAI_KEY", "")

# --- 辅助工具函数 ---

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

def tensor_to_pil(tensor: torch.Tensor) -> List[Image.Image]:
    """将torch张量转换为PIL图像列表"""
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

def safe_pil_to_rgb(image: Image.Image) -> Image.Image:
    """安全地将任何PIL图像转换为RGB模式"""
    if image.mode == 'RGBA':
        background = Image.new("RGB", image.size, (255, 255, 255))
        background.paste(image, mask=image.split()[3])
        return background
    elif image.mode == 'RGB':
        return image
    else:
        return image.convert('RGB')

# --- 上传功能 ---
def get_upload_token_zh(api_key: str, data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    url = "https://grsai.dakka.com.cn/client/resource/newUploadTokenZH"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    response = requests.post(url=url, headers=headers, json=data or {}, timeout=30)
    response.raise_for_status()
    return response.json()

def upload_file_zh(file_path: str = "") -> str:
    """上传文件并返回URL"""
    api_key = os.getenv("GRSAI_KEY")
    if not file_path or not api_key: return ""
    if not os.path.exists(file_path): raise FileNotFoundError(f"文件不存在: {file_path}")
    
    for attempt in range(3):
        try:
            file_extension = os.path.splitext(file_path)[1].lstrip(".") or "png"
            result = get_upload_token_zh(api_key, {"sux": file_extension})
            token, key, url, domain = (result["data"]["token"], result["data"]["key"], result["data"]["url"], result["data"]["domain"])
            with open(file_path, "rb") as file:
                upload_response = requests.post(url=url, data={"token": token, "key": key}, files={"file": file}, timeout=120)
                upload_response.raise_for_status()
                return f"{domain}/{key}"
        except Exception as e:
            if attempt == 2: raise e
            time.sleep(1)
    return ""

# --- API 客户端 ---
class GrsaiAPIError(Exception): pass

class GrsaiAPI:
    def __init__(self, api_key: str):
        if not api_key or not api_key.strip(): raise GrsaiAPIError("API密钥不能为空")
        self.api_key = api_key
        self.session = requests.Session()
        self.session.headers.update({
            "Content-Type": "application/json; charset=utf-8", 
            "User-Agent": "ComfyUI-GptImage/1.0", 
            "Authorization": f"Bearer {self.api_key}"
        })
        self.base_url = "https://grsai.dakka.com.cn"

    def _make_request(self, method: str, endpoint: str, data: Optional[Dict] = None, timeout: int = 300) -> Dict[str, Any]:
        """通用请求处理"""
        url = f"{self.base_url}{endpoint}"
        try:
            # 同步请求，设置较长超时时间 (300秒)
            response = self.session.request(method, url, json=data, timeout=timeout)
            response.raise_for_status()
            
            # 处理可能的 SSE 格式残留 (虽然 shutProgress=True 应该返回纯 JSON，但做个防御)
            text = response.text.strip()
            if text.startswith("data: "):
                text = text[6:]
            
            return json.loads(text)
        except json.JSONDecodeError:
            raise GrsaiAPIError(f"API返回了无法解析的数据: {response.text[:100]}...")
        except Exception as e:
            raise GrsaiAPIError(f"API请求失败: {str(e)}")

    def gpt_image_generate(self, prompt: str, urls: List[str], size: str, variants: int) -> Tuple[List[Image.Image], List[str], List[str]]:
        """
        同步生成图片
        shutProgress=True -> 等待接口直接返回结果
        """
        payload = {
            "model": "sora-image",
            "prompt": prompt,
            "urls": urls,
            "size": size,
            "variants": variants,
            "shutProgress": True, # 关闭进度推送，直接等待最终结果
            # 不传 webHook
        }

        # 调用接口
        data = self._make_request("POST", "/v1/draw/completions", data=payload, timeout=300)

        # 检查业务状态码
        if data.get("code") and data.get("code") != 0:
             raise GrsaiAPIError(f"API 错误: {data.get('msg')}")

        # 解析结果兼容性处理
        results_info = []
        
        # 优先查找标准 results 数组
        if "results" in data and isinstance(data["results"], list):
            results_info = data["results"]
        # 其次查找 data.results (有些接口包裹在 data 字段里)
        elif "data" in data and isinstance(data["data"], dict) and "results" in data["data"]:
            results_info = data["data"]["results"]
        # 再次查找 data.url
        elif "data" in data and isinstance(data["data"], dict) and "url" in data["data"]:
            results_info = [{"url": data["data"]["url"]}]
        # 最后查找根目录 url
        elif "url" in data:
            results_info = [{"url": data["url"]}]

        if not results_info:
             raise GrsaiAPIError(f"未在响应中找到图片URL: {data}")

        target_urls = [r["url"] for r in results_info if "url" in r]
        
        # 下载图片
        pil_images = []
        download_errors = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(target_urls) or 1) as executor:
            future_to_url = {executor.submit(download_image, url): url for url in target_urls}
            for future in concurrent.futures.as_completed(future_to_url):
                url = future_to_url[future]
                try:
                    img = future.result()
                    if img:
                        pil_images.append(img)
                    else:
                        download_errors.append(f"下载失败: {url}")
                except Exception as e:
                    download_errors.append(f"下载异常 {url}: {e}")
        
        return pil_images, target_urls, download_errors

# --- 节点基类 ---
class _GrsaiNodeBase:
    FUNCTION = "execute"
    @classmethod
    def IS_CHANGED(cls, **kwargs): return float("NaN")

    def _create_error_result(self, error_message: str):
        print(f"节点执行错误: {error_message}")
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
        import tempfile
        uploaded_urls, temp_files = [], []
        if not any(img is not None for img in images_in): return uploaded_urls, temp_files
        try:
            for i, image_tensor in enumerate(images_in):
                if image_tensor is None: continue
                pil_list = tensor_to_pil(image_tensor)
                for j, pil_image in enumerate(pil_list):
                    rgb_pil = safe_pil_to_rgb(pil_image)
                    with tempfile.NamedTemporaryFile(suffix=f"_{i}_{j}.png", delete=False) as temp_file:
                        rgb_pil.save(temp_file, "PNG")
                        temp_files.append(temp_file.name)
            
            for path in temp_files:
                uploaded_urls.append(upload_file_zh(path))
            
            if not uploaded_urls: return {"error": "提供了输入图像，但无法处理或上传。"}, temp_files
            return uploaded_urls, temp_files
        except Exception as e:
            return {"error": f"图像上传失败: {str(e)}"}, temp_files

    def _cleanup_temp_files(self, temp_files: List[str]):
        for path in temp_files:
            try:
                if os.path.exists(path): os.unlink(path)
            except: pass

# --- 主要节点: GrsaiGptImage ---
class GrsaiGptImage(_GrsaiNodeBase):
    CATEGORY = "Nkxx/图像"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": "A cute cat running on the grass"}),
                "size": (["auto", "1:1", "3:2", "2:3"], {"default": "1:1"}),
                "variants": ("INT", {"default": 1, "min": 1, "max": 2, "step": 1, "label": "单次生成数量(1-2)"}),
                "concurrency": ("INT", {"default": 1, "min": 1, "max": 5, "step": 1, "label": "并发任务数"}),
            },
            "optional": {
                "api_key": ("STRING", {"multiline": False, "default": "", "placeholder": "留空则使用环境变量 GRSAI_KEY"}),
                "image_1": ("IMAGE",),
                "image_2": ("IMAGE",),
                "image_3": ("IMAGE",),
                "image_4": ("IMAGE",),
                "image_5": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "status")
    
    def execute(self, prompt: str, size: str, variants: int, concurrency: int, api_key: str = "", **kwargs):
        # 1. API Key 处理
        final_api_key = get_api_key(api_key)
        if not final_api_key: 
            return self._create_error_result("API Key 不能为空。请配置 GRSAI_KEY 或填入节点。")
        os.environ["GRSAI_KEY"] = final_api_key

        # 2. 参考图处理 (最多5张)
        images_in = [kwargs.get(f"image_{i}") for i in range(1, 6)]
        uploaded_urls, temp_files = self._handle_image_uploads(images_in)
        
        # 错误检查
        if isinstance(uploaded_urls, dict) and "error" in uploaded_urls:
            self._cleanup_temp_files(temp_files)
            return self._create_error_result(uploaded_urls["error"])

        # 3. 执行任务 (同步等待)
        try:
            api_client = GrsaiAPI(api_key=final_api_key)
            all_pil_images = []
            all_errors = []

            # 定义单个任务函数
            def task_runner(_):
                try:
                    pils, _, errs = api_client.gpt_image_generate(
                        prompt=prompt,
                        urls=uploaded_urls,
                        size=size,
                        variants=variants
                    )
                    return pils, errs
                except Exception as e:
                    return [], [str(e)]

            # 并发执行
            # 这里的并发是指：如果 concurrency=3，会同时发出3个HTTP请求，
            # 它们在后台同时等待(sleep)，直到各自返回。
            with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
                results = executor.map(task_runner, range(concurrency))
                for pils, errs in results:
                    if pils: all_pil_images.extend(pils)
                    if errs: all_errors.extend(errs)

            # 4. 结果处理
            if not all_pil_images:
                err_msg = "; ".join(all_errors) if all_errors else "未知错误"
                return self._create_error_result(f"生成失败: {err_msg}")

            final_tensor = pil_to_tensor(all_pil_images)
            credits = self._get_credits_balance(final_api_key)
            
            status_text = (
                f"成功: {len(all_pil_images)} 张 (Task x{concurrency}, Var x{variants}) | "
                f"失败: {len(all_errors)} | 积分余额: {credits}"
            )
            
            return {"ui": {"string": [status_text]}, "result": (final_tensor, status_text)}

        except Exception as e:
            traceback.print_exc()
            return self._create_error_result(f"系统错误: {str(e)}")
        finally:
            self._cleanup_temp_files(temp_files)

# --- 节点注册 ---
NODE_CLASS_MAPPINGS = {
    "GrsaiGptImage": GrsaiGptImage,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GrsaiGptImage": "🤖 Grsai GPT Image (Sora-Image)",
}