# grsai/banana_nodes.py
import os
import re
import json
import base64
import time
import traceback
import concurrent.futures
import pandas as pd
import requests
from io import BytesIO
import torch
import folder_paths
from PIL import Image

# 动态获取全局配置的 API Key
import builtins

# 从我们强大的核心工具箱导入公共函数
from ..utils import (
    download_image, 
    safe_pil_to_rgb, 
    tensor_to_pil, 
    pil_to_tensor, 
    safe_pil_batch_to_tensor, 
    upload_image_grsai
)

# --- 工具函数 ---
def format_error_message(error: Exception) -> str:
    return f"{type(error).__name__}: {str(error)}"

def sanitize_filename(text: str, max_length: int = 100) -> str:
    sanitized = re.sub(r'[\\/*?:"<>|]', '_', text)
    sanitized = re.sub(r'[\s_]+', '_', sanitized)
    sanitized = sanitized.strip('_')
    return sanitized[:max_length]

# --- API 客户端 ---
class GrsaiAPIError(Exception): pass

class GrsaiAPI:
    def __init__(self, api_key: str):
        if not api_key or not api_key.strip(): 
            raise GrsaiAPIError("API密钥不能为空")
        self.api_key = api_key
        self.session = requests.Session()
        self.session.headers.update({
            "Content-Type": "application/json; charset=utf-8", 
            "User-Agent": "ComfyUI-Nkxx/2.0", 
            "Authorization": f"Bearer {self.api_key}"
        })

    def _make_request(self, method: str, endpoint: str, data: dict = None, timeout: int = 300) -> dict:
        url = f"https://grsai.dakka.com.cn{endpoint}"
        response = self.session.request(method, url, json=data, timeout=timeout)
        response.raise_for_status()
        text = response.text
        json_data = text[6:] if text.startswith("data: ") else text
        return json.loads(json_data)

    def nano_banana_generate_image(self, prompt: str, model: str, urls: list[str], aspectRatio: str, imageSize: str = "1K") -> tuple[list[Image.Image], list[str], list[str]]:
        payload = {
            "model": model, 
            "prompt": prompt, 
            "urls": urls, 
            "shutProgress": True, 
            "aspectRatio": aspectRatio
        }
        
        # Pro 系列及 2.0 模型支持分辨率参数
        if model in ["nano-banana-pro", "nano-banana-pro-vt", "nano-banana-pro-cl", "nano-banana-2", "nano-banana-2-cl", "nano-banana-2-4k-cl"]:
            payload["imageSize"] = imageSize

        response = self._make_request("POST", "/v1/draw/nano-banana", data=payload)
        if response.get("status") != "succeeded":
            raise GrsaiAPIError(f"图像生成失败: {response.get('error', '未知错误')}")
            
        resultsUrls = [r["url"] for r in response.get("results", []) if "url" in r]
        if not resultsUrls: 
            raise GrsaiAPIError("API未返回有效的图像URL")
        
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
# 增加了 nano-banana-2-cl
SUPPORTED_MODELS = ["nano-banana-fast", "nano-banana-pro", "nano-banana-pro-vt", "nano-banana-pro-cl", "nano-banana-2", "nano-banana-2-cl"]

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
        
    def _handle_image_uploads(self, images_in: list[torch.Tensor], api_key: str):
        """调用工具库进行图片上传，返回 URL 列表"""
        uploaded_urls = []
        if not any(img is not None for img in images_in): 
            return uploaded_urls
            
        try:
            for image_tensor in images_in:
                if image_tensor is None: continue
                url = upload_image_grsai(api_key, image_tensor)
                if url:
                    uploaded_urls.append(url)
                else:
                    return {"error": "部分图片上传失败，请检查网络或 API Key"}
            return uploaded_urls
        except Exception as e:
            return {"error": f"图像上传失败: {format_error_message(e)}"}

# ======================================================================
# 节点 1: GrsaiNanoBanana (统一的基础/Pro融合版：14张图)
# ======================================================================
class GrsaiNanoBanana(_GrsaiNodeBase):
    CATEGORY = "Nkxx/Grsai/图像"
    
    @classmethod
    def INPUT_TYPES(cls):
        inputs = {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": "一只可爱的小猫"}),
                "model": (SUPPORTED_MODELS, {"default": "nano-banana-pro"}),
                "concurrency": ("INT", {"default": 1, "min": 1, "max": 10, "step": 1}),
                "aspect_ratio": (SUPPORTED_ASPECT_RATIOS, {"default": "auto"}),
                "image_size": (["默认", "1K", "2K", "4K"], {"default": "默认"}),
            },
            "optional": {
                "api_key": ("STRING", {"multiline": False, "default": "", "placeholder": "留空则读取全局配置"}),
            }
        }
        for i in range(1, 15):
            inputs["optional"][f"image_{i}"] = ("IMAGE",)
        return inputs

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "status")
    
    def execute(self, prompt: str, model: str, concurrency: int, aspect_ratio: str, image_size: str, api_key: str = "", **kwargs):
        final_api_key = builtins.get_api_key(api_key, "grsai")
        if not final_api_key: return self._create_error_result("未配置 Grsai API Key。")
        
        images_in = [kwargs.get(f"image_{i}") for i in range(1, 15)]
        uploaded_urls = self._handle_image_uploads(images_in, final_api_key)
        
        if isinstance(uploaded_urls, dict):
            return self._create_error_result(uploaded_urls["error"])

        target_size = image_size
        if image_size == "默认":
            target_size = "2K" if model in ["nano-banana-pro", "nano-banana-pro-vt", "nano-banana-pro-cl", "nano-banana-2", "nano-banana-2-cl"] else "1K"

        # 分辨率动态映射逻辑
        actual_model = model
        if model == "nano-banana-2-cl" and target_size == "4K":
            actual_model = "nano-banana-2-4k-cl"

        try:
            api_client = GrsaiAPI(api_key=final_api_key)
            all_pil_images, all_errors = [], []
            
            def submit_task(_):
                try:
                    pils, _, errs = api_client.nano_banana_generate_image(
                        prompt, actual_model, uploaded_urls, aspect_ratio, imageSize=target_size
                    )
                    return (pils, errs)
                except Exception as e:
                    return ([], [format_error_message(e)])

            with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
                results = executor.map(submit_task, range(concurrency))
                for pils, errs in results:
                    if pils: all_pil_images.extend(pils)
                    if errs: all_errors.extend(errs)
            
            if not all_pil_images: 
                return self._create_error_result(f"所有图像生成均失败: {'; '.join(all_errors)}")
                
            credits_left = self._get_credits_balance(final_api_key)
            status = f"成功: {len(all_pil_images)} | 失败: {len(all_errors)} | 积分: {credits_left if credits_left >=0 else 'N/A'}"
            
            return {"ui": {"string": [status]}, "result": (pil_to_tensor(all_pil_images), status)}
        except Exception as e:
            return self._create_error_result(f"执行时发生意外错误: {format_error_message(e)}")

# ======================================================================
# 节点 2: GrsaiNanoBananaBatch (CSV/Excel 批量版)
# ======================================================================
class GrsaiNanoBananaBatch(_GrsaiNodeBase):
    CATEGORY = "Nkxx/Grsai/图像"
    
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
                "api_key": ("STRING", {"multiline": False, "default": "", "placeholder": "留空则读取全局配置"}),
                "image_1": ("IMAGE",), "image_2": ("IMAGE",),
                "image_3": ("IMAGE",), "image_4": ("IMAGE",),
            }}

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images_batch", "status")

    def execute(self, file_path: str, column_name: str, prompt_prefix: str, model: str, concurrency: int, max_count: int, aspect_ratio: str, image_size: str, executions_per_prompt: int, api_key: str = "", **kwargs):
        final_api_key = builtins.get_api_key(api_key, "grsai")
        if not final_api_key: return self._create_error_result("未配置 Grsai API Key。")

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
        uploaded_urls = self._handle_image_uploads(images_in, final_api_key)
        
        if isinstance(uploaded_urls, dict):
            return self._create_error_result(uploaded_urls["error"])
        
        target_size = image_size
        if image_size == "默认":
            target_size = "2K" if model in ["nano-banana-pro", "nano-banana-pro-vt", "nano-banana-pro-cl", "nano-banana-2", "nano-banana-2-cl"] else "1K"

        # 分辨率动态映射逻辑
        actual_model = model
        if model == "nano-banana-2-cl" and target_size == "4K":
            actual_model = "nano-banana-2-4k-cl"

        try:
            api_client, all_pil_images, errors = GrsaiAPI(api_key=final_api_key), [], []
            
            def submit_task(prompt):
                try:
                    pils, _, errs = api_client.nano_banana_generate_image(
                        prompt, actual_model, uploaded_urls, aspect_ratio, imageSize=target_size
                    )
                    return (pils, errs)
                except Exception as e:
                    return ([], [format_error_message(e)])

            with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
                results = executor.map(submit_task, prompts)
                for pils, errs in results:
                    if pils: all_pil_images.extend(pils)
                    if errs: errors.extend(errs)
            
            if not all_pil_images: 
                return self._create_error_result(f"所有图像生成均失败: {'; '.join(errors)}")
                
            credits_left = self._get_credits_balance(final_api_key)
            task_info = f"{len(base_prompts)}个Prompt x {executions_per_prompt}次 = {len(prompts)}个总任务" if executions_per_prompt > 1 else f"{len(prompts)}个总任务"
            status = f"批量完成 | {task_info} | 成功: {len(all_pil_images)} | 失败: {len(errors)} | 积分: {credits_left if credits_left >=0 else 'N/A'}"
            
            return {"ui": {"string": [status]}, "result": (safe_pil_batch_to_tensor(all_pil_images), status)}
        except Exception as e:
            return self._create_error_result(f"批量执行时发生意外错误: {format_error_message(e)}")

# ======================================================================
# 节点 3: GrsaiNanoBananaSaveWithPrompt (命名细化版)
# ======================================================================
class GrsaiNanoBananaSaveWithPrompt(_GrsaiNodeBase):
    CATEGORY = "Nkxx/Grsai/图像"
    
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
            }, "optional": { 
                "api_key": ("STRING", {"multiline": False, "default": "", "placeholder": "留空则读取全局配置"}),
                "image_1": ("IMAGE",), "image_2": ("IMAGE",), "image_3": ("IMAGE",), "image_4": ("IMAGE",),
            }}

    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("image", "status", "filenames")

    def execute(self, file_path: str, column_name: str, prompt_prefix: str, filename_prefix: str, 
                  rm_prompt_prefix: str, rm_prompt_suffix: str, 
                  model: str, concurrency: int, max_count: int, aspect_ratio: str, image_size: str, 
                  ui_display_mode: str, api_key: str = "", **kwargs):
                  
        final_api_key = builtins.get_api_key(api_key, "grsai")
        if not final_api_key: return self._create_error_result("未配置 Grsai API Key。")

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
        uploaded_urls = self._handle_image_uploads(images_in, final_api_key)
        
        if isinstance(uploaded_urls, dict):
            return self._create_error_result(uploaded_urls["error"])
        
        target_size = image_size
        if image_size == "默认":
            target_size = "2K" if model in ["nano-banana-pro", "nano-banana-pro-vt", "nano-banana-pro-cl", "nano-banana-2", "nano-banana-2-cl"] else "1K"

        # 分辨率动态映射逻辑
        actual_model = model
        if model == "nano-banana-2-cl" and target_size == "4K":
            actual_model = "nano-banana-2-4k-cl"

        try:
            api_client = GrsaiAPI(api_key=final_api_key)
            all_pil_images, ui_image_info, saved_filenames, errors, saved_count = [], [], [], [], 0
            
            def submit_task(prompt):
                try:
                    pils, _, errs = api_client.nano_banana_generate_image(
                        prompt, actual_model, uploaded_urls, aspect_ratio, imageSize=target_size
                    )
                    return (pils, errs)
                except Exception as e:
                    return ([], [format_error_message(e)])
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
                import tempfile
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
                
            credits_left = self._get_credits_balance(final_api_key)
            if ui_display_mode == "保存 (Saved)":
                status = f"批量完成 | 总Prompt: {len(prompts)} | 成功保存: {saved_count} | 失败: {len(errors)} | 积分: {credits_left if credits_left >=0 else 'N/A'}"
            else:
                status = f"批量预览 | 总Prompt: {len(prompts)} | 成功预览: {len(all_pil_images)} | 失败: {len(errors)} | 积分: {credits_left if credits_left >=0 else 'N/A'}"
                
            return {
                "ui": {"string": [status], "images": ui_image_info},
                "result": (safe_pil_batch_to_tensor(all_pil_images), status, "\n".join(saved_filenames))
            }
        except Exception as e:
            traceback.print_exc()
            return self._create_error_result(f"执行时发生意外错误: {format_error_message(e)}")

# ======================================================================
# 节点 4: GrsaiLLMWriter (生成 Prompt)
# ======================================================================
class GrsaiLLMWriter(_GrsaiNodeBase):
    CATEGORY = "Nkxx/Grsai/语言模型"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
                "model": (["gemini-3-flash", "gemini-3-pro", "gemini-3.1-pro", "gemini-2.5-flash", "gemini-2.5-flash-lite"], {"default": "gemini-3-flash"}),
                "main_prompt": ("STRING", {"multiline": True, "default": "请为我生成5条关于“夏日海滩”的Midjourney绘画prompt"}),
                "system_prompt": ("STRING", {"default": "You are a helpful assistant.", "multiline": True}),
                "output_filename": ("STRING", {"default": "generated_prompts.csv"}),
                "column_name": ("STRING", {"default": "prompt"}),
            }, "optional": {
                "api_key": ("STRING", {"multiline": False, "default": "", "placeholder": "留空则读取全局配置"}),
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
            "User-Agent": "Mozilla/5.0"
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
        final_api_key = builtins.get_api_key(api_key, "grsai")
        if not final_api_key: return self._create_error_result("未配置 Grsai API Key。", is_text_output=True)
        
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
            if not parsed_list or not isinstance(parsed_list, list): 
                return self._create_error_result(f"LLM未返回有效列表。收到: {llm_response}", is_text_output=True)
            
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


# ======================================================================
# 节点 5: GrsaiNanoBananaBatchDir (全功能增强版)
# ======================================================================
class GrsaiNanoBananaBatchDir(_GrsaiNodeBase):
    CATEGORY = "Nkxx/Grsai/图像"
    
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
                "api_key": ("STRING", {"default": "", "placeholder": "留空则读取全局配置"}),
            }}

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images_preview_15", "status_report")

    def _is_valid_image(self, file_path):
        try:
            if os.path.getsize(file_path) == 0: return False
            with Image.open(file_path) as img:
                img.verify() 
            return True
        except:
            return False

    def process_and_save_single_file(self, image_idx, file_path, prompts, output_dir, model, aspect_ratio, image_size, final_api_key, need_preview):
        if not self._is_valid_image(file_path):
            return 0, [], [f"文件损坏或非图片: {os.path.basename(file_path)}"]

        base_filename = os.path.splitext(os.path.basename(file_path))[0]
        base_filename = sanitize_filename(base_filename) 

        try:
            # === [修改点] 直接调用 utils.py 的上传方法 ===
            with Image.open(file_path) as img:
                uploaded_url = upload_image_grsai(final_api_key, img)
                
            if not uploaded_url:
                return 0, [], [f"{base_filename} 上传失败"]
            
            banana_refs = [uploaded_url]
            api_client = GrsaiAPI(api_key=final_api_key)
            inner_workers = min(len(prompts), 5) 
            
            file_success_count = 0
            file_pils = []
            file_errors = []

            def strict_task_with_retry(prompt_idx, p_text):
                last_error = ""
                for attempt in range(3):
                    try:
                        pils, _, errs = api_client.nano_banana_generate_image(
                            p_text, model, banana_refs, aspect_ratio, imageSize=image_size
                        )
                        if not pils:
                            if errs: last_error = errs[0]
                            time.sleep(1) 
                            continue 

                        saved_local = 0
                        saved_pils = []
                        
                        for j, img in enumerate(pils):
                            prefix = f"Img{image_idx+1:03d}_{base_filename}"
                            if len(pils) > 1:
                                suffix = f"P{prompt_idx+1:03d}_{j+1}"
                            else:
                                suffix = f"P{prompt_idx+1:03d}" 

                            save_name = f"{prefix}_{suffix}.png"
                            save_path = os.path.join(output_dir, save_name)
                            
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

                        return saved_local, saved_pils, [] 
                    except Exception as e:
                        err_str = str(e).lower()
                        if "credit" in err_str or "balance" in err_str:
                            return 0, [], [f"CRITICAL_NO_CREDITS: {str(e)}"]
                        last_error = str(e)
                        time.sleep(1)
                
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
        final_api_key = builtins.get_api_key(api_key, "grsai")
        if not final_api_key: return self._create_error_result("未配置 Grsai API Key。")

        if not directory_path or not os.path.exists(directory_path):
            return self._create_error_result("文件夹路径不存在")

        valid_exts = {'.png', '.jpg', '.jpeg', '.webp', '.bmp'}
        image_files = [os.path.join(directory_path, f) for f in os.listdir(directory_path) 
                       if os.path.splitext(f)[1].lower() in valid_exts]
        image_files.sort()

        if not image_files: return self._create_error_result("文件夹内无有效图片")

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

        import datetime
        base_output = folder_paths.get_output_directory()
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        batch_output_dir = os.path.join(base_output, f"Banana_Batch_{timestamp}")
        os.makedirs(batch_output_dir, exist_ok=True)

        target_size = image_size
        if image_size == "默认":
            target_size = "2K" if model in ["nano-banana-pro", "nano-banana-pro-vt", "nano-banana-pro-cl", "nano-banana-2", "nano-banana-2-cl"] else "1K"

        # 分辨率动态映射逻辑
        actual_model = model
        if model == "nano-banana-2-cl" and target_size == "4K":
            actual_model = "nano-banana-2-4k-cl"

        failed_list = []   
        total_success_imgs = 0
        preview_pil_images = []
        all_file_logs = [] 
        
        all_file_logs.append(f"=== Banana Batch Log ({timestamp}) ===")
        all_file_logs.append(f"Source: {source_mode}, Prompts: {len(base_prompts)}, Images: {len(image_files)}")
        
        consecutive_failures = 0
        MAX_FAILURES = 10 
        ABORT_FLAG = False

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_concurrent_files) as executor:
            futures_map = {} 
            
            for idx, f in enumerate(image_files):
                if ABORT_FLAG: break
                
                need_preview = (len(preview_pil_images) < 15)
                future = executor.submit(
                    self.process_and_save_single_file, 
                    idx, f, final_prompts, batch_output_dir, actual_model, aspect_ratio, target_size, final_api_key, need_preview
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
                        consecutive_failures = 0 
                        all_file_logs.append(f"[√] {fname} 生成 {count} 张")
                        if pils and len(preview_pil_images) < 15:
                            remain = 15 - len(preview_pil_images)
                            preview_pil_images.extend(pils[:remain])
                    else:
                        consecutive_failures += 1
                        err_msg = errs[0] if errs else "0 output"
                        all_file_logs.append(f"[X] {fname} {err_msg}")
                        failed_list.append(f"{fname} -> {err_msg}")
                        
                        if "CRITICAL_NO_CREDITS" in err_msg:
                            consecutive_failures = MAX_FAILURES
                            print("🚨 错误：积分耗尽，立即终止任务！")
                        
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
    "GrsaiLLMWriter": GrsaiLLMWriter,
    "GrsaiNanoBananaBatchDir": GrsaiNanoBananaBatchDir,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GrsaiNanoBanana": "🍌 Grsai Nano Banana",
    "GrsaiNanoBananaBatch": "🍌 Grsai Nano Banana Batch (CSV/Excel)",
    "GrsaiNanoBananaSaveWithPrompt": "🍌 Grsai Nano Banana 命名细化版",
    "GrsaiLLMWriter": "✍️ Grsai LLM/VLM Writer",
    "GrsaiNanoBananaBatchDir": "🍌📂 Grsai 文件夹批量处理(CSV)",
}