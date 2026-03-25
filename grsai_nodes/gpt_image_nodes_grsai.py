# grsai_nodes/gpt_image_nodes_grsai.py
import os
import json
import concurrent.futures
import traceback
import time
import requests
import torch
from typing import Any, Dict, Optional, List, Tuple
from PIL import Image

import builtins

# 从公共工具箱导入需要的核心函数
from ..utils import (
    download_image,
    pil_to_tensor,
    tensor_to_pil,
    upload_image_grsai
)

# --- API 客户端 (专门处理 GPT Image 特殊的 SSE 数据流) ---
class GrsaiAPIError(Exception): pass

class GrsaiGPTAPI:
    def __init__(self, api_key: str):
        if not api_key or not api_key.strip(): 
            raise GrsaiAPIError("API密钥不能为空")
        self.api_key = api_key
        self.session = requests.Session()
        self.session.headers.update({
            "Content-Type": "application/json; charset=utf-8", 
            "User-Agent": "ComfyUI-GptImage/2.0", 
            "Authorization": f"Bearer {self.api_key}"
        })
        self.base_url = "https://grsai.dakka.com.cn"

    def _make_request(self, method: str, endpoint: str, data: Optional[Dict] = None, timeout: int = 300) -> Dict[str, Any]:
        """专门处理 GPT Image 接口可能返回的 SSE (data: ...) 格式流"""
        url = f"{self.base_url}{endpoint}"
        response = None
        try:
            response = self.session.request(method, url, json=data, timeout=timeout)
            response.raise_for_status()
            raw_text = response.text.strip()
            
            # 1. 尝试作为标准 JSON 解析
            try:
                return json.loads(raw_text)
            except json.JSONDecodeError:
                pass 

            # 2. 尝试按 SSE 流处理 (提取最后一条有效 JSON)
            lines = raw_text.split('\n')
            last_valid_json = None
            for line in lines:
                line = line.strip()
                if line.startswith("data:"):
                    try:
                        json_str = line[5:].strip()
                        if json_str == "[DONE]" or not json_str: continue 
                        last_valid_json = json.loads(json_str)
                    except:
                        continue
            
            if last_valid_json:
                return last_valid_json
                
            raise GrsaiAPIError(f"无法解析API响应数据: {raw_text[:200]}...")

        except requests.exceptions.RequestException as e:
            msg = str(e)
            if response is not None: msg += f" | Response: {response.text[:100]}"
            raise GrsaiAPIError(f"网络请求失败: {msg}")

    def gpt_image_generate(self, prompt: str, urls: List[str], size: str, variants: int) -> Tuple[List[Image.Image], List[str]]:
        payload = {
            "model": "sora-image",
            "prompt": prompt,
            "urls": urls,
            "size": size,
            "variants": variants,
            "shutProgress": True
        }

        data = self._make_request("POST", "/v1/draw/completions", data=payload, timeout=300)

        # 错误检查
        if isinstance(data, dict):
            if data.get("code") and data.get("code") != 0:
                 raise GrsaiAPIError(f"API 业务错误: {data.get('msg')} (Code: {data.get('code')})")
            if data.get("status") == "failed":
                 raise GrsaiAPIError(f"生成任务失败: {data.get('failure_reason', '未知原因')}")

        # 解析结果图片 URL (兼容新旧接口结构)
        results_info = []
        if "results" in data and isinstance(data["results"], list):
            results_info = data["results"]
        elif "data" in data and isinstance(data["data"], dict) and "results" in data["data"]:
            results_info = data["data"]["results"]
        elif "url" in data and isinstance(data["url"], str) and data["url"]:
            results_info = [{"url": data["url"]}]
        elif "data" in data and isinstance(data["data"], dict) and "url" in data["data"] and data["data"]["url"]:
            results_info = [{"url": data["data"]["url"]}]

        if not results_info:
             raise GrsaiAPIError("API响应成功但未找到图片URL，请检查控制台日志。")

        target_urls = [r["url"] for r in results_info if "url" in r and r["url"]]
        
        # 多线程下载
        pil_images, download_errors = [], []
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, len(target_urls))) as executor:
            future_to_url = {executor.submit(download_image, url): url for url in target_urls}
            for future in concurrent.futures.as_completed(future_to_url):
                url = future_to_url[future]
                try:
                    img = future.result()
                    if img: pil_images.append(img)
                    else: download_errors.append(f"下载失败: {url}")
                except Exception as e:
                    download_errors.append(f"下载异常: {e}")
        
        return pil_images, download_errors


# --- 主要节点: GrsaiGptImage ---
class GrsaiGptImage:
    CATEGORY = "Nkxx/Grsai/图像"
    
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
                "api_key": ("STRING", {"multiline": False, "default": "", "placeholder": "留空则读取全局配置"}),
                "image_1": ("IMAGE",), "image_2": ("IMAGE",), "image_3": ("IMAGE",),
                "image_4": ("IMAGE",), "image_5": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "status")
    FUNCTION = "execute"

    @classmethod
    def IS_CHANGED(cls, **kwargs): return float("NaN")

    def _get_credits_balance(self, api_key: str) -> str:
        try:
            url = f"https://grsai.dakka.com.cn/client/common/getCredits?apikey={api_key}"
            res = requests.get(url, timeout=10)
            if res.status_code == 200:
                data = res.json()
                if data.get("code") == 0: return str(int(data["data"]["credits"]))
        except: pass
        return "N/A"

    def execute(self, prompt: str, size: str, variants: int, concurrency: int, api_key: str = "", **kwargs):
        final_api_key = builtins.get_api_key(api_key, "grsai")
        if not final_api_key: 
            return (torch.zeros((1, 1, 1, 3)), "错误: 未配置 Grsai API Key。")

        # 1. 提取并上传参考图 (调用 utils.py)
        images_in = [kwargs.get(f"image_{i}") for i in range(1, 6) if kwargs.get(f"image_{i}") is not None]
        uploaded_urls = []
        for img_tensor in images_in:
            url = upload_image_grsai(final_api_key, img_tensor)
            if url: uploaded_urls.append(url)
            else: return (torch.zeros((1, 1, 1, 3)), "错误: 部分参考图上传失败。")

        # 2. 并发执行生成任务
        try:
            api_client = GrsaiGPTAPI(api_key=final_api_key)
            all_pil_images, all_errors = [], []

            def task_runner(_):
                try:
                    return api_client.gpt_image_generate(prompt, uploaded_urls, size, variants)
                except Exception as e:
                    return [], [str(e)]

            with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
                results = executor.map(task_runner, range(concurrency))
                for pils, errs in results:
                    if pils: all_pil_images.extend(pils)
                    if errs: all_errors.extend(errs)

            # 3. 结果处理
            if not all_pil_images:
                err_msg = "; ".join(all_errors) if all_errors else "未知错误"
                return (torch.zeros((1, 1, 1, 3)), f"生成失败: {err_msg}")

            final_tensor = pil_to_tensor(all_pil_images)
            credits = self._get_credits_balance(final_api_key)
            status_text = f"成功: {len(all_pil_images)} 张 (任务 x{concurrency}, 变体 x{variants}) | 失败: {len(all_errors)} | 积分: {credits}"
            
            return {"ui": {"string": [status_text]}, "result": (final_tensor, status_text)}

        except Exception as e:
            traceback.print_exc()
            return (torch.zeros((1, 1, 1, 3)), f"系统错误: {str(e)}")

# --- 节点注册 ---
NODE_CLASS_MAPPINGS = {
    "GrsaiGptImage": GrsaiGptImage,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GrsaiGptImage": "🤖 Grsai GPT Image (Sora-Image)",
}