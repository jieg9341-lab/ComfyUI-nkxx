# grsaivlmllm_nodes.py
import requests
import base64
import torch
import numpy as np
from PIL import Image
from io import BytesIO
import secrets
import time

# ============= 配置 =============
API_BASE_URL = "https://api.grsai.com/v1/chat/completions"

SUPPORTED_MODELS = [
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite"
]

SYMBOLS = "▲★◆●■▶◀▼◇○◎⊕⊗☆♠♥♦♣♪♫☀☁☂☃☄☇☈☉☊☋✓✔✕✖✗✘☛☞☚☜☝☟☠☡☢☣"

# ============= 工具函数 =============

def tensor2pil(image):
    """将 ComfyUI 的 Tensor 图像转为 PIL Image"""
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

def pil2base64(image):
    """将 PIL Image 转为 base64 字符串"""
    try:
        buffered = BytesIO()
        image.convert("RGB").save(buffered, format="JPEG", quality=95)
        return base64.b64encode(buffered.getvalue()).decode('utf-8')
    except Exception as e:
        print(f"[VLMAPI] 图像转 base64 失败: {e}")
        return None

def call_grsai_api(api_key, model, messages):
    """调用 grsai API，返回 (response_text, error_msg)"""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    payload = {
        "model": model,
        "messages": messages,
        "stream": False
    }

    try:
        response = requests.post(API_BASE_URL, headers=headers, json=payload, timeout=60)
        if response.status_code != 200:
            return None, f"HTTP {response.status_code}: {response.text}"
        result = response.json()
        message = result["choices"][0]["message"]["content"]
        return message.strip(), None
    except requests.exceptions.Timeout:
        return None, "请求超时，请检查网络或重试"
    except requests.exceptions.RequestException as e:
        return None, f"网络请求失败: {str(e)}"
    except Exception as e:
        return None, f"未知错误: {str(e)}"

# ============= LLM 节点 =============

class GRSAILLMNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (SUPPORTED_MODELS, {"default": "gemini-2.5-flash"}),
                "system_prompt": ("STRING", {"default": "You are a helpful assistant.", "multiline": True}),
                "user_prompt": ("STRING", {"default": "Hello!", "multiline": True}),
                "random_mode": (["固定", "随机"], {"default": "固定"}),
            },
            "optional": {
                "api_key": ("STRING", {"default": "", "multiline": False, "placeholder": "留空则使用 __init__.py 中的配置"}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("response_text", "error_msg")
    FUNCTION = "generate"
    CATEGORY = "Nkxx/语言模型"

    def generate(self, model, system_prompt, user_prompt, random_mode, api_key=""):
        final_api_key = get_api_key(api_key)
        if not final_api_key:
            return (None, "API Key 不能为空。请在节点中填写，或在 __init__.py 文件中设置。")
        
        final_prompt = user_prompt.strip()
        if random_mode == "随机":
            final_prompt = f"{final_prompt}\u200b{secrets.choice(SYMBOLS)}"

        messages = [
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": final_prompt}
        ]
        response, error = call_grsai_api(final_api_key, model, messages)
        return (response, error)

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        if kwargs.get("random_mode") == "随机":
            return time.time_ns()
        return False

# ============= VLM 节点 =============

class GRSAIVLMNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (SUPPORTED_MODELS, {"default": "gemini-2.5-pro"}),
                "image": ("IMAGE",),
                "prompt": ("STRING", {"default": "Describe this image.", "multiline": True}),
                "random_mode": (["固定", "随机"], {"default": "固定"}),
            },
            "optional": {
                "api_key": ("STRING", {"default": "", "multiline": False, "placeholder": "留空则使用 __init__.py 中的配置"}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("response_text", "error_msg")
    FUNCTION = "generate_with_image"
    CATEGORY = "Nkxx/语言模型"

    def generate_with_image(self, model, image, prompt, random_mode, api_key=""):
        final_api_key = get_api_key(api_key)
        if not final_api_key:
            return (None, "API Key 不能为空。请在节点中填写，或在 __init__.py 文件中设置。")

        final_prompt = prompt.strip()
        if random_mode == "随机":
            final_prompt = f"{final_prompt}\u200b{secrets.choice(SYMBOLS)}"

        img_base64 = pil2base64(tensor2pil(image))
        if img_base64 is None:
            return (None, "图像处理失败")

        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": final_prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}}
            ]
        }]
        response, error = call_grsai_api(final_api_key, model, messages)
        return (response, error)

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        if kwargs.get("random_mode") == "随机":
            return time.time_ns()
        return False

# ============= 注册节点 =============
NODE_CLASS_MAPPINGS = {
    "GRSAILLMNode": GRSAILLMNode,
    "GRSAIVLMNode": GRSAIVLMNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GRSAILLMNode": "GRSAI LLM (文本对话)",
    "GRSAIVLMNode": "GRSAI VLM (图文对话)"
}
