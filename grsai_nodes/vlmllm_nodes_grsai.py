# grsai_nodes/grsaivlmllm_nodes_grsai.py
import requests
import base64
import secrets
import time
from io import BytesIO

import builtins

# 从公共工具箱导入核心函数
from ..utils import tensor_to_pil, safe_pil_to_rgb

# ============= 配置 =============
API_BASE_URL = "https://grsai.dakka.com.cn/v1/chat/completions"

SUPPORTED_MODELS = [
    "gemini-3.1-pro",
    "gemini-3-flash",
    "gemini-3-pro",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite"
]

ZERO_WIDTH_CHARS = ["\u200b", "\u200c", "\u200d", "\ufeff", "\u180e", "\u200e", "\u200f"]

# ============= 工具函数 =============

def pil2base64(image):
    """将 PIL Image 安全转为 base64 字符串"""
    try:
        buffered = BytesIO()
        rgb_image = safe_pil_to_rgb(image)
        rgb_image.save(buffered, format="JPEG", quality=95)
        return base64.b64encode(buffered.getvalue()).decode('utf-8')
    except Exception as e:
        print(f"[VLM API] 图像转 base64 失败: {e}")
        return None

def call_grsai_api(api_key, model, messages):
    """调用 Grsai Chat Completions API"""
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
        if "choices" in result and len(result["choices"]) > 0:
            message = result["choices"][0]["message"]["content"]
            return message.strip(), None
        else:
            return None, f"API 返回格式异常: {result}"
            
    except requests.exceptions.Timeout:
        return None, "请求超时，请检查网络或重试"
    except requests.exceptions.RequestException as e:
        return None, f"网络请求失败: {str(e)}"
    except Exception as e:
        return None, f"未知错误: {str(e)}"

# ============= 节点 1: LLM (纯文本) =============
class GRSAILLMNode:
    CATEGORY = "Nkxx/Grsai/语言模型"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (SUPPORTED_MODELS, {"default": "gemini-2.5-flash"}),
                "prompt": ("STRING", {"default": "Hello! You are a helpful assistant.", "multiline": True}),
                "random_mode": (["固定", "随机"], {"default": "固定"}),
            },
            "optional": {
                "api_key": ("STRING", {"default": "", "multiline": False, "placeholder": "留空则读取全局配置"}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("response_text", "error_msg")
    FUNCTION = "generate"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        if kwargs.get("random_mode") == "随机":
            return time.time_ns()
        return False

    def generate(self, model, prompt, random_mode, api_key=""):
        final_api_key = builtins.get_api_key(api_key, "grsai")
        if not final_api_key:
            return (None, "未配置 Grsai API Key。")
        
        final_prompt = prompt.strip()
        
        # 随机模式下附加零宽字符以防止缓存
        if random_mode == "随机":
            final_prompt = f"{final_prompt}{secrets.choice(ZERO_WIDTH_CHARS)}"

        messages = [{"role": "user", "content": final_prompt}]
        
        response, error = call_grsai_api(final_api_key, model, messages)
        return (response, error)

# ============= 节点 2: VLM (图文对话) =============
class GRSAIVLMNode:
    CATEGORY = "Nkxx/Grsai/语言模型"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (SUPPORTED_MODELS, {"default": "gemini-3.1-pro"}), # 将默认模型更新为了最新的 gemini-3.1-pro
                "prompt": ("STRING", {"default": "Describe these images.", "multiline": True}),
                "random_mode": (["固定", "随机"], {"default": "固定"}),
            },
            "optional": {
                "api_key": ("STRING", {"default": "", "multiline": False, "placeholder": "留空则读取全局配置"}),
                "image_1": ("IMAGE",), "image_2": ("IMAGE",), "image_3": ("IMAGE",),
                "image_4": ("IMAGE",), "image_5": ("IMAGE",), "image_6": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("response_text", "error_msg")
    FUNCTION = "generate_with_image"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        if kwargs.get("random_mode") == "随机":
            return time.time_ns()
        return False

    def generate_with_image(self, model, prompt, random_mode, api_key="", **kwargs):
        final_api_key = builtins.get_api_key(api_key, "grsai")
        if not final_api_key:
            return (None, "未配置 Grsai API Key。")

        final_prompt = prompt.strip()
        if random_mode == "随机":
            final_prompt = f"{final_prompt}{secrets.choice(ZERO_WIDTH_CHARS)}"

        # 收集所有非空的图片输入
        images_in = [kwargs.get(f"image_{i}") for i in range(1, 7)]
        valid_images = [img for img in images_in if img is not None]

        messages = []

        # === 纯文本模式 ===
        if not valid_images:
            messages = [{"role": "user", "content": final_prompt}]
        
        # === 多模态模式 ===
        else:
            content_list = [{"type": "text", "text": final_prompt}]
            processed_count = 0
            
            for img_tensor in valid_images:
                try:
                    # 使用工具类进行转换
                    pil_images = tensor_to_pil(img_tensor)
                    for pil_img in pil_images:
                        img_base64 = pil2base64(pil_img)
                        if img_base64:
                            content_list.append({
                                "type": "image_url", 
                                "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}
                            })
                            processed_count += 1
                except Exception as e:
                    print(f"[VLM API] 处理图片时出错: {e}")

            if processed_count == 0:
                return (None, "图片输入已连接但处理失败")

            messages = [{"role": "user", "content": content_list}]
        
        response, error = call_grsai_api(final_api_key, model, messages)
        return (response, error)

# ============= 注册节点 =============
NODE_CLASS_MAPPINGS = {
    "GRSAILLMNode": GRSAILLMNode,
    "GRSAIVLMNode": GRSAIVLMNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GRSAILLMNode": "💬 GRSAI LLM (纯文本对话)",
    "GRSAIVLMNode": "👁️ GRSAI VLM (图文视觉对话)"
}