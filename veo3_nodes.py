import os
import requests
import time
import uuid
import folder_paths
import comfy.utils
import traceback
import shutil
import cv2
from comfy.comfy_types import IO
import tempfile
from PIL import Image
import torch
import numpy as np
import secrets # 导入secrets模块

# --- 辅助工具 ---

# 定义用于随机化prompt的零宽字符，避免API缓存
ZERO_WIDTH_CHARS = [
    "\u200b",
    "\u200c",
    "\u200d",
    "\ufeff",
    "\u180e",
    "\u200e",
    "\u200f",
]

class VideoAdapter:
    """
    一个视频适配器，用于封装生成的视频路径，使其能被ComfyUI的视频节点（如Save Video）接收。
    """
    def __init__(self, video_path: str):
        self.video_path = video_path

    def get_dimensions(self):
        try:
            if not self.video_path or not os.path.exists(self.video_path):
                return 1920, 1080
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                return 1920, 1080
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            return width, height
        except Exception:
            return 1920, 1080

    def save_to(self, output_path, format="auto", codec="auto", metadata=None):
        try:
            if self.video_path and os.path.exists(self.video_path):
                shutil.copyfile(self.video_path, output_path)
                return True
            else:
                print(f"[VideoAdapter] 错误: 源视频文件路径无效: {self.video_path}")
                return False
        except Exception as e:
            print(f"[VideoAdapter] 保存视频时出错: {e}")
            return False

# --- 共享的辅助方法 ---

def _get_headers(api_key: str) -> dict:
    """构建请求头"""
    return {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

def _get_credits_balance(api_key: str) -> str:
    """查询API积分余额"""
    try:
        host = "https://grsai.dakka.com.cn"
        url = f"{host}/client/common/getCredits?apikey={api_key}"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if data.get("code") == 0 and "data" in data and "credits" in data["data"]:
                return str(int(data["data"]["credits"]))
    except Exception as e:
        print(f"[Veo3.1] 查询积分失败: {e}")
    return "查询失败"

def _upload_image(api_key: str, image_tensor: torch.Tensor) -> str:
    """将 ComfyUI 的图像张量上传并获取其可访问的 URL"""
    i = 255. * image_tensor.cpu().numpy()
    img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
    if img.mode != 'RGB':
        img = img.convert('RGB')

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_f:
        img.save(temp_f, "PNG")
        temp_f_path = temp_f.name

    try:
        host = "https://grsai.dakka.com.cn"
        headers = _get_headers(api_key)
        token_res = requests.post(f"{host}/client/resource/newUploadTokenZH", headers=headers, json={"sux": "png"}, timeout=30)
        token_res.raise_for_status()
        token_data = token_res.json()["data"]
        
        token = token_data.get("token")
        key = token_data.get("key")
        up_url = token_data.get("url")
        domain = token_data.get("domain")

        if not all([token, key, up_url, domain]):
            raise Exception("从API获取的上传令牌数据不完整")

        with open(temp_f_path, "rb") as f:
            response = requests.post(url=up_url, data={"token": token, "key": key}, files={"file": f}, timeout=120)
            response.raise_for_status()
        
        return f"{domain}/{key}"
    finally:
        if os.path.exists(temp_f_path):
            os.unlink(temp_f_path)

def _execute_veo_task(api_key, payload):
    """
    通用的任务提交和轮询函数
    返回: (video_adapter, video_url, status_msg)
    """
    host = "https://grsai.dakka.com.cn"
    headers = _get_headers(api_key)
    pbar = comfy.utils.ProgressBar(100)
    pbar.update_absolute(20) # 假设图片上传已完成
    
    try:
        # 1. 提交任务
        print(f"[Veo3.1] 正在提交视频生成任务... Payload: {payload}")
        submit_url = f"{host}/v1/video/veo"
        submit_response = requests.post(submit_url, headers=headers, json=payload, timeout=60)
        submit_response.raise_for_status()
        submit_data = submit_response.json()

        if submit_data.get("code") != 0:
            raise Exception(f"API提交任务失败: {submit_data.get('msg', '未知错误')}")
        
        task_id = submit_data.get("data", {}).get("id")
        if not task_id:
            raise Exception("API未能返回有效的任务ID。")
        
        print(f"[Veo3.1] 任务提交成功, 任务ID: {task_id}")
        pbar.update_absolute(30)

        # 2. 轮询结果
        print("[Veo3.1] 开始轮询任务结果...")
        result_url = f"{host}/v1/draw/result"
        last_progress, start_time, timeout = 0, time.time(), 900

        while time.time() - start_time < timeout:
            time.sleep(5)
            status_response = requests.post(result_url, headers=headers, json={"id": task_id}, timeout=30)
            
            if status_response.status_code != 200:
                print(f"[Veo3.1] 查询状态失败, HTTP {status_response.status_code}, 将重试...")
                continue
            
            status_data = status_response.json()
            
            if status_data.get("code") != 0:
                raise Exception(f"轮询失败: {status_data.get('msg', f'错误码: {status_data.get("code")}')}")
            
            task_info = status_data.get("data", {})
            if not task_info:
                continue

            status = task_info.get("status")
            progress = task_info.get("progress")

            if progress is not None and progress > last_progress:
                pbar.update_absolute(30 + int(progress * 0.6))
                last_progress = progress
            
            if status == "succeeded":
                pbar.update_absolute(90)
                video_url = task_info.get("url")
                if not video_url:
                    raise Exception("任务成功但未找到视频URL。")
                
                try:
                    output_dir = folder_paths.get_output_directory()
                    filename = f"veo3.1_{uuid.uuid4().hex[:8]}.mp4"
                    output_path = os.path.join(output_dir, filename)
                    os.makedirs(output_dir, exist_ok=True)
                    
                    print(f"[Veo3.1] 正在下载视频到: {output_path}")
                    video_response = requests.get(video_url, stream=True, timeout=120)
                    video_response.raise_for_status()
                    with open(output_path, 'wb') as f:
                        for chunk in video_response.iter_content(chunk_size=8120):
                            if chunk:
                                f.write(chunk)
                    
                    pbar.update_absolute(100)
                    video_adapter = VideoAdapter(output_path)
                    credits_text = _get_credits_balance(api_key)
                    status_msg = f"状态: 成功\n任务ID: {task_id}\n视频已下载\n剩余积分: {credits_text}"
                    return (video_adapter, video_url, status_msg)

                except Exception as download_error:
                    pbar.update_absolute(100)
                    print(f"[Veo3.1] 视频下载失败，但生成成功。错误: {download_error}")
                    credits_text = _get_credits_balance(api_key)
                    status_msg = f"状态: 下载失败\n任务ID: {task_id}\n错误: {download_error}\n剩余积分: {credits_text}"
                    return (VideoAdapter(""), video_url, status_msg)

            elif status == "failed":
                fail_reason = task_info.get('failure_reason', '未知原因')
                error_details = task_info.get('error', '')
                reason_map = {
                    "output_moderation": "输出内容违规",
                    "input_moderation": "输入内容违规 (Prompt违规)",
                    "error": "其他错误"
                }
                full_error_msg = f"视频生成任务失败: {reason_map.get(fail_reason, fail_reason)}"
                if error_details:
                    full_error_msg += f"\n详细信息: {error_details}"
                raise Exception(full_error_msg)

        raise Exception(f"轮询超时 ({timeout}秒)，任务未完成。")

    except Exception as e:
        error_message = f"状态: 失败\n错误信息: {e}"
        print(f"[Veo3.1] 执行节点时出错: {e}")
        traceback.print_exc()
        return (VideoAdapter(""), "", error_message)


# --- 节点一: 参考图生成 ---
class Veo3_1_RefGenerator:
    """
    Veo3.1 视频生成 (grsai) - 支持最多3张参考图
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": "", "multiline": False, "tooltips": "请输入您的API密钥"}),
                
                # --- (*** 这是修改部分 ***) ---
                # 根据API文档, 'urls' 仅支持 veo3.1-fast
                "model": (["veo3.1-fast"], {"default": "veo3.1-fast"}),
                "prompt": ("STRING", {"default": "A cute cat playing on the grass", "multiline": True, "tooltips": "提示词，只支持英文"}),
                # 根据API文档, 'urls' 仅支持 16:9
                "aspect_ratio": (["16:9"], {"default": "16:9"}),
                # --- (*** 修改结束 ***) ---
            },
            "optional": {
                "ref_image_1": ("IMAGE",),
                "ref_image_2": ("IMAGE",),
                "ref_image_3": ("IMAGE",),
            }
        }

    RETURN_TYPES = (IO.VIDEO, "STRING", "STRING")
    RETURN_NAMES = ("video", "video_url", "response")
    FUNCTION = "execute"
    CATEGORY = "Nkxx/视频"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return time.time_ns()

    def execute(self, api_key, model, prompt, aspect_ratio, 
                ref_image_1=None, ref_image_2=None, ref_image_3=None):
        
        if not api_key or not api_key.strip():
            return (VideoAdapter(""), "", "状态: 失败\n错误: API Key 不能为空。")

        try:
            print("[Veo3.1-Ref] 检查并上传参考图像...")
            
            # --- (*** 这是修改部分 ***) ---
            
            # 1. 创建一个列表来存储所有有效的URL
            ref_urls = []

            if ref_image_1 is not None:
                print("[Veo3.1-Ref] 上传参考图 1...")
                ref_urls.append(_upload_image(api_key, ref_image_1[0]))
            if ref_image_2 is not None:
                print("[Veo3.1-Ref] 上传参考图 2...")
                ref_urls.append(_upload_image(api_key, ref_image_2[0]))
            if ref_image_3 is not None:
                print("[Veo3.1-Ref] 上传参考图 3...")
                ref_urls.append(_upload_image(api_key, ref_image_3[0]))
            
            print("[Veo3.1-Ref] 图像上传完成。")

            final_prompt = f"{prompt.strip()}{secrets.choice(ZERO_WIDTH_CHARS)}"
            
            payload = {
                "model": model,
                "prompt": final_prompt,
                "aspectRatio": aspect_ratio,
                "webHook": "-1"
            }
            
            # 2. 根据API文档, 使用 'urls' 键, 并且值为一个列表
            if ref_urls: # 仅当列表不为空时才添加
                payload["urls"] = ref_urls
            
            # --- (*** 修改结束 ***) ---
            
            # 调用通用执行器
            return _execute_veo_task(api_key, payload)

        except Exception as e:
            error_message = f"状态: 失败\n错误信息: {e}"
            print(f"[Veo3.1-Ref] 执行节点时出错: {e}")
            traceback.print_exc()
            return (VideoAdapter(""), "", error_message)


# --- 节点二: 首尾帧生成 ---
class Veo3_1_FramesGenerator:
    """
    veo3.1首尾帧(grsai) - 支持首帧和尾帧
    (此节点根据API文档是正确的，无需修改)
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": "", "multiline": False, "tooltips": "请输入您的API密钥"}),
                # 'lastFrameUrl' 支持 'veo3.1-fast', 'veo3.1-pro'
                "model": (["veo3.1-fast", "veo3.1-pro"], {"default": "veo3.1-fast"}),
                "prompt": ("STRING", {"default": "A cute cat playing on the grass", "multiline": True, "tooltips": "提示词，只支持英文"}),
                # 文档未限制首尾帧的宽高比, 保留两者
                "aspect_ratio": (["16:9", "9:16"], {"default": "16:9"}),
            },
            "optional": {
                "first_frame": ("IMAGE",),
                "last_frame": ("IMAGE",),
            }
        }

    RETURN_TYPES = (IO.VIDEO, "STRING", "STRING")
    RETURN_NAMES = ("video", "video_url", "response")
    FUNCTION = "execute"
    CATEGORY = "Nkxx/视频"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return time.time_ns()

    def execute(self, api_key, model, prompt, aspect_ratio, 
                first_frame=None, last_frame=None):
        
        if not api_key or not api_key.strip():
            return (VideoAdapter(""), "", "状态: 失败\n错误: API Key 不能为空。")

        try:
            print("[Veo3.1-Frames] 检查并上传首尾帧图像...")
            first_frame_url = None
            last_frame_url = None
            
            if first_frame is not None:
                print("[Veo3.1-Frames] 上传首帧...")
                first_frame_url = _upload_image(api_key, first_frame[0])
            if last_frame is not None:
                print("[Veo3.1-Frames] 上传尾帧...")
                last_frame_url = _upload_image(api_key, last_frame[0])
            
            # API文档说: lastFrameUrl 需搭配 firstFrameUrl 使用
            if last_frame_url and not first_frame_url:
                raise Exception("API错误: 提供了尾帧(lastFrameUrl)但未提供首帧(firstFrameUrl)。")

            print("[Veo3.1-Frames] 图像上传完成。")

            final_prompt = f"{prompt.strip()}{secrets.choice(ZERO_WIDTH_CHARS)}"
            
            payload = {
                "model": model,
                "prompt": final_prompt,
                "aspectRatio": aspect_ratio,
                "webHook": "-1"
            }
            
            if first_frame_url:
                payload["firstFrameUrl"] = first_frame_url
            if last_frame_url:
                payload["lastFrameUrl"] = last_frame_url
            
            # 调用通用执行器
            return _execute_veo_task(api_key, payload)

        except Exception as e:
            error_message = f"状态: 失败\n错误信息: {e}"
            print(f"[Veo3.1-Frames] 执行节点时出错: {e}")
            traceback.print_exc()
            return (VideoAdapter(""), "", error_message)


# --- 节点注册 ---
NODE_CLASS_MAPPINGS = {
    "Veo3_1_RefGenerator": Veo3_1_RefGenerator,
    "Veo3_1_FramesGenerator": Veo3_1_FramesGenerator
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "Veo3_1_RefGenerator": "Veo3.1 参考图生成 (grsai)",
    "Veo3_1_FramesGenerator": "veo3.1首尾帧(grsai)"
}