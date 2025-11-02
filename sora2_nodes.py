import os
import requests
import time
import json
import uuid
import folder_paths
import comfy.utils
import traceback
import tempfile
from PIL import Image
import torch
import numpy as np
import cv2
from comfy.comfy_types import IO
import shutil
import secrets # 导入secrets模块用于生成随机内容

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

class GrsaiVideoAdapter:
    """
    一个视频适配器，用于封装生成的视频路径，使其能被ComfyUI的视频节点（如Save Video）接收。
    """
    def __init__(self, video_path: str):
        self.video_path = video_path

    def get_dimensions(self):
        """获取视频的宽度和高度"""
        try:
            if not self.video_path or not os.path.exists(self.video_path):
                return 1280, 720 # 如果路径无效，返回默认值
            cap = cv2.VideoCapture(self.video_path)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            return width, height
        except Exception:
            # 如果出错，返回一个默认值
            return 1280, 720

    def save_to(self, output_path, format="auto", codec="auto", metadata=None):
        try:
            if self.video_path and os.path.exists(self.video_path):
                shutil.copyfile(self.video_path, output_path)
                return True
            else:
                print(f"[GrsaiVideoAdapter] 错误: 源视频文件路径无效: {self.video_path}")
                return False
        except Exception as e:
            print(f"[GrsaiVideoAdapter] 保存视频时出错: {e}")
            return False

def _robust_download_video(video_url: str, output_path: str, max_retries: int = 3, timeout: int = 300):
    """
    :param video_url: 视频的URL
    :param output_path: 保存的本地路径
    :param max_retries: 最大重试次数
    :param timeout: 每次请求的超时时间（秒）
    """
    
    # 模拟浏览器的User-Agent
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"
    }
    
    for attempt in range(1, max_retries + 1):
        try:
            print(f"[Sora2 Downloader] 尝试第 {attempt}/{max_retries} 次下载: {video_url}")
            # 使用 stream=True 进行流式下载
            with requests.get(video_url, stream=True, headers=headers, timeout=timeout) as response:
                # 检查HTTP状态码
                response.raise_for_status()  # 如果状态码是 4xx 或 5xx，将引发异常
                
                # 写入文件
                with open(output_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk: # 过滤掉 keep-alive new chunks
                            f.write(chunk)
                
                print(f"[Sora2 Downloader] 视频下载成功: {output_path}")
                return True # 下载成功
                
        except requests.exceptions.RequestException as e:
            print(f"[Sora2 Downloader] 第 {attempt} 次下载失败: {e}")
            if isinstance(e, requests.exceptions.HTTPError) and 400 <= e.response.status_code < 500:
                # 客户端错误 (如 403 Forbidden, 404 Not Found)，重试也无用
                print("[Sora2 Downloader] 遇到客户端错误，停止重试。")
                raise e # 重新引发异常，让外部捕获
            
            if attempt < max_retries:
                # 如果不是最后一次尝试，则等待后重试（指数退避）
                wait_time = 5 * (2 ** (attempt - 1)) # 5s, 10s
                print(f"[Sora2 Downloader] 等待 {wait_time} 秒后重试...")
                time.sleep(wait_time)
            else:
                # 达到最大重试次数
                print("[Sora2 Downloader] 达到最大重试次数，下载失败。")
                raise e # 重新引发最后的异常

    return False 

# --- 主节点类 ---
class Sora2Generator:
    """
    使用Grsai API生成Sora视频的ComfyUI节点。
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"default": "A cute cat playing on the grass", "multiline": True}),
                "aspect_ratio": (["16:9", "9:16"], {"default": "16:9"}),
                "duration": (["10", "15"], {"default": "10"}), 
            },
            "optional": {
                "api_key": ("STRING", {"default": "", "multiline": False, "placeholder": "留空则使用 __init__.py 中的配置"}),
                "image": ("IMAGE",)
            }
        }

    RETURN_TYPES = (IO.VIDEO, "STRING", "STRING")
    RETURN_NAMES = ("video", "video_url", "response")
    FUNCTION = "execute"
    CATEGORY = "Nkxx/视频"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        """
        强制节点每次都重新执行，以实现随机性。
        此方法告诉ComfyUI节点的输出总是“过时的”，强制它重新运行，即使输入没有改变。
        """
        return time.time_ns()

    def _get_headers(self, api_key: str) -> dict:
        """构建请求头"""
        return {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}

    def _get_credits_balance(self, api_key: str) -> str:
        """查询API积分余额，返回字符串"""
        try:
            url = f"https://grsai.dakka.com.cn/client/common/getCredits?apikey={api_key}"
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data.get("code") == 0 and "data" in data and "credits" in data["data"]:
                    return str(int(data["data"]["credits"]))
        except Exception:
            pass
        return "查询失败"
    
    def _upload_image(self, api_key: str, image_tensor: torch.Tensor) -> str:
        """将图像上传并获取URL"""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_f:
            pil_img = Image.fromarray(np.clip(255. * image_tensor.cpu().numpy(), 0, 255).astype(np.uint8))
            if pil_img.mode != 'RGB': pil_img = pil_img.convert('RGB')
            pil_img.save(temp_f, "PNG"); temp_f_path = temp_f.name
        try:
            headers = self._get_headers(api_key)
            token_res = requests.post("https://grsai.dakka.com.cn/client/resource/newUploadTokenZH", headers=headers, json={"sux": "png"}, timeout=30)
            token_res.raise_for_status(); token_data = token_res.json()["data"]
            token, key, up_url, domain = (token_data["token"], token_data["key"], token_data["url"], token_data["domain"])
            with open(temp_f_path, "rb") as f:
                requests.post(url=up_url, data={"token": token, "key": key}, files={"file": f}, timeout=120).raise_for_status()
            return f"{domain}/{key}"
        finally:
            if os.path.exists(temp_f_path): os.unlink(temp_f_path)
    
    def execute(self, prompt, aspect_ratio, duration, api_key="", image=None):
        final_api_key = get_api_key(api_key)
        if not final_api_key:
            return (GrsaiVideoAdapter(""), "", "状态: 失败\n错误: API Key 不能为空。请在节点中或 __init__.py 中设置。")

        pbar = comfy.utils.ProgressBar(100)
        pbar.update_absolute(5)

        try:
            # 1. 准备Payload, 并添加随机后缀以避免缓存
            final_prompt = f"{prompt.strip()}{secrets.choice(ZERO_WIDTH_CHARS)}"
            
            payload = {
                "model": "sora-2", "prompt": final_prompt, "aspectRatio": aspect_ratio,
                "duration": int(duration), "size": "small", "webHook": "-1" # size 已被写死
            }
            if image is not None:
                pbar.update_absolute(10)
                print("[Sora2] 检测到图像输入，开始上传...")
                image_url = self._upload_image(final_api_key, image[0])
                if not image_url: raise Exception("图像上传失败")
                payload["url"] = image_url
                print(f"[Sora2] 图像上传成功: {image_url}")

            # 2. 提交任务
            pbar.update_absolute(20)
            print("[Sora2] 正在提交视频生成任务...")
            submit_response = requests.post("https://grsai.dakka.com.cn/v1/video/sora-video", headers=self._get_headers(final_api_key), json=payload, timeout=60)
            submit_response.raise_for_status(); submit_data = submit_response.json()
            if submit_data.get("code") != 0: raise Exception(f"API提交任务失败: {submit_data.get('msg', '未知错误')}")
            task_id = submit_data.get("data", {}).get("id")
            if not task_id: raise Exception("API未能返回有效的任务ID")
            print(f"[Sora2] 任务提交成功, 任务ID: {task_id}")
            pbar.update_absolute(30)

            # 3. 轮询结果
            print("[Sora2] 开始轮询任务结果...")
            last_progress, start_time, timeout = 0, time.time(), 900
            while time.time() - start_time < timeout:
                time.sleep(5)
                status_response = requests.post("https://grsai.dakka.com.cn/v1/draw/result", headers=self._get_headers(final_api_key), json={"id": task_id}, timeout=30)
                if status_response.status_code != 200:
                    print(f"[Sora2] 查询状态失败, HTTP {status_response.status_code}, 将重试...")
                    continue
                status_data = status_response.json()
                if status_data.get("code") == -22: raise Exception(f"轮询失败: {status_data.get('msg', '任务不存在或已过期')}")
                
                task_info = status_data.get("data", {})
                status = task_info.get("status")
                progress = task_info.get("progress", 0)

                if progress and progress > last_progress:
                    pbar.update_absolute(30 + int(progress * 0.6)); last_progress = progress
                
                if status == "succeeded":
                    pbar.update_absolute(90)
                    video_url = task_info.get("results", [{}])[0].get("url")
                    if not video_url: raise Exception("任务成功但未找到视频URL")
                    
                    try:
                        output_dir, filename = folder_paths.get_output_directory(), f"sora2_{uuid.uuid4().hex[:8]}.mp4"
                        output_path = os.path.join(output_dir, filename)
                        os.makedirs(output_dir, exist_ok=True)
                        
                        # 调用新的下载函数
                        _robust_download_video(video_url, output_path, max_retries=3, timeout=300)
                        
                        pbar.update_absolute(100)
                        video_adapter = GrsaiVideoAdapter(output_path)
                        credits_text = self._get_credits_balance(final_api_key)
                        status_msg = f"状态:  succes\n任务ID: {task_id}\n视频已下载\n剩余积分: {credits_text}"
                        return (video_adapter, video_url, status_msg)

                    except Exception as download_error:
                        # 如果重试后仍然失败
                        pbar.update_absolute(100)
                        print(f"[Sora2] 视频下载失败（已重试），但生成成功。错误: {download_error}")
                        credits_text = self._get_credits_balance(final_api_key)
                        status_msg = f"状态: 下载失败\n任务ID: {task_id}\n错误: {download_error}\n剩余积分: {credits_text}"
                        # 即使下载失败，也返回 URL，让用户可以手动下载
                        return (GrsaiVideoAdapter(""), video_url, status_msg)
                    # *** 【修改结束】 ***

                elif status == "failed":
                    fail_reason = task_info.get('failure_reason', '未知原因')
                    reason_map = {
                        "output_moderation": "输出内容审核失败",
                        "input_moderation": "输入内容审核失败 (Prompt违规)",
                        "error": "其他未知错误"
                    }
                    raise Exception(f"视频生成任务失败: {reason_map.get(fail_reason, fail_reason)}")

            raise Exception(f"轮询超时 ({timeout}秒)，任务未完成。")

        except Exception as e:
            error_message = f"状态: 失败\n错误信息: {e}"
            print(f"[Sora2] 执行Sora节点时出错: {e}")
            traceback.print_exc()
            return (GrsaiVideoAdapter(""), "", error_message)

# --- 节点注册 ---
NODE_CLASS_MAPPINGS = { "Sora2Generator_Grsai": Sora2Generator }
NODE_DISPLAY_NAME_MAPPINGS = { "Sora2Generator_Grsai": "Sora2 Video Generator (Grsai)" }