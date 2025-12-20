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
import secrets
import subprocess 
import sys
import importlib
from datetime import datetime 
import yt_dlp

# --- SSL相关导入 ---
import ssl
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
ssl._create_default_https_context = ssl._create_unverified_context
HOST = "https://grsai.dakka.com.cn"  

ZERO_WIDTH_CHARS = [
    "\u200b", "\u200c", "\u200d", "\ufeff",
    "\u180e", "\u200e", "\u200f",
]

# --- 辅助类与函数 ---

class GrsaiVideoAdapter:
    def __init__(self, video_path: str):
        self.video_path = video_path

    def get_dimensions(self):
        try:
            if not self.video_path or not os.path.exists(self.video_path):
                return 1280, 720
            
            # [新增] 防止 0KB 文件导致 cv2 卡死
            if os.path.getsize(self.video_path) < 1024:
                return 1280, 720

            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                return 1280, 720
                
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            
            if width <= 0 or height <= 0:
                return 1280, 720
                
            return width, height
        except Exception:
            return 1280, 720

    def save_to(self, output_path, format="auto", codec="auto", metadata=None):
        try:
            if self.video_path and os.path.exists(self.video_path):
                shutil.copyfile(self.video_path, output_path)
                return True
            else:
                return False
        except Exception as e:
            return False

# --- 🔥 核心下载函数：修复版 (yt-dlp优先 + 反死锁Curl) ---
def _robust_download_video(video_url: str, output_path: str, max_retries: int = 3, timeout: int = 300):
    print(f"[Sora2 Downloader] 准备下载: {video_url}")
    
    # --- 方案 1: 优先尝试 yt-dlp ---
    # yt-dlp 是纯Python库，通常更稳定，不易产生IO管道死锁
    try:
        # print(f"[Sora2 Downloader] 正在使用 yt-dlp 下载...") # 可选日志
        
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Referer": "https://grsai.dakka.com.cn/"
        }
        
        ydl_opts = {
            'outtmpl': output_path,
            'retries': max_retries,
            'http_headers_str': "\r\n".join([f"{k}: {v}" for k, v in headers.items()]),
            'quiet': True, 
            'noplaylist': True,
            'merge_output_format': 'mp4',
            'socket_timeout': 30, # 单次连接超时
            'nocheckcertificate': True, 
        }

            
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([video_url])
        
        # 验证文件有效性（大小 > 1KB）
        if os.path.exists(output_path) and os.path.getsize(output_path) > 1024:
             return True
        else:
            print("[Sora2 Downloader] yt-dlp 完成但文件无效，尝试备用方案...")
            
    except Exception as e:
        print(f"[Sora2 Downloader] yt-dlp 下载失败或未安装: {e}。切换至 Curl 备用方案。")
        if os.path.exists(output_path):
            try: os.remove(output_path)
            except: pass


    # --- 方案 2: 备用 Curl (反死锁改进版) ---
    curl_path = shutil.which("curl")
    if curl_path:
        for attempt in range(max_retries):
            try:
                print(f"[Sora2 Downloader] 尝试调用系统 Curl (第 {attempt+1}/{max_retries} 次)...")
                # 构建命令
                cmd = [
                    curl_path, 
                    "-k", 
                    "-L", 
                    "--connect-timeout", "20",
                    "--max-time", str(timeout),
                    "-o", output_path, 
                    video_url
                ]
                
                # [关键修改] 使用 subprocess.run 代替 check_call
                # stdout/stderr 设置为 DEVNULL 防止管道缓冲区填满导致死锁
                # timeout 参数确保 Python 能强制杀死卡死的 Curl 进程
                subprocess.run(
                    cmd, 
                    stdout=subprocess.DEVNULL, 
                    stderr=subprocess.DEVNULL, 
                    timeout=timeout + 10, 
                    check=True
                )
                
                if os.path.exists(output_path) and os.path.getsize(output_path) > 1024:
                    return True
                else:
                    print("[Sora2 Downloader] Curl 命令结束但文件无效")
                    
            except subprocess.TimeoutExpired:
                print(f"[Sora2 Downloader] ⚠️ Curl 进程超时 (Python层强制终止)")
            except subprocess.CalledProcessError:
                print(f"[Sora2 Downloader] ⚠️ Curl 返回错误代码")
            except Exception as e:
                print(f"[Sora2 Downloader] Curl 执行异常: {e}")

            time.sleep(2) # 重试前冷却
    else:
        print("[Sora2 Downloader] 系统未找到 Curl，且 yt-dlp 失败。")

    # 若到此处仍未成功，抛出异常
    raise Exception("所有下载方案均失败 (yt-dlp 和 Curl)")


def _get_common_headers(api_key: str) -> dict:
    return {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}

def _save_character_record(char_id, remark_text, source_info):
    """保存角色信息到本地TXT"""
    try:
        current_dir = os.path.dirname(os.path.realpath(__file__))
        file_path = os.path.join(current_dir, "character_library.txt")
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        remark_str = remark_text.strip() if remark_text else "无"
        
        new_record = (
            f"[{timestamp}]\n"
            f"备注: {remark_str}\n"
            f"ID:   {char_id}\n"
            f"来源: {source_info}\n"
            f"----------------------------------------\n"
        )
        
        old_content = ""
        if os.path.exists(file_path):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    old_content = f.read()
            except:
                pass
        
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(new_record + old_content)
        return True
    except Exception as e:
        print(f"[Sora2 Record] 保存失败: {e}")
        return False

def _upload_local_file(api_key: str, file_path: str) -> str:
    if not os.path.exists(file_path):
        raise Exception(f"文件不存在: {file_path}")
    suffix = os.path.splitext(file_path)[1][1:].lower()
    if not suffix: suffix = "mp4"

    try:
        headers = _get_common_headers(api_key)
        print(f"[Sora2 Upload] 获取Token...")
        token_res = requests.post(f"{HOST}/client/resource/newUploadTokenZH", headers=headers, json={"sux": suffix}, timeout=30)
        token_res.raise_for_status()
        token_data = token_res.json().get("data")
        
        if not token_data:
            raise Exception("Token数据为空")
        token, key, up_url, domain = (token_data["token"], token_data["key"], token_data["url"], token_data["domain"])
        
        print(f"[Sora2 Upload] 上传文件中: {file_path}")
        with open(file_path, "rb") as f:
            requests.post(url=up_url, data={"token": token, "key": key}, files={"file": f}, timeout=300).raise_for_status()
            
        return f"{domain}/{key}"
    except Exception as e:
        print(f"[Sora2 Upload] 上传错误: {e}")
        raise e

def _poll_task_result(api_key: str, task_id: str, target_field: str = "url", timeout: int = 900):
    print(f"[Sora2 Poll] 轮询任务 {task_id}...")
    start_time = time.time()
    last_progress = 0
    pbar = comfy.utils.ProgressBar(100)
    
    while time.time() - start_time < timeout:
        time.sleep(3)
        try:
            status_response = requests.post(f"{HOST}/v1/draw/result", headers=_get_common_headers(api_key), json={"id": task_id}, timeout=30)
            if status_response.status_code != 200:
                continue
                
            status_data = status_response.json()
            if status_data.get("code") == -22:
                raise Exception(f"任务过期: {status_data.get('msg')}")
            
            task_info = status_data.get("data", {})
            status = task_info.get("status")
            progress = task_info.get("progress", 0)

            if progress and progress > last_progress:
                pbar.update_absolute(int(progress))
                last_progress = progress
            
            if status == "succeeded":
                pbar.update_absolute(100)
                results_list = task_info.get("results", [])
                if not results_list:
                    raise Exception("未返回结果数据")
                
                first_result = results_list[0]
                if target_field == "character_id":
                    return first_result.get("character_id"), task_info
                else:
                    return first_result.get("url"), task_info

            elif status == "failed":
                raise Exception(f"任务失败: {task_info.get('failure_reason')}")

        except Exception as e:
            if "任务失败" in str(e):
                raise e
            pass
            
    raise Exception("轮询超时")


# --- 主视频生成节点 ---
class Sora2Generator:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"default": "A cute cat playing on the grass", "multiline": True}),
                "aspect_ratio": (["16:9", "9:16"], {"default": "16:9"}),
                "duration": (["10", "15"], {"default": "10"}),
                "size": (["small", "large"], {"default": "small"}), 
            },
            "optional": {
                "image": ("IMAGE",),
                "remixTargetId": ("STRING", {"default": "默认无", "multiline": False, "placeholder": "视频续作的目标id (s_xxxx)"}),
                "api_key": ("STRING", {"default": "", "multiline": False, "placeholder": "留空则使用 __init__.py 中的配置"}),
            }
        }

    RETURN_TYPES = (IO.VIDEO, "STRING", "STRING")
    RETURN_NAMES = ("video", "video_url", "response")
    FUNCTION = "execute"
    CATEGORY = "Nkxx/视频"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return time.time_ns()

    def _get_credits_balance(self, api_key: str) -> str:
        try:
            url = f"{HOST}/client/common/getCredits?apikey={api_key}"
            res = requests.get(url, timeout=5)
            if res.status_code == 200 and res.json().get("code") == 0:
                return str(int(res.json()["data"].get("credits", 0)))
        except:
            pass
        return "查询失败"
    
    def _upload_image(self, api_key: str, image_tensor: torch.Tensor) -> str:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_f:
            pil_img = Image.fromarray(np.clip(255. * image_tensor.cpu().numpy(), 0, 255).astype(np.uint8))
            if pil_img.mode != 'RGB':
                pil_img = pil_img.convert('RGB')
            pil_img.save(temp_f, "PNG")
            temp_f_path = temp_f.name
        try:
            return _upload_local_file(api_key, temp_f_path)
        finally:
            if os.path.exists(temp_f_path):
                os.unlink(temp_f_path)
    
    def execute(self, prompt, aspect_ratio, duration, size, api_key="", image=None, remixTargetId=None):
        def get_api_key(user_key):
            if user_key and user_key.strip():
                return user_key.strip()
            return ""
            
        final_api_key = get_api_key(api_key) 
        if not final_api_key:
            return (GrsaiVideoAdapter(""), "", "状态: 失败\n错误: API Key 不能为空。")

        try:
            final_prompt = f"{prompt.strip()}{secrets.choice(ZERO_WIDTH_CHARS)}"
            payload = {
                "model": "sora-2", "prompt": final_prompt, 
                "aspectRatio": aspect_ratio, "duration": int(duration), 
                "size": size, "webHook": "-1", "removeWatermark": True
            }
            if image is not None:
                print("[Sora2] 上传参考图...")
                payload["url"] = self._upload_image(final_api_key, image[0])

            if remixTargetId:
                cid = remixTargetId.strip()
                if cid and cid != "默认无":
                    payload["remixTargetId"] = cid

            print("[Sora2] 提交任务...")
            res = requests.post(f"{HOST}/v1/video/sora-video", headers=_get_common_headers(final_api_key), json=payload, timeout=60)
            res.raise_for_status()
            data = res.json()
            if data.get("code") != 0:
                raise Exception(f"API提交失败: {data.get('msg')}")
            task_id = data.get("data", {}).get("id")
            
            video_url, task_info = _poll_task_result(final_api_key, task_id, target_field="url")
            pid = task_info.get("results", [{}])[0].get("pid", "")

            output_dir, filename = folder_paths.get_output_directory(), f"sora2_{uuid.uuid4().hex[:8]}.mp4"
            output_path = os.path.join(output_dir, filename)
            os.makedirs(output_dir, exist_ok=True)
            
            try:
                _robust_download_video(video_url, output_path, max_retries=3, timeout=300)
                return (GrsaiVideoAdapter(output_path), video_url, f"状态: success\n任务ID: {task_id}\nPID: {pid}\n积分: {self._get_credits_balance(final_api_key)}")
            except Exception as dl_err:
                 return (GrsaiVideoAdapter(""), video_url, f"状态: 生成成功但下载失败\nURL: {video_url}\nPID: {pid}")

        except Exception as e:
            traceback.print_exc()
            return (GrsaiVideoAdapter(""), "", f"状态: 失败\n错误: {e}")

# --- 节点 1: 上传创建角色 (路径模式 + 自动去引号) ---
class Sora2UploadCharacter:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # 简单直接：字符串输入框
                "video_path": ("STRING", {"default": "", "multiline": False, "placeholder": "请填入本地视频的绝对路径 (例如 E:\\video\\1.mp4)"}),
                "start_time": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "end_time": ("FLOAT", {"default": 3.0, "min": 0.1, "max": 100.0, "step": 0.1}),
            },
            "optional": {
                "api_key": ("STRING", {"default": "", "multiline": False, "placeholder": "API Key"}),
                "备注": ("STRING", {"default": "", "multiline": False, "placeholder": "角色备注 (将保存到本地TXT)"}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("character_id", "status")
    FUNCTION = "execute"
    CATEGORY = "Nkxx/视频"
    OUTPUT_NODE = True

    def execute(self, video_path, start_time, end_time, api_key="", 备注=""):
        if not api_key:
            return ("", "Error: API Key required")
        
        # 🔥 自动去除引号逻辑
        clean_path = video_path.strip().replace('"', '').replace("'", "")
        
        if not clean_path:
            return ("", "Error: 视频路径不能为空")
        
        if not os.path.exists(clean_path):
            return ("", f"Error: 文件不存在: {clean_path}")

        try:
            print(f"[Sora2 Char] 准备上传: {clean_path}")
            video_url = _upload_local_file(api_key, clean_path)
            
            ts_str = f"{float(start_time):g},{float(end_time):g}"
            payload = {
                "url": video_url, "timestamps": ts_str,
                "webHook": "-1", "shutProgress": False
            }
            
            print(f"[Sora2 Char] 提交角色任务 (范围: {ts_str})...")
            res = requests.post(f"{HOST}/v1/video/sora-upload-character", headers=_get_common_headers(api_key), json=payload, timeout=60)
            res.raise_for_status()
            res_json = res.json()
            if res_json.get("code") != 0:
                raise Exception(f"提交失败: {res_json.get('msg')}")
            task_id = res_json.get("data", {}).get("id")
            
            print(f"[Sora2 Char] 任务提交成功 ID: {task_id}，等待结果...")
            char_id, _ = _poll_task_result(api_key, task_id, target_field="character_id")
            
            save_msg = ""
            if char_id:
                _save_character_record(char_id, 备注, f"File: {os.path.basename(clean_path)}")
                save_msg = "\n(已保存到 character_library.txt)"

            return (char_id, f"Success\nID: {char_id}\n备注: {备注}{save_msg}")

        except Exception as e:
            traceback.print_exc()
            return ("", f"Error: {str(e)}")

# --- 节点 2: 从 PID 创建角色 (修复：已补充时间参数) ---
class Sora2FromPidCharacter:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pid": ("STRING", {"default": "", "multiline": False, "placeholder": "原视频PID (s_xxxx...)"}),
                # 🔥 补充：PID 模式也需要指定时间范围
                "start_time": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "end_time": ("FLOAT", {"default": 3.0, "min": 0.1, "max": 100.0, "step": 0.1}),
            },
            "optional": {
                "api_key": ("STRING", {"default": "", "multiline": False, "placeholder": "API Key"}),
                "备注": ("STRING", {"default": "", "multiline": False, "placeholder": "角色备注 (将保存到本地TXT)"}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("character_id", "status")
    FUNCTION = "execute"
    CATEGORY = "Nkxx/视频"
    OUTPUT_NODE = True

    def execute(self, pid, start_time, end_time, api_key="", 备注=""):
        if not api_key:
            return ("", "Error: API Key required")
        pid = pid.strip()
        if not pid:
            return ("", "Error: PID cannot be empty")

        try:
            # 🔥 补充：构造 timestamps 参数
            ts_str = f"{float(start_time):g},{float(end_time):g}"
            
            payload = {
                "pid": pid,
                "timestamps": ts_str, # 添加时间戳
                "webHook": "-1",
                "shutProgress": False
            }
            
            print(f"[Sora2 Char] 提交PID任务: {pid} (范围: {ts_str})")
            res = requests.post(f"{HOST}/v1/video/sora-create-character", headers=_get_common_headers(api_key), json=payload, timeout=60)
            res.raise_for_status()
            res_json = res.json()
            if res_json.get("code") != 0:
                raise Exception(f"提交失败: {res_json.get('msg')}")
            task_id = res_json.get("data", {}).get("id")
            
            char_id, _ = _poll_task_result(api_key, task_id, target_field="character_id")
            
            save_msg = ""
            if char_id:
                _save_character_record(char_id, 备注, f"PID: {pid}")
                save_msg = "\n(已保存到 character_library.txt)"

            return (char_id, f"Success\nID: {char_id}\n备注: {备注}{save_msg}")

        except Exception as e:
            traceback.print_exc()
            return ("", f"Error: {str(e)}")

# --- 节点映射 ---
NODE_CLASS_MAPPINGS = { 
    "Sora2Generator_Grsai": Sora2Generator,
    "Sora2UploadCharacter_Grsai": Sora2UploadCharacter,
    "Sora2FromPidCharacter_Grsai": Sora2FromPidCharacter
}

NODE_DISPLAY_NAME_MAPPINGS = { 
    "Sora2Generator_Grsai": "Sora2 Video Generator (Grsai)",
    "Sora2UploadCharacter_Grsai": "Sora2 上传创建角色 (Grsai)",
    "Sora2FromPidCharacter_Grsai": "Sora2 原视频创建角色 (Grsai)"
}