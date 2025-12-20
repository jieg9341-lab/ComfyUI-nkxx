import os
import requests
import time
import uuid
import folder_paths
import traceback
import shutil
import cv2
from comfy.comfy_types import IO
import tempfile
from PIL import Image
import torch
import numpy as np
import secrets
import json
from datetime import datetime
import pandas as pd 
import concurrent.futures 
import threading 
import subprocess
import sys
import importlib
from . import get_api_key 
import yt_dlp

# --- SSL配置 ---
import ssl
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
ssl._create_default_https_context = ssl._create_unverified_context

# --- 全局配置 ---
HOST = "https://grsai.dakka.com.cn"
ZERO_WIDTH_CHARS = ["\u200b", "\u200c", "\u200d", "\ufeff", "\u180e", "\u200e", "\u200f"]
NODE_DIR = os.path.dirname(__file__)
TASK_FILE = os.path.join(NODE_DIR, "sora2_task_history.json") 
MAX_COMPLETED_HISTORY = 5 
task_file_lock = threading.Lock() 

# --- 辅助函数 ---
def _get_headers(api_key: str) -> dict:
    return {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}

def _get_credits_balance(api_key: str) -> str:
    try:
        url = f"{HOST}/client/common/getCredits?apikey={api_key}"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if data.get("code") == 0 and "data" in data and "credits" in data["data"]:
                return str(int(data["data"]["credits"]))
    except Exception as e:
        print(f"[Nkxx Task Manager] 查询积分失败: {e}")
    return "查询失败"

def _read_tasks():
    if not os.path.exists(TASK_FILE): return {}
    try:
        with open(TASK_FILE, 'r', encoding='utf-8') as f:
            tasks = json.load(f)
        return tasks if isinstance(tasks, dict) else {}
    except (json.JSONDecodeError, IOError):
        return {}

def _write_tasks(tasks):
    try:
        with open(TASK_FILE, 'w', encoding='utf-8') as f:
            json.dump(tasks, f, indent=4, ensure_ascii=False, sort_keys=True)
    except IOError as e:
        print(f"[Nkxx Task Manager] 写入任务文件失败: {e}")

def _trim_history(tasks):
    in_progress_statuses = ["pending", "running", "succeeded"]
    in_progress_tasks = {tid: t for tid, t in tasks.items() if t.get("status") in in_progress_statuses}
    completed_tasks = {tid: t for tid, t in tasks.items() if t.get("status") not in in_progress_statuses}
    if len(completed_tasks) > MAX_COMPLETED_HISTORY:
        sorted_completed = sorted(completed_tasks.items(), key=lambda item: item[1].get('submitted_at', '1970-01-01 00:00:00'), reverse=True)
        kept_completed = dict(sorted_completed[:MAX_COMPLETED_HISTORY])
        return {**in_progress_tasks, **kept_completed}
    else:
        return tasks

def _upload_image(api_key: str, image_tensor: torch.Tensor) -> str:
    i = 255. * image_tensor.cpu().numpy(); img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
    if img.mode != 'RGB': img = img.convert('RGB')
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_f:
        img.save(temp_f, "PNG"); temp_f_path = temp_f.name
    try:
        headers = _get_headers(api_key)
        token_res = requests.post(f"{HOST}/client/resource/newUploadTokenZH", headers=headers, json={"sux": "png"}, timeout=30)
        token_res.raise_for_status(); token_data = token_res.json()["data"]
        token, key, up_url, domain = (token_data["token"], token_data["key"], token_data["url"], token_data["domain"])
        with open(temp_f_path, "rb") as f:
            requests.post(url=up_url, data={"token": token, "key": key}, files={"file": f}, timeout=120).raise_for_status()
        return f"{domain}/{key}"
    finally:
        if os.path.exists(temp_f_path): os.unlink(temp_f_path)


# --- 🔥 核心下载函数：修复版 (yt-dlp优先 + 反死锁Curl) ---
def _robust_download_video(video_url: str, output_path: str, max_retries: int = 3, timeout: int = 300):
    print(f"[Sora2 Downloader] 准备下载: {video_url}")
    
    # --- 方案 1: 优先尝试 yt-dlp ---
    # yt-dlp 是纯Python库，通常更稳定，不易产生IO管道死锁
    try:
        print(f"[Sora2 Downloader] 正在使用 yt-dlp 下载...")
        
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Referer": "https://grsai.dakka.com.cn/"
        }
        
        ydl_opts = {
            'outtmpl': output_path,
            'retries': max_retries,
            'http_headers_str': "\r\n".join([f"{k}: {v}" for k, v in headers.items()]),
            'quiet': True, # 开启日志以便调试，若卡顿可改为True
            'noplaylist': True,
            'merge_output_format': 'mp4',
            'socket_timeout': 30, # 单次连接超时
            'nocheckcertificate': True, 
        }

            
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([video_url])
        
        # 验证文件有效性（大小 > 1KB）
        if os.path.exists(output_path) and os.path.getsize(output_path) > 1024:
             print(f"[Sora2 Downloader] ✓ yt-dlp 下载成功")
             return True
        else:
            print("[Sora2 Downloader] yt-dlp 完成但文件无效，尝试备用方案...")
            
    except Exception as e:
        print(f"[Sora2 Downloader] yt-dlp 下载失败或未安装: {e}。切换至 Curl 备用方案。")
        # 清理可能的残留文件
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
                    print(f"[Sora2 Downloader] ✓ Curl 下载成功")
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


# --- 视频对象适配器 (增加坏文件防御) ---
class GrsaiVideoAdapter:
    def __init__(self, video_path: str):
        self.video_path = video_path
        self.width, self.height, self.fps, self.bit_rate = self._get_video_details(video_path)

    def _get_video_details(self, path):
        try:
            if not path or not os.path.exists(path):
                return 1280, 720, 30, 0 
            
            # [新增] 防止 0KB 文件导致 cv2 卡死
            if os.path.getsize(path) < 1024:
                 print(f"[GrsaiVideoAdapter] 警告: 视频文件无效/过小 ({os.path.getsize(path)} bytes)，跳过解析")
                 return 1280, 720, 30, 0

            cap = cv2.VideoCapture(path)
            if not cap.isOpened():
                return 1280, 720, 30, 0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            bit_rate = 0 
            cap.release()
            if fps == 0:
                fps = 30
            return width, height, fps, bit_rate
        except Exception as e:
            print(f"[GrsaiVideoAdapter] 读取视频详情失败: {e}")
            return 1280, 720, 30, 0 

    def get_dimensions(self):
        return self.width, self.height

    def save_to(self, output_path, **kwargs):
        try:
            if self.video_path and os.path.exists(self.video_path):
                shutil.copyfile(self.video_path, output_path)
                return True
            return False
        except Exception as e:
            print(f"[GrsaiVideoAdapter] 保存视频时出错: {e}")
            return False


# --- 节点 1: 提交任务 ---
class Sora2SubmitAndRecordTask:
    @classmethod
    def INPUT_TYPES(cls):
        return { "required": {
                "prompt": ("STRING", {"default": "A cute cat playing on the grass", "multiline": True}),
                "aspect_ratio": (["16:9", "9:16"], {"default": "16:9"}),
                "duration": (["10", "15"], {"default": "10"}),
                "size": (["small", "large"], {"default": "small"}), 
            }, "optional": {
                "image": ("IMAGE",),
                "remixTargetId": ("STRING", {"default": "默认无", "multiline": False, "placeholder": "视频续作的目标id (s_xxxx)"}),
                "api_key": ("STRING", {"default": "", "multiline": False, "placeholder": "留空则使用 __init__.py 中的配置"}),
            }
        }
    RETURN_TYPES = ("STRING",); RETURN_NAMES = ("response",)
    FUNCTION = "submit"; CATEGORY = "Nkxx/视频/Sora2 Task Manager"
    @classmethod
    def IS_CHANGED(cls, **kwargs): return time.time_ns()

    def submit(self, prompt, aspect_ratio, duration, size, api_key="", image=None, remixTargetId=None):
        final_api_key = get_api_key(api_key)
        if not final_api_key:
            return ("API Key 不能为空。请在节点中填写，或在 __init__.py 文件中设置。",)
        try:
            final_prompt = f"{prompt.strip()}{secrets.choice(ZERO_WIDTH_CHARS)}"
            payload = { 
                "model": "sora-2", 
                "prompt": final_prompt, 
                "aspectRatio": aspect_ratio,
                "duration": int(duration), 
                "size": size, 
                "webHook": "-1",
                "removeWatermark": True 
            }
            if image is not None:
                payload["url"] = _upload_image(final_api_key, image[0])
            
            if remixTargetId:
                cid = remixTargetId.strip()
                if cid and cid != "默认无": 
                    payload["remixTargetId"] = cid
                    print(f"[Sora2 Task Submitter] 已添加视频续作ID: {cid}")

            submit_response = requests.post(f"{HOST}/v1/video/sora-video", headers=_get_headers(final_api_key), json=payload, timeout=60)
            submit_response.raise_for_status(); submit_data = submit_response.json()
            if submit_data.get("code") != 0: raise Exception(f"API提交失败: {submit_data.get('msg', '未知错误')}")
            task_id = submit_data.get("data", {}).get("id")
            if not task_id: raise Exception("API未能返回有效的任务ID")
            
            with task_file_lock:
                tasks = _read_tasks()
                tasks[task_id] = { "prompt": prompt, "submitted_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "status": "pending" }
                trimmed_tasks = _trim_history(tasks)
                _write_tasks(trimmed_tasks)
                
            return (f"任务提交成功!\nID: {task_id}\n请使用 '查询任务状态' 节点刷新。",)
        except Exception as e:
            return (f"提交失败: {e}",)


# --- 节点 1.5: 批量提交任务 ---
class Sora2SubmitBatchTask:
    @classmethod
    def INPUT_TYPES(cls):
        return { "required": {
                "file_path": ("STRING", {"default": "", "placeholder": "拖拽 CSV/Excel 文件至此"}),
                "column_name": ("STRING", {"default": "prompt"}),
                "aspect_ratio": (["16:9", "9:16"], {"default": "16:9"}),
                "duration": (["10", "15"], {"default": "10"}),
                "size": (["small", "large"], {"default": "small"}), 
                "concurrency": ("INT", {"default": 5, "min": 1, "max": 20, "step": 1}),
            }, "optional": {
                "prompt_prefix": ("STRING", {"multiline": True, "default": ""}),
                "max_count": ("INT", {"default": 50, "min": 1, "max": 999}),
                "image": ("IMAGE",), 
                "remixTargetId": ("STRING", {"default": "默认无", "multiline": False, "placeholder": "(可选) 应用于所有任务的续作id"}),
                "api_key": ("STRING", {"default": "", "multiline": False, "placeholder": "留空则使用 __init__.py 中的配置"}),
            }
        }
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("task_ids", "report")
    FUNCTION = "submit_batch"
    CATEGORY = "Nkxx/视频/Sora2 Task Manager"
    
    @classmethod
    def IS_CHANGED(cls, **kwargs): return time.time_ns()

    def submit_batch(self, file_path, column_name, aspect_ratio, duration, size, concurrency, 
                     api_key="", prompt_prefix="", max_count=50, image=None, remixTargetId=None):
        
        final_api_key = get_api_key(api_key)
        if not final_api_key:
            return ("", "API Key 不能为空。")

        if not file_path or not os.path.exists(file_path):
            return ("", "文件路径为空或文件不存在。")
        
        try:
            if file_path.lower().endswith('.csv'): 
                df = pd.read_csv(file_path, encoding='utf-8')
            elif file_path.lower().endswith(('.xls', '.xlsx')): 
                df = pd.read_excel(file_path)
            else: 
                return ("", "仅支持 .csv, .xls, .xlsx 文件。")
        except Exception as e: 
            return ("", f"读取文件失败: {e}")
        
        if column_name not in df.columns: 
            return ("", f"列 '{column_name}' 不存在。")
        
        prompts = [f"{prompt_prefix}{p}" for p in df[column_name].dropna().astype(str).tolist()[:max_count]]
        if not prompts: 
            return ("", f"列 '{column_name}' 中未找到有效 prompt。")

        uploaded_image_url = None
        try:
            if image is not None:
                uploaded_image_url = _upload_image(final_api_key, image[0])
        except Exception as e:
            return ("", f"图像上传失败: {e}")

        final_remix_id = None
        if remixTargetId:
            cid = remixTargetId.strip()
            if cid and cid != "默认无":
                final_remix_id = cid
                
        if final_remix_id:
            print(f"[Sora2 Batch Submitter] 将为所有任务应用续作ID: {final_remix_id}")

        def submit_task_internal(prompt):
            try:
                final_prompt = f"{prompt.strip()}{secrets.choice(ZERO_WIDTH_CHARS)}"
                payload = { 
                    "model": "sora-2", 
                    "prompt": final_prompt, 
                    "aspectRatio": aspect_ratio,
                    "duration": int(duration), 
                    "size": size, 
                    "webHook": "-1",
                    "removeWatermark": True
                }
                if uploaded_image_url:
                    payload["url"] = uploaded_image_url
                
                if final_remix_id:
                    payload["remixTargetId"] = final_remix_id
                
                submit_response = requests.post(f"{HOST}/v1/video/sora-video", headers=_get_headers(final_api_key), json=payload, timeout=60)
                submit_response.raise_for_status()
                submit_data = submit_response.json()

                if submit_data.get("code") != 0:
                    return (prompt, f"API失败: {submit_data.get('msg', '未知错误')}")
                
                task_id = submit_data.get("data", {}).get("id")
                if not task_id:
                    return (prompt, "API未返回有效ID")

                with task_file_lock:
                    tasks = _read_tasks()
                    tasks[task_id] = { 
                        "prompt": prompt, 
                        "submitted_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 
                        "status": "pending" 
                    }
                    trimmed_tasks = _trim_history(tasks)
                    _write_tasks(trimmed_tasks)
                
                return (prompt, task_id)
            except Exception as e:
                return (prompt, f"提交异常: {str(e)}")

        all_results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
            all_results = list(executor.map(submit_task_internal, prompts))

        success_count = 0
        fail_count = 0
        output_lines = []

        for _prompt, result in all_results:
            if "失败" in result or "异常" in result or "错误" in result or "未返回" in result:
                fail_count += 1
                output_lines.append(result) 
            else:
                success_count += 1
                output_lines.append(result) 
        
        report = f"批量提交完成 | 总数: {len(prompts)} | 成功: {success_count} | 失败: {fail_count}"
        task_ids_string = "\n".join(output_lines)
        
        return (task_ids_string, report)


# --- 节点 2: 查询任务状态 ---
class Sora2QueryTasks:
    @classmethod
    def INPUT_TYPES(cls):
        return {"optional": {"api_key": ("STRING", {"default": "", "multiline": False, "placeholder": "留空则使用 __init__.py 中的配置"})}}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("report",)
    FUNCTION = "query_tasks"
    CATEGORY = "Nkxx/视频/Sora2 Task Manager"
    
    @classmethod
    def IS_CHANGED(cls, **kwargs): return time.time_ns()

    def query_tasks(self, api_key=""):
        final_api_key = get_api_key(api_key)
        if not final_api_key:
            return ("API Key 不能为空。",)
        
        tasks = _read_tasks(); updated = False
        
        tasks_to_query = {tid: t for tid, t in tasks.items() if t.get("status") not in ["downloaded", "failed", "download_failed"]}
        for task_id, task_info in tasks_to_query.items():
            if task_info.get("status") == "succeeded": continue
            try:
                status_response = requests.post(f"{HOST}/v1/draw/result", headers=_get_headers(final_api_key), json={"id": task_id}, timeout=30)
                status_data = status_response.json()
                if status_data.get("code") == 0 and "data" in status_data:
                    data = status_data["data"]
                    new_status = data.get("status")
                    new_progress = data.get("progress", 0)
                    old_status = tasks[task_id].get("status")
                    old_progress = tasks[task_id].get("progress", 0)
                    
                    if old_status != new_status or (new_status == 'running' and old_progress != new_progress):
                        tasks[task_id]['status'] = new_status
                        tasks[task_id]['progress'] = new_progress
                        if new_status == "succeeded":
                            first_result = data.get("results", [{}])[0]
                            tasks[task_id]['video_url'] = first_result.get("url")
                            tasks[task_id]['pid'] = first_result.get("pid", "") 
                        elif new_status == "failed":
                            tasks[task_id]['failure_reason'] = data.get("failure_reason", "未知")
                        updated = True
            except Exception:
                pass
        
        if updated: 
            _write_tasks(tasks)

        final_tasks_for_report = _trim_history(_read_tasks())
        
        if updated: 
             _write_tasks(final_tasks_for_report) 
        
        full_report_lines = ["--- 任务队列总览 ---"]
        sorted_tasks_report = sorted(final_tasks_for_report.items(), key=lambda item: item[1].get('submitted_at', ''), reverse=True)
        for tid, tinfo in sorted_tasks_report:
            status = tinfo.get('status', 'N/A')
            status_str = f"running {tinfo.get('progress', 0)}%" if status == 'running' else status
            
            pid_str = ""
            if status in ["succeeded", "downloaded", "download_failed"]:
                pid = tinfo.get("pid")
                if pid:
                    pid_str = f" (PID: {pid})"
            
            full_report_lines.append(f"[{status_str}] {tid[:8]}...{pid_str} - {tinfo.get('prompt', '')[:25]}...")
            
        full_report_lines.append(f"\n当前剩余积分: {_get_credits_balance(final_api_key)}")
        
        return ("\n".join(full_report_lines),)

# --- 节点 3: 获取下一个视频 ---
class Sora2GetNextVideo:
    @classmethod
    def INPUT_TYPES(cls):
        return {}

    RETURN_TYPES = (IO.VIDEO, "STRING")
    RETURN_NAMES = ("video", "report",)
    FUNCTION = "get_video"
    CATEGORY = "Nkxx/视频/Sora2 Task Manager"

    @classmethod
    def IS_CHANGED(cls, **kwargs): return time.time_ns()

    def get_video(self):
        tasks = _read_tasks()
        
        sorted_tasks = sorted(tasks.items(), key=lambda item: item[1].get('submitted_at', ''))
        task_to_download_id = next((tid for tid, t in sorted_tasks if t.get("status") == "succeeded"), None)

        if not task_to_download_id:
            return (GrsaiVideoAdapter(None), "当前无新完成的任务可供下载。")

        task_info = tasks[task_to_download_id]
        video_url = task_info.get("video_url")

        if not video_url:
            tasks[task_to_download_id]['status'] = 'failed'
            _write_tasks(tasks)
            return (GrsaiVideoAdapter(None), f"错误: Task {task_to_download_id[:8]}... 状态成功但无URL。")

        try:
            output_dir = folder_paths.get_output_directory()
            filename = f"sora2_{task_to_download_id[:8]}_{uuid.uuid4().hex[:4]}.mp4"
            output_path = os.path.join(output_dir, filename)
            
            _robust_download_video(video_url, output_path, max_retries=3, timeout=300)
            
            tasks[task_to_download_id]['status'] = 'downloaded'
            tasks[task_to_download_id]['video_path'] = output_path
            _write_tasks(tasks)
            
            video_adapter = GrsaiVideoAdapter(output_path)
            
            pid_info = f" (PID: {task_info.get('pid')})" if task_info.get('pid') else ""
            report_msg = f"下载成功: {task_to_download_id[:8]}...{pid_info}"
            
            return (video_adapter, report_msg)
            
        except Exception as e:
            tasks[task_to_download_id]['status'] = 'download_failed'
            _write_tasks(tasks)
            print(f"[Sora2 GetNextVideo] 视频下载失败（已重试）。错误: {e}")
            traceback.print_exc()
            
            error_report = (
                f"❌ 自动下载失败: {task_to_download_id[:8]}...\n"
                f"原因: {str(e)[:50]}...\n\n"
                f"⚠️ 请复制下方链接手动下载:\n"
                f"{video_url}"
            )
            return (GrsaiVideoAdapter(None), error_report)


# --- 节点注册 ---
NODE_CLASS_MAPPINGS = {
    "Sora2SubmitAndRecordTask": Sora2SubmitAndRecordTask,
    "Sora2SubmitBatchTask": Sora2SubmitBatchTask, 
    "Sora2QueryTasks": Sora2QueryTasks,
    "Sora2GetNextVideo": Sora2GetNextVideo,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "Sora2SubmitAndRecordTask": "1. Sora2 提交任务 (Grsai)",
    "Sora2SubmitBatchTask": "1.5. Sora2 批量提交 (CSV/Excel)", 
    "Sora2QueryTasks": "2. Sora2 查询任务状态 (Grsai)",
    "Sora2GetNextVideo": "3. Sora2 获取下一个视频 (Grsai)",
}