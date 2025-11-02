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

# --- 全局配置 ---
HOST = "https://grsai.dakka.com.cn"
ZERO_WIDTH_CHARS = [
    "\u200b",
    "\u200c",
    "\u200d",
    "\ufeff",
    "\u180e",
    "\u200e",
    "\u200f",
]
NODE_DIR = os.path.dirname(__file__)
TASK_FILE = os.path.join(NODE_DIR, "nkxx_task_history.json")
MAX_COMPLETED_HISTORY = 5 

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


def _robust_download_video(video_url: str, output_path: str, max_retries: int = 3, timeout: int = 300):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"
    }
    
    for attempt in range(1, max_retries + 1):
        try:
            print(f"[Sora2 Downloader] 尝试第 {attempt}/{max_retries} 次下载: {video_url}")
            with requests.get(video_url, stream=True, headers=headers, timeout=timeout) as response:
                response.raise_for_status()
                with open(output_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                print(f"[Sora2 Downloader] 视频下载成功: {output_path}")
                return True
        except requests.exceptions.RequestException as e:
            print(f"[Sora2 Downloader] 第 {attempt} 次下载失败: {e}")
            if isinstance(e, requests.exceptions.HTTPError) and 400 <= e.response.status_code < 500:
                print("[Sora2 Downloader] 遇到客户端错误 (如 403/404)，停止重试。")
                raise e
            if attempt < max_retries:
                wait_time = 5 * (2 ** (attempt - 1))
                print(f"[Sora2 Downloader] 等待 {wait_time} 秒后重试...")
                time.sleep(wait_time)
            else:
                print("[Sora2 Downloader] 达到最大重试次数，下载失败。")
                raise e
    return False


# --- 适配器 ---
class GrsaiVideoAdapter:
    def __init__(self, video_path: str):
        self.video_path = video_path
        self.width, self.height, self.fps, self.bit_rate = self._get_video_details(video_path)

    def _get_video_details(self, path):
        try:
            if not path or not os.path.exists(path):
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
            }, "optional": {
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "image": ("IMAGE",)}
            }
    RETURN_TYPES = ("STRING",); RETURN_NAMES = ("response",)
    FUNCTION = "submit"; CATEGORY = "Nkxx/视频/Sora2 Task Manager"
    @classmethod
    def IS_CHANGED(cls, **kwargs): return time.time_ns()
    def submit(self, prompt, aspect_ratio, duration, api_key="", image=None):
        final_api_key = get_api_key(api_key)
        if not final_api_key:
            return ("API Key 不能为空。请在节点中填写，或在 __init__.py 文件中设置。",)
        try:
            final_prompt = f"{prompt.strip()}{secrets.choice(ZERO_WIDTH_CHARS)}"
            payload = { "model": "sora-2", "prompt": final_prompt, "aspectRatio": aspect_ratio,
                        "duration": int(duration), "size": "small", "webHook": "-1" }
            if image is not None:
                payload["url"] = _upload_image(final_api_key, image[0])
            submit_response = requests.post(f"{HOST}/v1/video/sora-video", headers=_get_headers(final_api_key), json=payload, timeout=60)
            submit_response.raise_for_status(); submit_data = submit_response.json()
            if submit_data.get("code") != 0: raise Exception(f"API提交失败: {submit_data.get('msg', '未知错误')}")
            task_id = submit_data.get("data", {}).get("id")
            if not task_id: raise Exception("API未能返回有效的任务ID")
            tasks = _read_tasks()
            tasks[task_id] = { "prompt": prompt, "submitted_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "status": "pending" }
            trimmed_tasks = _trim_history(tasks)
            _write_tasks(trimmed_tasks)
            return (f"任务提交成功!\nID: {task_id}\n请使用 '查询任务状态' 节点刷新。",)
        except Exception as e:
            return (f"提交失败: {e}",)


# --- 节点 2: 查询任务状态 ---
class Sora2QueryTasks:
    @classmethod
    def INPUT_TYPES(cls):
        return {"optional": {"api_key": ("STRING", {"default": "", "multiline": False})}}

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
                            tasks[task_id]['video_url'] = data.get("results", [{}])[0].get("url")
                        elif new_status == "failed":
                            tasks[task_id]['failure_reason'] = data.get("failure_reason", "未知")
                        updated = True
            except Exception:
                pass
        if updated: _write_tasks(tasks)

        final_tasks_for_report = _trim_history(_read_tasks())
        _write_tasks(final_tasks_for_report) 
        
        full_report_lines = ["--- 任务队列总览 ---"]
        sorted_tasks_report = sorted(final_tasks_for_report.items(), key=lambda item: item[1].get('submitted_at', ''), reverse=True)
        for tid, tinfo in sorted_tasks_report:
            status = tinfo.get('status', 'N/A')
            status_str = f"running {tinfo.get('progress', 0)}%" if status == 'running' else status
            full_report_lines.append(f"[{status_str}] {tid[:8]}... - {tinfo.get('prompt', '')[:25]}...")
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
            return (GrsaiVideoAdapter(None), f"错误: 任务 {task_to_download_id[:8]}... 状态成功但无URL。")

        try:
            output_dir = folder_paths.get_output_directory()
            filename = f"sora2_{task_to_download_id[:8]}_{uuid.uuid4().hex[:4]}.mp4"
            output_path = os.path.join(output_dir, filename)
            
            _robust_download_video(video_url, output_path, max_retries=3, timeout=300)
            
            tasks[task_to_download_id]['status'] = 'downloaded'
            tasks[task_to_download_id]['video_path'] = output_path
            _write_tasks(tasks)
            
            video_adapter = GrsaiVideoAdapter(output_path)
            return (video_adapter, f"下载成功: {task_to_download_id[:8]}...")
            
        except Exception as e:
            tasks[task_to_download_id]['status'] = 'download_failed'
            _write_tasks(tasks)
            print(f"[Sora2 GetNextVideo] 视频下载失败（已重试）。错误: {e}")
            traceback.print_exc()
            return (GrsaiVideoAdapter(None), f"下载失败: {task_to_download_id[:8]}... 错误: {e}")


# --- 节点注册 ---
NODE_CLASS_MAPPINGS = {
    "Sora2SubmitAndRecordTask": Sora2SubmitAndRecordTask,
    "Sora2QueryTasks": Sora2QueryTasks,
    "Sora2GetNextVideo": Sora2GetNextVideo,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "Sora2SubmitAndRecordTask": "1. Sora2 提交任务 (Grsai)",
    "Sora2QueryTasks": "2. Sora2 查询任务状态 (Grsai)",
    "Sora2GetNextVideo": "3. Sora2 获取下一个视频 (Grsai)",
}