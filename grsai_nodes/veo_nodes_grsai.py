# grsai_nodes/veo_nodes_grsai.py
import os
import time
import uuid
import secrets
import requests
import traceback
from datetime import datetime
import folder_paths
import comfy.utils
import torch
from comfy.comfy_types import IO

import builtins

# 从公共工具箱导入需要的核心函数
from ..utils import (
    TaskManager,
    upload_image_grsai,
    robust_download_video,
    VideoAdapter
)

# --- 模型列表定义 ---
# 列表 A: 适用于首尾帧任务 (支持 Pro)
VEO_MODELS_ALL = [
    "veo3.1-fast", "veo3.1-fast-1080p", "veo3.1-fast-4k", 
    "veo3.1-pro", "veo3.1-pro-1080p", "veo3.1-pro-4k"
]

# 列表 B: 适用于多参任务 (仅支持 Fast, 不支持 Pro)
VEO_MODELS_FAST_ONLY = [
    "veo3.1-fast", "veo3.1-fast-1080p", "veo3.1-fast-4k"
]

ZERO_WIDTH_CHARS = ["\u200b", "\u200c", "\u200d", "\ufeff", "\u180e", "\u200e", "\u200f"]

# 初始化专属的异步任务管理器
veo_task_db = TaskManager("grsai_veo_history.json", max_completed_history=10)

# --- 核心 API 客户端 ---
class GrsaiVeoAPI:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.host = "https://grsai.dakka.com.cn"
        self.headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}

    def submit_veo_task(self, payload: dict) -> str:
        res = requests.post(f"{self.host}/v1/video/veo", headers=self.headers, json=payload, timeout=60)
        res.raise_for_status()
        data = res.json()
        if data.get("code") != 0:
            raise Exception(f"API提交任务失败: {data.get('msg', '未知错误')}")
        task_id = data.get("data", {}).get("id")
        if not task_id:
            raise Exception("API未能返回有效的任务ID。")
        return task_id

    def query_task_status(self, task_id: str) -> dict:
        res = requests.post(f"{self.host}/v1/draw/result", headers=self.headers, json={"id": task_id}, timeout=30)
        res.raise_for_status()
        return res.json()

    def get_credits_balance(self) -> str:
        try:
            url = f"{self.host}/client/common/getCredits?apikey={self.api_key}"
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data.get("code") == 0 and "data" in data and "credits" in data["data"]:
                    return str(int(data["data"]["credits"]))
        except Exception as e:
            print(f"[Veo API] 查询积分失败: {e}")
        return "查询失败"


# ======================================================================
# 同步阻塞执行基类 (供同步节点复用)
# ======================================================================
def _execute_veo_task_sync(api_key: str, payload: dict):
    """通用的同步轮询执行器"""
    api = GrsaiVeoAPI(api_key)
    pbar = comfy.utils.ProgressBar(100)
    pbar.update_absolute(20)
    
    try:
        print(f"[Veo Sync] 正在提交视频生成任务...")
        task_id = api.submit_veo_task(payload)
        print(f"[Veo Sync] 任务提交成功, 任务ID: {task_id}")
        pbar.update_absolute(30)

        print("[Veo Sync] 开始轮询任务结果...")
        last_progress, start_time, timeout = 0, time.time(), 900

        while time.time() - start_time < timeout:
            time.sleep(5)
            status_data = api.query_task_status(task_id)
            
            if status_data.get("code") != 0:
                raise Exception(f"轮询失败: {status_data.get('msg', status_data.get('code'))}")
            
            task_info = status_data.get("data", {})
            if not task_info: continue

            status = task_info.get("status")
            progress = task_info.get("progress", 0)

            if progress > last_progress:
                pbar.update_absolute(30 + int(progress * 0.6))
                last_progress = progress
            
            if status == "succeeded":
                pbar.update_absolute(90)
                video_url = task_info.get("url")
                if not video_url: raise Exception("任务成功但未找到视频URL。")
                
                try:
                    output_dir = folder_paths.get_output_directory()
                    filename = f"veo3.1_{uuid.uuid4().hex[:8]}.mp4"
                    output_path = os.path.join(output_dir, filename)
                    os.makedirs(output_dir, exist_ok=True)
                    
                    robust_download_video(video_url, output_path, max_retries=3)
                    pbar.update_absolute(100)
                    
                    credits_text = api.get_credits_balance()
                    status_msg = f"状态: 成功\n任务ID: {task_id}\n视频已下载\n剩余积分: {credits_text}"
                    return (VideoAdapter(output_path), video_url, status_msg)
                except Exception as e:
                    pbar.update_absolute(100)
                    credits_text = api.get_credits_balance()
                    return (VideoAdapter(""), video_url, f"状态: 下载失败\n任务: {task_id}\n错误: {e}\n积分: {credits_text}")

            elif status == "failed":
                fail_reason = task_info.get('failure_reason', '未知原因')
                reason_map = {"output_moderation": "输出违规", "input_moderation": "提示词违规", "error": "其他错误"}
                full_error_msg = f"生成失败: {reason_map.get(fail_reason, fail_reason)}"
                if task_info.get('error'): full_error_msg += f" | {task_info.get('error')}"
                raise Exception(full_error_msg)

        raise Exception(f"轮询超时 ({timeout}秒)，任务未完成。")

    except Exception as e:
        traceback.print_exc()
        return (VideoAdapter(""), "", f"状态: 失败\n错误信息: {e}")


# ======================================================================
# 【同步节点 1】: 参考图同步生成
# ======================================================================
class Veo3_1_RefGenerator_Sync:
    CATEGORY = "Nkxx/Grsai/视频 (同步)"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"default": "A cute cat playing on the grass", "multiline": True}),
                "model": (VEO_MODELS_FAST_ONLY, {"default": "veo3.1-fast"}),
                "aspect_ratio": (["16:9", "9:16"], {"default": "16:9"}),
            },
            "optional": {
                "api_key": ("STRING", {"default": "", "multiline": False, "placeholder": "留空则读取全局配置"}),
                "ref_image_1": ("IMAGE",), "ref_image_2": ("IMAGE",), "ref_image_3": ("IMAGE",),
            }
        }

    RETURN_TYPES = (IO.VIDEO, "STRING", "STRING")
    RETURN_NAMES = ("video", "video_url", "response")
    FUNCTION = "execute"
    @classmethod
    def IS_CHANGED(cls, **kwargs): return time.time_ns()

    def execute(self, prompt, model, aspect_ratio, api_key="", ref_image_1=None, ref_image_2=None, ref_image_3=None):
        final_api_key = builtins.get_api_key(api_key, "grsai")
        if not final_api_key: return (VideoAdapter(""), "", "未配置 Grsai API Key。")

        try:
            ref_urls = []
            for img in [ref_image_1, ref_image_2, ref_image_3]:
                if img is not None:
                    url = upload_image_grsai(final_api_key, img[0])
                    if url: ref_urls.append(url)
                    else: raise Exception("部分参考图上传失败。")

            payload = {
                "model": model,
                "prompt": f"{prompt.strip()}{secrets.choice(ZERO_WIDTH_CHARS)}",
                "aspectRatio": aspect_ratio,
                "webHook": "-1"
            }
            if ref_urls: payload["urls"] = ref_urls
            
            return _execute_veo_task_sync(final_api_key, payload)
        except Exception as e:
            return (VideoAdapter(""), "", f"执行失败: {e}")


# ======================================================================
# 【同步节点 2】: 首尾帧同步生成
# ======================================================================
class Veo3_1_FramesGenerator_Sync:
    CATEGORY = "Nkxx/Grsai/视频 (同步)"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"default": "A cinematic drone shot", "multiline": True}),
                "model": (VEO_MODELS_ALL, {"default": "veo3.1-fast"}),
                "aspect_ratio": (["16:9", "9:16"], {"default": "16:9"}),
            },
            "optional": {
                "api_key": ("STRING", {"default": "", "placeholder": "留空则读取全局配置"}),
                "first_frame": ("IMAGE",), "last_frame": ("IMAGE",),
            }
        }

    RETURN_TYPES = (IO.VIDEO, "STRING", "STRING")
    RETURN_NAMES = ("video", "video_url", "response")
    FUNCTION = "execute"
    @classmethod
    def IS_CHANGED(cls, **kwargs): return time.time_ns()

    def execute(self, prompt, model, aspect_ratio, api_key="", first_frame=None, last_frame=None):
        final_api_key = builtins.get_api_key(api_key, "grsai")
        if not final_api_key: return (VideoAdapter(""), "", "未配置 Grsai API Key。")

        try:
            first_url = upload_image_grsai(final_api_key, first_frame[0]) if first_frame is not None else None
            last_url = upload_image_grsai(final_api_key, last_frame[0]) if last_frame is not None else None
            
            if last_url and not first_url:
                raise Exception("不支持仅使用尾帧生成。必须同时提供首帧。")

            payload = {
                "model": model,
                "prompt": f"{prompt.strip()}{secrets.choice(ZERO_WIDTH_CHARS)}",
                "aspectRatio": aspect_ratio,
                "webHook": "-1"
            }
            if first_url: payload["firstFrameUrl"] = first_url
            if last_url: payload["lastFrameUrl"] = last_url
            
            return _execute_veo_task_sync(final_api_key, payload)
        except Exception as e:
            return (VideoAdapter(""), "", f"执行失败: {e}")


# ======================================================================
# 【异步节点 1】: 首尾帧异步提交
# ======================================================================
class VeoFramesSubmitTask_Async:
    CATEGORY = "Nkxx/Grsai/视频 (异步)"
    
    @classmethod
    def INPUT_TYPES(cls):
        return { "required": {
                "prompt": ("STRING", {"default": "A cinematic drone shot of a futuristic city", "multiline": True}),
                "model": (VEO_MODELS_ALL, {"default": "veo3.1-fast"}),
                "aspect_ratio": (["16:9", "9:16"], {"default": "16:9"}),
            }, "optional": {
                "api_key": ("STRING", {"default": "", "placeholder": "留空则读取全局配置"}),
                "first_frame": ("IMAGE",), "last_frame": ("IMAGE",),
            }
        }
    RETURN_TYPES = ("STRING",); RETURN_NAMES = ("response",)
    FUNCTION = "submit"
    @classmethod
    def IS_CHANGED(cls, **kwargs): return time.time_ns()

    def submit(self, prompt, model, aspect_ratio, api_key="", first_frame=None, last_frame=None):
        final_api_key = builtins.get_api_key(api_key, "grsai")
        if not final_api_key: return ("API Key 不能为空。",)
        
        try:
            first_url = upload_image_grsai(final_api_key, first_frame[0]) if first_frame is not None else None
            last_url = upload_image_grsai(final_api_key, last_frame[0]) if last_frame is not None else None

            if last_url and not first_url:
                return ("提交失败: Veo 不支持仅使用尾帧生成。若使用尾帧控制，必须同时提供首帧。",)

            payload = { 
                "model": model, 
                "prompt": f"{prompt.strip()}{secrets.choice(ZERO_WIDTH_CHARS)}", 
                "aspectRatio": aspect_ratio, 
                "webHook": "-1"
            }
            if first_url: payload["firstFrameUrl"] = first_url
            if last_url: payload["lastFrameUrl"] = last_url

            api = GrsaiVeoAPI(final_api_key)
            task_id = api.submit_veo_task(payload)
            
            veo_task_db.update_task(task_id, {
                "prompt": prompt, 
                "model": model,  
                "submitted_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 
                "status": "pending", 
                "type": "frames" 
            })
                
            return (f"Veo首尾帧任务(异步)提交成功!\nID: {task_id}\n请使用 '异步查询' 节点刷新。",)
        except Exception as e:
            return (f"提交失败: {e}",)


# ======================================================================
# 【异步节点 2】: 多参生视频异步提交
# ======================================================================
class VeoMultiRefSubmitTask_Async:
    CATEGORY = "Nkxx/Grsai/视频 (异步)"
    
    @classmethod
    def INPUT_TYPES(cls):
        return { "required": {
                "prompt": ("STRING", {"default": "A character turnaround", "multiline": True}),
                "model": (VEO_MODELS_FAST_ONLY, {"default": "veo3.1-fast"}),
                "aspect_ratio": (["16:9", "9:16"], {"default": "16:9"}),
            }, "optional": {
                "api_key": ("STRING", {"default": "", "placeholder": "留空则读取全局配置"}),
                "ref_image_1": ("IMAGE",), "ref_image_2": ("IMAGE",), "ref_image_3": ("IMAGE",),
            }
        }
    RETURN_TYPES = ("STRING",); RETURN_NAMES = ("response",)
    FUNCTION = "submit"
    @classmethod
    def IS_CHANGED(cls, **kwargs): return time.time_ns()

    def submit(self, prompt, model, aspect_ratio, api_key="", ref_image_1=None, ref_image_2=None, ref_image_3=None):
        final_api_key = builtins.get_api_key(api_key, "grsai")
        if not final_api_key: return ("API Key 不能为空。",)
        
        try:
            urls = []
            for img in [ref_image_1, ref_image_2, ref_image_3]:
                if img is not None:
                    url = upload_image_grsai(final_api_key, img[0])
                    if url: urls.append(url)

            payload = { 
                "model": model, 
                "prompt": f"{prompt.strip()}{secrets.choice(ZERO_WIDTH_CHARS)}", 
                "aspectRatio": aspect_ratio, 
                "webHook": "-1"
            }
            if urls: payload["urls"] = urls

            api = GrsaiVeoAPI(final_api_key)
            task_id = api.submit_veo_task(payload)
            
            veo_task_db.update_task(task_id, {
                "prompt": prompt, 
                "model": model, 
                "submitted_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 
                "status": "pending", 
                "type": "multi_ref" 
            })
                
            return (f"Veo多参任务(异步)提交成功!\nID: {task_id}\n请使用 '异步查询' 节点刷新。",)
        except Exception as e:
            return (f"提交失败: {e}",)


# ======================================================================
# 【异步节点 3】: 异步查询任务状态
# ======================================================================
class VeoQueryTasks_Async:
    CATEGORY = "Nkxx/Grsai/视频 (异步)"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {"optional": {"api_key": ("STRING", {"default": "", "placeholder": "留空则读取全局配置"})}}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("report",)
    FUNCTION = "query_tasks"
    @classmethod
    def IS_CHANGED(cls, **kwargs): return time.time_ns()

    def query_tasks(self, api_key=""):
        final_api_key = builtins.get_api_key(api_key, "grsai")
        if not final_api_key: return ("API Key 不能为空。",)
        
        with veo_task_db.lock:
            tasks = veo_task_db.read_tasks()
            
        api = GrsaiVeoAPI(final_api_key)
        updated = False
        
        for task_id, task_info in tasks.items():
            if task_info.get("status") in ["succeeded", "downloaded", "failed", "download_failed"]: 
                continue
                
            try:
                status_data = api.query_task_status(task_id)
                if status_data.get("code") == 0 and "data" in status_data:
                    data = status_data["data"]
                    new_status = data.get("status")
                    new_progress = data.get("progress", 0)
                    
                    if task_info.get("status") != new_status or task_info.get("progress", 0) != new_progress:
                        task_info['status'] = new_status
                        task_info['progress'] = new_progress
                        if new_status == "succeeded":
                            task_info['video_url'] = data.get("url") 
                            task_info['pid'] = data.get("pid", "")
                        elif new_status == "failed":
                            task_info['failure_reason'] = data.get("failure_reason", "未知")
                        
                        veo_task_db.update_task(task_id, task_info)
                        updated = True
            except Exception:
                pass
        
        final_tasks = veo_task_db.read_tasks()
        
        full_report_lines = ["--- Veo 异步任务队列总览 ---"]
        sorted_tasks = sorted(final_tasks.items(), key=lambda item: item[1].get('submitted_at', ''), reverse=True)
        
        for tid, tinfo in sorted_tasks:
            status = tinfo.get('status', 'N/A')
            progress = tinfo.get('progress', 0)
            reason = tinfo.get('failure_reason', '')
            
            if status == 'running': status_str = f"运行中 {progress}%"
            elif status == 'pending': status_str = "排队中"
            elif status == 'succeeded': status_str = "待下载"
            elif status == 'downloaded': status_str = "已下载"
            elif status == 'download_failed': status_str = "下载失败"
            elif status == 'failed': status_str = "违规" if reason in ['output_moderation', 'input_moderation'] else "失败"
            else: status_str = status
                
            model_name = tinfo.get('model', 'veo-unknown') 
            full_report_lines.append(f"[{status_str}] ({model_name}) - {tinfo.get('prompt', '')[:20]}...")
            
        full_report_lines.append(f"\n当前剩余积分: {api.get_credits_balance()}")
        return ("\n".join(full_report_lines),)


# ======================================================================
# 【异步节点 4】: 获取并下载完成的视频
# ======================================================================
class VeoGetNextVideo_Async:
    CATEGORY = "Nkxx/Grsai/视频 (异步)"
    
    @classmethod
    def INPUT_TYPES(cls): return {}
    RETURN_TYPES = (IO.VIDEO, "STRING")
    RETURN_NAMES = ("video", "report")
    FUNCTION = "get_video"
    @classmethod
    def IS_CHANGED(cls, **kwargs): return time.time_ns()

    def get_video(self):
        with veo_task_db.lock:
            tasks = veo_task_db.read_tasks()
            
        sorted_tasks = sorted(tasks.items(), key=lambda item: item[1].get('submitted_at', ''))
        task_id = next((tid for tid, t in sorted_tasks if t.get("status") == "succeeded"), None)

        if not task_id:
            return (VideoAdapter(""), "当前无待下载的已完成任务。")

        video_url = tasks[task_id].get("video_url")
        if not video_url:
            veo_task_db.update_task(task_id, {'status': 'failed'})
            return (VideoAdapter(""), f"错误: 状态成功但无URL。")

        try:
            output_dir = folder_paths.get_output_directory()
            filename = f"veo3_{task_id[:8]}_{uuid.uuid4().hex[:4]}.mp4"
            output_path = os.path.join(output_dir, filename)
            
            robust_download_video(video_url, output_path, max_retries=3, timeout=300)
            veo_task_db.update_task(task_id, {'status': 'downloaded', 'video_path': output_path})
            
            return (VideoAdapter(output_path), f"下载成功: 任务已保存")
            
        except Exception as e:
            veo_task_db.update_task(task_id, {'status': 'download_failed'})
            traceback.print_exc()
            return (VideoAdapter(""), f"❌ 自动下载失败。\n请手动下载: {video_url}")


# --- 节点注册 ---
NODE_CLASS_MAPPINGS = {
    # 同步节点
    "Veo3_1_RefGenerator_Sync": Veo3_1_RefGenerator_Sync,
    "Veo3_1_FramesGenerator_Sync": Veo3_1_FramesGenerator_Sync,
    # 异步节点
    "VeoFramesSubmitTask_Async": VeoFramesSubmitTask_Async,
    "VeoMultiRefSubmitTask_Async": VeoMultiRefSubmitTask_Async,
    "VeoQueryTasks_Async": VeoQueryTasks_Async,
    "VeoGetNextVideo_Async": VeoGetNextVideo_Async,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Veo3_1_RefGenerator_Sync": "🎬 Veo3.1 参考图生成同步 (Grsai)",
    "Veo3_1_FramesGenerator_Sync": "🎬 Veo3.1 首尾帧生成同步 (Grsai)",
    "VeoFramesSubmitTask_Async": "1. 🎬 Veo 首尾帧异步提交 (Grsai)",
    "VeoMultiRefSubmitTask_Async": "1.5 🎬 Veo 多参异步提交 (Grsai)", 
    "VeoQueryTasks_Async": "2. 🎬 Veo 异步查询状态 (Grsai)",
    "VeoGetNextVideo_Async": "3. 🎬 Veo 异步获取视频 (Grsai)",
}