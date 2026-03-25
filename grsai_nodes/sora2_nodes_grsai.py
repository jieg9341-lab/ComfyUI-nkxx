import os
import requests
import time
import uuid
import folder_paths
import traceback
import secrets
import json
import tempfile
from datetime import datetime
import pandas as pd
import concurrent.futures
import builtins
import comfy.utils
from comfy.comfy_types import IO

# ======================================================================
# 导入公共工具 (兼容根目录或子目录结构)
# ======================================================================
try:
    from ..utils import (
        upload_image_grsai, 
        robust_download_video, 
        VideoAdapter, 
        TaskManager, 
        ZERO_WIDTH_CHARS
    )
except ImportError:
    from utils import (
        upload_image_grsai, 
        robust_download_video, 
        VideoAdapter, 
        TaskManager, 
        ZERO_WIDTH_CHARS
    )

# ======================================================================
# 全局配置 & 辅助函数
# ======================================================================
HOST = "https://grsai.dakka.com.cn"
# 实例化全局任务管理器 (将自动存放在 data/sora2_task_history.json)
TASK_MANAGER = TaskManager("sora2_task_history.json", max_completed_history=5)

def _get_headers(api_key: str) -> dict:
    return {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}

def _get_credits_balance(api_key: str) -> str:
    try:
        url = f"{HOST}/client/common/getCredits?apikey={api_key}"
        res = requests.get(url, timeout=10)
        if res.status_code == 200:
            data = res.json()
            if data.get("code") == 0 and "data" in data and "credits" in data["data"]:
                return str(int(data["data"]["credits"]))
    except Exception as e:
        print(f"[Sora2] 查询积分失败: {e}")
    return "查询失败"

def _upload_local_video_file(api_key: str, file_path: str) -> str:
    """专为上传本地视频角色使用的本地文件上传功能"""
    if not os.path.exists(file_path):
        raise Exception(f"文件不存在: {file_path}")
    suffix = os.path.splitext(file_path)[1][1:].lower()
    if not suffix: suffix = "mp4"

    try:
        headers = _get_headers(api_key)
        print(f"[Sora2 Upload] 获取视频上传Token...")
        token_res = requests.post(f"{HOST}/client/resource/newUploadTokenZH", headers=headers, json={"sux": suffix}, timeout=30)
        token_res.raise_for_status()
        token_data = token_res.json().get("data")
        
        if not token_data: raise Exception("Token数据为空")
        token, key, up_url, domain = (token_data["token"], token_data["key"], token_data["url"], token_data["domain"])
        
        print(f"[Sora2 Upload] 上传视频文件中: {file_path}")
        with open(file_path, "rb") as f:
            requests.post(url=up_url, data={"token": token, "key": key}, files={"file": f}, timeout=300).raise_for_status()
            
        return f"{domain}/{key}"
    except Exception as e:
        print(f"[Sora2 Upload] 视频上传错误: {e}")
        raise e

def _save_character_record(char_id, remark_text, source_info):
    """保存角色信息到本地TXT (自动存入 data 目录)"""
    try:
        # 获取当前脚本所在目录
        current_dir = os.path.dirname(os.path.realpath(__file__))
        
        # 兼容判断：如果当前目录下就有 data 文件夹，则直接使用；如果没有，则去上一级目录找
        if os.path.exists(os.path.join(current_dir, "data")):
            data_dir = os.path.join(current_dir, "data")
        else:
            data_dir = os.path.join(os.path.dirname(current_dir), "data")
            
        # 确保 data 目录存在 (防止意外被删)
        os.makedirs(data_dir, exist_ok=True)
        
        # 拼接最终的文件路径
        file_path = os.path.join(data_dir, "character_library.txt")
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        remark_str = remark_text.strip() if remark_text else "无"
        
        new_record = (f"[{timestamp}]\n备注: {remark_str}\nID:   {char_id}\n来源: {source_info}\n----------------------------------------\n")
        
        old_content = ""
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f: old_content = f.read()
        with open(file_path, "w", encoding="utf-8") as f: f.write(new_record + old_content)
        return True
    except Exception as e:
        print(f"[Sora2 Record] 保存角色记录失败: {e}")
        return False

def _poll_task_result(api_key: str, task_id: str, target_field: str = "url", timeout: int = 900):
    """同步节点的轮询方法"""
    print(f"[Sora2 Poll] 轮询任务 {task_id}...")
    start_time = time.time()
    last_progress = 0
    pbar = comfy.utils.ProgressBar(100)
    
    while time.time() - start_time < timeout:
        time.sleep(3)
        try:
            res = requests.post(f"{HOST}/v1/draw/result", headers=_get_headers(api_key), json={"id": task_id}, timeout=30)
            if res.status_code != 200: continue
            
            status_data = res.json()
            if status_data.get("code") == -22: raise Exception(f"任务过期: {status_data.get('msg')}")
            
            task_info = status_data.get("data", {})
            status = task_info.get("status")
            progress = task_info.get("progress", 0)

            if progress and progress > last_progress:
                pbar.update_absolute(int(progress))
                last_progress = progress
            
            if status == "succeeded":
                pbar.update_absolute(100)
                results_list = task_info.get("results", [])
                if not results_list: raise Exception("未返回结果数据")
                
                first_result = results_list[0]
                return (first_result.get("character_id"), task_info) if target_field == "character_id" else (first_result.get("url"), task_info)

            elif status == "failed":
                raise Exception(f"任务失败: {task_info.get('failure_reason')}")
        except Exception as e:
            if "任务失败" in str(e): raise e
            pass
            
    raise Exception("轮询超时")


# ======================================================================
# 同步节点 (直出视频与创建角色)
# ======================================================================
class Sora2Generator_Grsai:
    CATEGORY = "Nkxx/Grsai/视频 (同步)"
    
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
                "api_key": ("STRING", {"default": "", "multiline": False, "placeholder": "留空则使用全局配置"}),
            }
        }

    RETURN_TYPES = (IO.VIDEO, "STRING", "STRING")
    RETURN_NAMES = ("video", "video_url", "response")
    FUNCTION = "execute"

    @classmethod
    def IS_CHANGED(cls, **kwargs): return time.time_ns()

    def execute(self, prompt, aspect_ratio, duration, size, api_key="", image=None, remixTargetId=None):
        final_api_key = builtins.get_api_key(api_key, "grsai")
        if not final_api_key:
            return (VideoAdapter(""), "", "状态: 失败\n错误: API Key 不能为空。")

        try:
            final_prompt = f"{prompt.strip()}{secrets.choice(ZERO_WIDTH_CHARS)}"
            payload = {
                "model": "sora-2", "prompt": final_prompt, 
                "aspectRatio": aspect_ratio, "duration": int(duration), 
                "size": size, "webHook": "-1", "removeWatermark": True
            }
            if image is not None:
                print("[Sora2] 上传参考图...")
                payload["url"] = upload_image_grsai(final_api_key, image)

            if remixTargetId and remixTargetId.strip() and remixTargetId.strip() != "默认无":
                payload["remixTargetId"] = remixTargetId.strip()

            print("[Sora2] 提交任务...")
            res = requests.post(f"{HOST}/v1/video/sora-video", headers=_get_headers(final_api_key), json=payload, timeout=60)
            res.raise_for_status()
            data = res.json()
            if data.get("code") != 0: raise Exception(f"API提交失败: {data.get('msg')}")
            
            task_id = data.get("data", {}).get("id")
            video_url, task_info = _poll_task_result(final_api_key, task_id, target_field="url")
            pid = task_info.get("results", [{}])[0].get("pid", "")

            output_dir, filename = folder_paths.get_output_directory(), f"sora2_{uuid.uuid4().hex[:8]}.mp4"
            output_path = os.path.join(output_dir, filename)
            os.makedirs(output_dir, exist_ok=True)
            
            try:
                robust_download_video(video_url, output_path, max_retries=3, timeout=300)
                return (VideoAdapter(output_path), video_url, f"状态: success\n任务ID: {task_id}\nPID: {pid}\n积分: {_get_credits_balance(final_api_key)}")
            except Exception:
                 return (VideoAdapter(""), video_url, f"状态: 生成成功但下载失败\nURL: {video_url}\nPID: {pid}")

        except Exception as e:
            traceback.print_exc()
            return (VideoAdapter(""), "", f"状态: 失败\n错误: {e}")

class Sora2UploadCharacter_Grsai:
    CATEGORY = "Nkxx/Grsai/视频 (同步)"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_path": ("STRING", {"default": "", "multiline": False, "placeholder": "请填入本地视频的绝对路径"}),
                "start_time": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "end_time": ("FLOAT", {"default": 3.0, "min": 0.1, "max": 100.0, "step": 0.1}),
            },
            "optional": {
                "api_key": ("STRING", {"default": "", "multiline": False, "placeholder": "API Key"}),
                "备注": ("STRING", {"default": "", "multiline": False, "placeholder": "角色备注"}),
            }
        }
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("character_id", "status")
    FUNCTION = "execute"
    OUTPUT_NODE = True

    def execute(self, video_path, start_time, end_time, api_key="", 备注=""):
        final_api_key = builtins.get_api_key(api_key, "grsai")
        if not final_api_key: return ("", "Error: API Key required")
        
        clean_path = video_path.strip().replace('"', '').replace("'", "")
        if not clean_path or not os.path.exists(clean_path): return ("", f"Error: 文件不存在: {clean_path}")

        try:
            print(f"[Sora2 Char] 准备上传: {clean_path}")
            video_url = _upload_local_video_file(final_api_key, clean_path)
            ts_str = f"{float(start_time):g},{float(end_time):g}"
            payload = {"url": video_url, "timestamps": ts_str, "webHook": "-1", "shutProgress": False}
            
            res = requests.post(f"{HOST}/v1/video/sora-upload-character", headers=_get_headers(final_api_key), json=payload, timeout=60)
            res.raise_for_status()
            res_json = res.json()
            if res_json.get("code") != 0: raise Exception(f"提交失败: {res_json.get('msg')}")
            
            char_id, _ = _poll_task_result(final_api_key, res_json.get("data", {}).get("id"), target_field="character_id")
            
            save_msg = ""
            if char_id:
                _save_character_record(char_id, 备注, f"File: {os.path.basename(clean_path)}")
                save_msg = "\n(已保存到 data/character_library.txt)"

            return (char_id, f"Success\nID: {char_id}\n备注: {备注}{save_msg}")
        except Exception as e:
            traceback.print_exc()
            return ("", f"Error: {str(e)}")

class Sora2FromPidCharacter_Grsai:
    CATEGORY = "Nkxx/Grsai/视频 (同步)"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pid": ("STRING", {"default": "", "multiline": False, "placeholder": "原视频PID (s_xxxx...)"}),
                "start_time": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "end_time": ("FLOAT", {"default": 3.0, "min": 0.1, "max": 100.0, "step": 0.1}),
            },
            "optional": {
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "备注": ("STRING", {"default": "", "multiline": False}),
            }
        }
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("character_id", "status")
    FUNCTION = "execute"
    OUTPUT_NODE = True

    def execute(self, pid, start_time, end_time, api_key="", 备注=""):
        final_api_key = builtins.get_api_key(api_key, "grsai")
        if not final_api_key or not pid.strip(): return ("", "Error: Missing API Key or PID")

        try:
            ts_str = f"{float(start_time):g},{float(end_time):g}"
            payload = {"pid": pid.strip(), "timestamps": ts_str, "webHook": "-1", "shutProgress": False}
            
            res = requests.post(f"{HOST}/v1/video/sora-create-character", headers=_get_headers(final_api_key), json=payload, timeout=60)
            res.raise_for_status()
            res_json = res.json()
            if res_json.get("code") != 0: raise Exception(f"提交失败: {res_json.get('msg')}")
            
            char_id, _ = _poll_task_result(final_api_key, res_json.get("data", {}).get("id"), target_field="character_id")
            
            save_msg = ""
            if char_id:
                _save_character_record(char_id, 备注, f"PID: {pid}")
                save_msg = "\n(已保存到 data/character_library.txt)"

            return (char_id, f"Success\nID: {char_id}\n备注: {备注}{save_msg}")
        except Exception as e:
            traceback.print_exc()
            return ("", f"Error: {str(e)}")


# ======================================================================
# 异步节点 (队列与批量管理)
# ======================================================================
class Sora2SubmitAndRecordTask_Grsai:
    CATEGORY = "Nkxx/Grsai/视频 (异步)"
    
    @classmethod
    def INPUT_TYPES(cls):
        return { "required": {
                "prompt": ("STRING", {"default": "A cute cat playing on the grass", "multiline": True}),
                "aspect_ratio": (["16:9", "9:16"], {"default": "16:9"}),
                "duration": (["10", "15"], {"default": "10"}),
                "size": (["small", "large"], {"default": "small"}), 
            }, "optional": {
                "image": ("IMAGE",),
                "remixTargetId": ("STRING", {"default": "默认无", "multiline": False, "placeholder": "视频续作目标id"}),
                "api_key": ("STRING", {"default": "", "multiline": False}),
            }
        }
    RETURN_TYPES = ("STRING",); RETURN_NAMES = ("response",)
    FUNCTION = "submit"
    
    @classmethod
    def IS_CHANGED(cls, **kwargs): return time.time_ns()

    def submit(self, prompt, aspect_ratio, duration, size, api_key="", image=None, remixTargetId=None):
        final_api_key = builtins.get_api_key(api_key, "grsai")
        if not final_api_key: return ("API Key 不能为空。",)
        
        try:
            final_prompt = f"{prompt.strip()}{secrets.choice(ZERO_WIDTH_CHARS)}"
            payload = {"model": "sora-2", "prompt": final_prompt, "aspectRatio": aspect_ratio,
                       "duration": int(duration), "size": size, "webHook": "-1", "removeWatermark": True}
            
            if image is not None: payload["url"] = upload_image_grsai(final_api_key, image)
            if remixTargetId and remixTargetId.strip() != "默认无": payload["remixTargetId"] = remixTargetId.strip()

            res = requests.post(f"{HOST}/v1/video/sora-video", headers=_get_headers(final_api_key), json=payload, timeout=60)
            res.raise_for_status(); data = res.json()
            if data.get("code") != 0: raise Exception(f"API提交失败: {data.get('msg')}")
            
            task_id = data.get("data", {}).get("id")
            if not task_id: raise Exception("未返回有效任务ID")
            
            with TASK_MANAGER.lock:
                tasks = TASK_MANAGER.read_tasks()
                tasks[task_id] = { "prompt": prompt, "submitted_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "status": "pending" }
                TASK_MANAGER.write_tasks(tasks)
                
            return (f"任务提交成功!\nID: {task_id}\n请使用 '查询任务状态' 节点刷新。",)
        except Exception as e:
            return (f"提交失败: {e}",)

class Sora2SubmitBatchTask_Grsai:
    CATEGORY = "Nkxx/Grsai/视频 (异步)"
    
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
                "remixTargetId": ("STRING", {"default": "默认无"}),
                "api_key": ("STRING", {"default": ""}),
            }
        }
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("task_ids", "report")
    FUNCTION = "submit_batch"
    
    @classmethod
    def IS_CHANGED(cls, **kwargs): return time.time_ns()

    def submit_batch(self, file_path, column_name, aspect_ratio, duration, size, concurrency, 
                     api_key="", prompt_prefix="", max_count=50, image=None, remixTargetId=None):
        
        final_api_key = builtins.get_api_key(api_key, "grsai")
        if not final_api_key: return ("", "API Key 不能为空。")
        if not file_path or not os.path.exists(file_path): return ("", "文件不存在。")
        
        try:
            df = pd.read_csv(file_path, encoding='utf-8') if file_path.lower().endswith('.csv') else pd.read_excel(file_path)
            if column_name not in df.columns: return ("", f"列 '{column_name}' 不存在。")
            prompts = [f"{prompt_prefix}{p}" for p in df[column_name].dropna().astype(str).tolist()[:max_count]]
            if not prompts: return ("", f"列 '{column_name}' 中未找到有效 prompt。")
        except Exception as e: return ("", f"读取文件失败: {e}")

        uploaded_img_url = upload_image_grsai(final_api_key, image) if image is not None else None
        final_remix_id = remixTargetId.strip() if (remixTargetId and remixTargetId.strip() != "默认无") else None

        def _submit_single(prompt):
            try:
                payload = {"model": "sora-2", "prompt": f"{prompt.strip()}{secrets.choice(ZERO_WIDTH_CHARS)}", 
                           "aspectRatio": aspect_ratio, "duration": int(duration), "size": size, 
                           "webHook": "-1", "removeWatermark": True}
                if uploaded_img_url: payload["url"] = uploaded_img_url
                if final_remix_id: payload["remixTargetId"] = final_remix_id
                
                res = requests.post(f"{HOST}/v1/video/sora-video", headers=_get_headers(final_api_key), json=payload, timeout=60)
                res.raise_for_status()
                data = res.json()
                if data.get("code") != 0: return (prompt, f"API失败: {data.get('msg')}")
                
                task_id = data.get("data", {}).get("id")
                with TASK_MANAGER.lock:
                    tasks = TASK_MANAGER.read_tasks()
                    tasks[task_id] = {"prompt": prompt, "submitted_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "status": "pending"}
                    TASK_MANAGER.write_tasks(tasks)
                return (prompt, task_id)
            except Exception as e: return (prompt, f"异常: {e}")

        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
            results = list(executor.map(_submit_single, prompts))

        success, fail, lines = 0, 0, []
        for p, r in results:
            if "失败" in r or "异常" in r: fail += 1
            else: success += 1
            lines.append(r) 
        
        return ("\n".join(lines), f"批量完成 | 总数: {len(prompts)} | 成功: {success} | 失败: {fail}")

class Sora2QueryTasks_Grsai:
    CATEGORY = "Nkxx/Grsai/视频 (异步)"
    
    @classmethod
    def INPUT_TYPES(cls): return {"optional": {"api_key": ("STRING", {"default": ""})}}
    RETURN_TYPES = ("STRING",); RETURN_NAMES = ("report",); FUNCTION = "query_tasks"
    
    @classmethod
    def IS_CHANGED(cls, **kwargs): return time.time_ns()

    def query_tasks(self, api_key=""):
        final_api_key = builtins.get_api_key(api_key, "grsai")
        if not final_api_key: return ("API Key 不能为空。",)
        
        with TASK_MANAGER.lock:
            tasks = TASK_MANAGER.read_tasks()
            updated = False
            for tid, tinfo in tasks.items():
                if tinfo.get("status") in ["downloaded", "failed", "download_failed", "succeeded"]: continue
                try:
                    res = requests.post(f"{HOST}/v1/draw/result", headers=_get_headers(final_api_key), json={"id": tid}, timeout=30)
                    if res.json().get("code") == 0 and "data" in res.json():
                        data = res.json()["data"]
                        new_status, new_prog = data.get("status"), data.get("progress", 0)
                        if tinfo.get("status") != new_status or (new_status == 'running' and tinfo.get("progress") != new_prog):
                            tinfo['status'], tinfo['progress'] = new_status, new_prog
                            if new_status == "succeeded":
                                tinfo['video_url'] = data.get("results", [{}])[0].get("url")
                                tinfo['pid'] = data.get("results", [{}])[0].get("pid", "") 
                            elif new_status == "failed":
                                tinfo['failure_reason'] = data.get("failure_reason", "未知")
                            updated = True
                except: pass
            
            if updated: TASK_MANAGER.write_tasks(tasks)
            # 重新获取以便格式化输出
            tasks_for_report = TASK_MANAGER.read_tasks()
            
        lines = ["--- 任务队列总览 ---"]
        sorted_tasks = sorted(tasks_for_report.items(), key=lambda x: x[1].get('submitted_at', ''), reverse=True)
        for tid, tinfo in sorted_tasks:
            st = tinfo.get('status', 'N/A')
            st_str = f"running {tinfo.get('progress', 0)}%" if st == 'running' else st
            pid_str = f" (PID: {tinfo.get('pid')})" if st in ["succeeded", "downloaded", "download_failed"] and tinfo.get("pid") else ""
            lines.append(f"[{st_str}] {tid[:8]}...{pid_str} - {tinfo.get('prompt', '')[:25]}...")
            
        lines.append(f"\n当前剩余积分: {_get_credits_balance(final_api_key)}")
        return ("\n".join(lines),)

class Sora2GetNextVideo_Grsai:
    CATEGORY = "Nkxx/Grsai/视频 (异步)"
    
    @classmethod
    def INPUT_TYPES(cls): return {}
    RETURN_TYPES = (IO.VIDEO, "STRING"); RETURN_NAMES = ("video", "report"); FUNCTION = "get_video"
    
    @classmethod
    def IS_CHANGED(cls, **kwargs): return time.time_ns()

    def get_video(self):
        with TASK_MANAGER.lock: tasks = TASK_MANAGER.read_tasks()
        
        sorted_tasks = sorted(tasks.items(), key=lambda x: x[1].get('submitted_at', ''))
        target_id = next((tid for tid, t in sorted_tasks if t.get("status") == "succeeded"), None)

        if not target_id: return (VideoAdapter(""), "当前无新完成的任务可供下载。")

        task_info = tasks[target_id]
        if not task_info.get("video_url"):
            TASK_MANAGER.update_task(target_id, {'status': 'failed'})
            return (VideoAdapter(""), f"错误: 状态成功但无URL。")

        try:
            out_dir = folder_paths.get_output_directory()
            out_path = os.path.join(out_dir, f"sora2_{target_id[:8]}_{uuid.uuid4().hex[:4]}.mp4")
            
            robust_download_video(task_info["video_url"], out_path, max_retries=3, timeout=300)
            
            TASK_MANAGER.update_task(target_id, {'status': 'downloaded', 'video_path': out_path})
            pid_info = f" (PID: {task_info.get('pid')})" if task_info.get('pid') else ""
            
            return (VideoAdapter(out_path), f"下载成功: {target_id[:8]}...{pid_info}")
            
        except Exception as e:
            TASK_MANAGER.update_task(target_id, {'status': 'download_failed'})
            print(f"[Sora2 Downloader] 失败: {e}")
            return (VideoAdapter(""), f"❌ 自动下载失败: {target_id[:8]}...\n请复制链接手动下载:\n{task_info['video_url']}")


# ======================================================================
# 节点注册
# ======================================================================
NODE_CLASS_MAPPINGS = {
    "Sora2Generator_Grsai": Sora2Generator_Grsai,
    "Sora2UploadCharacter_Grsai": Sora2UploadCharacter_Grsai,
    "Sora2FromPidCharacter_Grsai": Sora2FromPidCharacter_Grsai,
    "Sora2SubmitAndRecordTask_Grsai": Sora2SubmitAndRecordTask_Grsai,
    "Sora2SubmitBatchTask_Grsai": Sora2SubmitBatchTask_Grsai, 
    "Sora2QueryTasks_Grsai": Sora2QueryTasks_Grsai,
    "Sora2GetNextVideo_Grsai": Sora2GetNextVideo_Grsai,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Sora2Generator_Grsai": "Sora2 视频生成同步 (Grsai)",
    "Sora2UploadCharacter_Grsai": "Sora2 上传角色同步 (Grsai)",
    "Sora2FromPidCharacter_Grsai": "Sora2 原视频角色同步 (Grsai)",
    "Sora2SubmitAndRecordTask_Grsai": "1. Sora2 异步提交任务 (Grsai)",
    "Sora2SubmitBatchTask_Grsai": "1.5 Sora2 异步批量提交 (Grsai)", 
    "Sora2QueryTasks_Grsai": "2. Sora2 异步查询状态 (Grsai)",
    "Sora2GetNextVideo_Grsai": "3. Sora2 异步获取视频 (Grsai)",
}