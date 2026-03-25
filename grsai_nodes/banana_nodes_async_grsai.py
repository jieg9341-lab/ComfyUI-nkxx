# grsai_nodes/banana_nodes_async_grsai.py
import os
import json
import time
import secrets
import traceback
import concurrent.futures
import pandas as pd
from datetime import datetime
from typing import Dict, Any, Optional, List
import requests
import torch

import builtins
import folder_paths

# 从核心工具箱导入公共函数
from ..utils import (
    TaskManager,
    download_image,
    safe_pil_batch_to_tensor,
    upload_image_grsai
)

# --- 核心配置与工具 ---
SUPPORTED_ASPECT_RATIOS = ["auto", "1:1", "16:9", "9:16", "4:3", "3:4", "3:2", "2:3", "5:4", "4:5", "21:9"]
# 增加了 nano-banana-2-cl
SUPPORTED_MODELS_ASYNC = ["nano-banana-fast", "nano-banana-pro", "nano-banana-pro-vt", "nano-banana-pro-cl", "nano-banana-2", "nano-banana-2-cl"]
ZERO_WIDTH_CHARS = ["\u200b", "\u200c", "\u200d", "\ufeff", "\u180e", "\u200e", "\u200f"]

# 初始化专属的异步任务管理器
banana_task_db = TaskManager("grsai_banana_async_history.json", max_completed_history=5)

def format_error_message(error: Exception) -> str:
    return f"{type(error).__name__}: {str(error)}"

def _get_next_task_number(tasks: dict, is_batch=False) -> int:
    """根据现有任务获取下一个任务序号"""
    prefix = "批量任务" if is_batch else "任务"
    nums = []
    for k in tasks.keys():
        if k.startswith(prefix):
            if not is_batch and k.startswith("批量任务"): 
                continue
            try: 
                nums.append(int(k[len(prefix):]))
            except ValueError: 
                pass
    return max(nums, default=0) + 1

# --- API 客户端 ---
class GrsaiAPIError(Exception): pass

class GrsaiAsyncAPI:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://grsai.dakka.com.cn"
        self.headers = {
            "Content-Type": "application/json; charset=utf-8", 
            "User-Agent": "ComfyUI-Nkxx/2.0", 
            "Authorization": f"Bearer {self.api_key}"
        }

    def make_request(self, endpoint: str, payload: dict) -> dict:
        try:
            res = requests.post(f"{self.base_url}{endpoint}", headers=self.headers, json=payload, timeout=60)
            res.raise_for_status()
            data = res.json()
            if data.get("code") != 0:
                raise GrsaiAPIError(f"API 错误: {data.get('msg', '未知')}")
            return data
        except Exception as e:
            raise GrsaiAPIError(str(e))

# --- 节点基类 ---
class _GrsaiAsyncNodeBase:
    FUNCTION = "execute"
    @classmethod
    def IS_CHANGED(cls, **kwargs): return float("NaN")

    def _create_error_result(self, error_message: str, is_text_output: bool = False):
        print(f"节点执行错误: {error_message}")
        if is_text_output:
            return {"ui": {"string": [error_message]}, "result": (None, error_message)}
        image_out = torch.zeros((1, 1, 1, 3), dtype=torch.float32)
        return {"ui": {"string": [error_message]}, "result": (image_out, f"失败: {error_message}")}

    def _get_credits_balance(self, api_key: str) -> int:
        try:
            url = f"https://grsai.dakka.com.cn/client/common/getCredits?apikey={api_key}"
            res = requests.get(url, timeout=10)
            if res.status_code == 200:
                data = res.json()
                if data.get("code") == 0 and "data" in data and "credits" in data["data"]:
                    return int(data["data"]["credits"])
        except: pass
        return -1
        
    def _handle_image_uploads(self, images_in: list[torch.Tensor], api_key: str):
        uploaded_urls = []
        if not any(img is not None for img in images_in): 
            return uploaded_urls
        try:
            for img_tensor in images_in:
                if img_tensor is None: continue
                url = upload_image_grsai(api_key, img_tensor)
                if url: uploaded_urls.append(url)
            return uploaded_urls
        except Exception as e:
            raise Exception(f"图像上传失败: {format_error_message(e)}")

# ======================================================================
# 节点 1: NanoBanana 异步提交 (单任务)
# ======================================================================
class NanoBananaAsyncSubmit(_GrsaiAsyncNodeBase):
    CATEGORY = "Nkxx/Grsai/图像异步"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
                    "prompt": ("STRING", {"multiline": True, "default": "一只可爱的小猫"}),
                    "model": (SUPPORTED_MODELS_ASYNC, {"default": "nano-banana-fast"}),
                    "image_size": (["默认", "1K", "2K", "4K"], {"default": "默认"}),
                    "concurrency": ("INT", {"default": 1, "min": 1, "max": 10, "step": 1}),
                    "aspect_ratio": (SUPPORTED_ASPECT_RATIOS, {"default": "auto"}),
                }, "optional": {
                    "api_key": ("STRING", {"multiline": False, "default": "", "placeholder": "留空则读取全局配置"}),
                    "image_1": ("IMAGE",), "image_2": ("IMAGE",), "image_3": ("IMAGE",), "image_4": ("IMAGE",),
                }}
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    FUNCTION = "submit"
    
    def submit(self, prompt: str, model: str, image_size: str, concurrency: int, aspect_ratio: str, api_key: str = "", **kwargs):
        final_api_key = builtins.get_api_key(api_key, "grsai")
        if not final_api_key: return self._create_error_result("未配置 Grsai API Key。", is_text_output=True)
        
        images_in = [kwargs.get(f"image_{i}") for i in range(1, 5)]
        try:
            uploaded_urls = self._handle_image_uploads(images_in, final_api_key)
            
            with banana_task_db.lock:
                tasks = banana_task_db.read_tasks()
                
                # === 新增：安全熔断机制（链式间隔检测） ===
                # 1. 筛选出所有“未下载”的任务，并按提交时间倒序（最新的在最前）
                active_tasks = [t for t in tasks.values() if t.get("status") != "downloaded"]
                active_tasks.sort(key=lambda x: x.get("submitted_at", ""), reverse=True)

                # 2. 统计连续且时间间隔极短的相同提示词数量
                identical_count = 0
                last_time = datetime.now() # 初始化比较基准为当前时间
                
                for t in active_tasks:
                    if t.get("type") == "normal" and t.get("prompt") == prompt:
                        try:
                            # 解析历史任务的提交时间
                            t_time = datetime.strptime(t.get("submitted_at", ""), "%Y-%m-%d %H:%M:%S")
                            
                            # 核心：计算当前任务与上一个任务(或当前时间)的间隔
                            # 如果是 ComfyUI 自动连续排队，间隔通常只有几秒(API耗时)
                            if (last_time - t_time).total_seconds() <= 15: 
                                identical_count += 1
                                last_time = t_time # 将基准时间推移到当前遍历的任务
                            else:
                                # 如果间隔超过 15 秒，说明用户中断过，不属于瞬间爆破，打断计数
                                break
                        except Exception:
                            break # 时间解析异常也安全打断
                    else:
                        break 
                
                # 3. 触发熔断：连续 10 次且都在极短间隔内触发，则直接抛错拦截 (测试完毕后可改回 15)
                if identical_count >= 10:
                    error_msg = (
                        f"【安全熔断触发】检测到系统正在自动连续提交相同的提示词 (已拦截第 {identical_count + 1} 次)！\n"
                        f"为防止您的积分因误操作被异常消耗，当前提交已被强制阻断。\n"
                        f"💡 您的前 {identical_count} 次任务已成功发送，请放心。\n"
                        f"请检查并清理 ComfyUI 队列，或关闭自动排队 (Auto Queue)。"
                    )
                    raise ValueError(error_msg)
                # ==========================================
                
                task_num = _get_next_task_number(tasks, is_batch=False)
                task_id = f"任务{task_num}"
            
            api_client = GrsaiAsyncAPI(api_key=final_api_key)
            subtasks = []

            target_size = image_size
            if image_size == "默认":
                target_size = "2K" if model in ["nano-banana-pro", "nano-banana-pro-vt", "nano-banana-pro-cl", "nano-banana-2", "nano-banana-2-cl"] else "1K"
            
            # 分辨率动态映射逻辑
            actual_model = model
            if model == "nano-banana-2-cl" and target_size == "4K":
                actual_model = "nano-banana-2-4k-cl"
            
            for i in range(concurrency):
                payload = {
                    "model": actual_model,
                    "prompt": f"{prompt}{secrets.choice(ZERO_WIDTH_CHARS) * i}",
                    "aspectRatio": aspect_ratio,
                    "urls": uploaded_urls,
                    "webHook": "-1",
                    "shutProgress": True,
                    "imageSize": target_size
                }
                response = api_client.make_request("/v1/draw/nano-banana", payload)
                subtasks.append({
                    "api_task_id": response["data"]["id"],
                    "status": "running",
                    "image_url": None,
                    "progress": 0,
                    "failure_reason": None
                })
            
            banana_task_db.update_task(task_id, {
                "type": "normal",
                "prompt": prompt,
                "model": actual_model,
                "submitted_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "status": "running",
                "aspect_ratio": aspect_ratio,
                "concurrency": concurrency,
                "subtasks": subtasks
            })
            
            credits_left = self._get_credits_balance(final_api_key)
            status_msg = f"任务提交成功 | {task_id} | 模型: {actual_model} | 子任务数: {concurrency} | 积分: {credits_left if credits_left >= 0 else 'N/A'}"
            return {"ui": {"string": [status_msg]}, "result": (status_msg,)}
        except Exception as e:
            # 如果是我们的熔断抛出的 ValueError，直接往上抛，触发 ComfyUI 的红色弹窗打断机制
            if isinstance(e, ValueError) and "【安全熔断触发】" in str(e):
                raise e
            return self._create_error_result(f"提交失败: {format_error_message(e)}", is_text_output=True)

# ======================================================================
# 节点 2: NanoBanana 异步批量提交
# ======================================================================
class NanoBananaAsyncBatchSubmit(_GrsaiAsyncNodeBase):
    CATEGORY = "Nkxx/Grsai/图像异步"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
                "file_path": ("STRING", {"default": "", "placeholder": "拖拽 CSV/Excel 文件至此"}),
                "column_name": ("STRING", {"default": "prompt"}),
                "prompt_prefix": ("STRING", {"multiline": True, "default": ""}),
                "model": (SUPPORTED_MODELS_ASYNC, {"default": "nano-banana-fast"}),
                "image_size": (["默认", "1K", "2K", "4K"], {"default": "默认"}),
                "aspect_ratio": (SUPPORTED_ASPECT_RATIOS, {"default": "auto"}),
                "executions_per_prompt": ("INT", {"default": 1, "min": 1, "max": 10, "step": 1, "label": "单提示词执行次数"}),
            }, "optional": {
                "api_key": ("STRING", {"multiline": False, "default": "", "placeholder": "留空则读取全局配置"}),
                "image_1": ("IMAGE",), "image_2": ("IMAGE",), "image_3": ("IMAGE",), "image_4": ("IMAGE",),
            }}
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    FUNCTION = "submit_batch"
    
    def submit_batch(self, file_path: str, column_name: str, prompt_prefix: str, model: str, image_size: str, aspect_ratio: str, executions_per_prompt: int, api_key: str = "", **kwargs):
        final_api_key = builtins.get_api_key(api_key, "grsai")
        if not final_api_key: return self._create_error_result("未配置 Grsai API Key。", is_text_output=True)
        
        if not file_path or not os.path.exists(file_path):
            return self._create_error_result("文件路径为空或文件不存在。", is_text_output=True)
        
        try:
            if file_path.lower().endswith('.csv'): df = pd.read_csv(file_path, encoding='utf-8')
            elif file_path.lower().endswith(('.xls', '.xlsx')): df = pd.read_excel(file_path)
            else: return self._create_error_result("仅支持 .csv, .xls, .xlsx 文件。", is_text_output=True)
        except Exception as e: return self._create_error_result(f"读取文件失败: {format_error_message(e)}", is_text_output=True)
        
        if column_name not in df.columns: return self._create_error_result(f"列 '{column_name}' 不存在。", is_text_output=True)
        
        base_prompts = [f"{prompt_prefix}{p}" for p in df[column_name].dropna().astype(str).tolist()]
        if not base_prompts: return self._create_error_result(f"列 '{column_name}' 中未找到有效 prompt。", is_text_output=True)
        
        prompts = [p for p in base_prompts for _ in range(max(1, executions_per_prompt))]
        total_tasks = len(prompts)

        images_in = [kwargs.get(f"image_{i}") for i in range(1, 5)]
        try:
            uploaded_urls = self._handle_image_uploads(images_in, final_api_key)
        except Exception as e:
            return self._create_error_result(str(e), is_text_output=True)

        target_size = image_size
        if image_size == "默认":
            target_size = "2K" if model in ["nano-banana-pro", "nano-banana-pro-vt", "nano-banana-pro-cl", "nano-banana-2", "nano-banana-2-cl"] else "1K"
        
        actual_model = model
        if model == "nano-banana-2-cl" and target_size == "4K":
            actual_model = "nano-banana-2-4k-cl"
        
        api_client = GrsaiAsyncAPI(api_key=final_api_key)
        subtasks, errors = [], []

        def submit_single_req(p_idx, prompt_text):
            try:
                final_prompt = f"{prompt_text}{secrets.choice(ZERO_WIDTH_CHARS) * (p_idx % 10)}"
                payload = {
                    "model": actual_model,
                    "prompt": final_prompt,
                    "aspectRatio": aspect_ratio,
                    "urls": uploaded_urls,
                    "webHook": "-1",
                    "shutProgress": True,
                    "imageSize": target_size
                }
                response = api_client.make_request("/v1/draw/nano-banana", payload)
                return {
                    "api_task_id": response["data"]["id"],
                    "status": "running",
                    "image_url": None,
                    "progress": 0,
                    "failure_reason": None,
                    "original_prompt": prompt_text
                }
            except Exception as e:
                return {"error": str(e)}

        print(f"[Banana Async Batch] 开始并发提交 {total_tasks} 个任务...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            future_to_idx = {executor.submit(submit_single_req, i, p): i for i, p in enumerate(prompts)}
            for future in concurrent.futures.as_completed(future_to_idx):
                res = future.result()
                if "error" in res:
                    errors.append(res["error"])
                else:
                    subtasks.append(res)
        
        if not subtasks:
             return self._create_error_result(f"所有任务提交均失败。错误示例: {errors[0] if errors else '未知'}", is_text_output=True)

        with banana_task_db.lock:
            tasks = banana_task_db.read_tasks()
            task_num = _get_next_task_number(tasks, is_batch=True)
            task_id = f"批量任务{task_num}"
        
        banana_task_db.update_task(task_id, {
            "type": "batch",
            "prompt": f"批量文件: {os.path.basename(file_path)}",
            "model": actual_model,
            "submitted_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "status": "running",
            "aspect_ratio": aspect_ratio,
            "concurrency": total_tasks,
            "subtasks": subtasks
        })

        credits_left = self._get_credits_balance(final_api_key)
        status_msg = f"批量提交完成 | {task_id} | 成功提交: {len(subtasks)}/{total_tasks} | 积分: {credits_left if credits_left >= 0 else 'N/A'}"
        if errors: status_msg += f" | 提交失败数: {len(errors)}"
            
        return {"ui": {"string": [status_msg]}, "result": (status_msg,)}

# ======================================================================
# 节点 3: NanoBanana 异步查询下载
# ======================================================================
class NanoBananaAsyncQuery(_GrsaiAsyncNodeBase):
    CATEGORY = "Nkxx/Grsai/图像异步"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {"optional": {
                    "api_key": ("STRING", {"multiline": False, "default": "", "placeholder": "留空则读取全局配置"}),
                }}
    
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "status")
    FUNCTION = "query_and_download"

    def _generate_display_status(self, tasks: dict) -> str:
        all_tasks_status = []
        status_map = {"running": "运行中", "pending": "排队中", "succeeded": "待下载", "failed": "失败", "downloaded": "已下载"}

        for tid, tinfo in sorted(tasks.items(), key=lambda x: x[1].get('submitted_at', ''), reverse=True):
            raw_status = tinfo.get('status', 'running')
            task_type = tinfo.get('type', 'normal')
            concurrency = tinfo.get('concurrency', 0)
            
            subtasks = tinfo.get('subtasks', [])
            total_sub = len(subtasks)
            success_sub = sum(1 for s in subtasks if s.get('status') == 'succeeded')
            fail_sub = sum(1 for s in subtasks if s.get('status') == 'failed')
            done_sub = success_sub + fail_sub
            
            if raw_status == 'pending': raw_status = 'running'
            display_str = status_map.get(raw_status, raw_status).upper()
            
            if task_type == 'batch':
                info_text = f"进度: {done_sub}/{total_sub}"
                if done_sub > 0: info_text += f" (成功: {success_sub} | 失败: {fail_sub})"
                all_tasks_status.append(f"[{display_str}] {tid} - {info_text}")
            else:
                prompt_full = tinfo.get('prompt', '')
                prompt_snippet = prompt_full[:15] + "..." if len(prompt_full) > 15 else prompt_full
                all_tasks_status.append(f"[{display_str}] {tid} ({prompt_snippet}) - {concurrency}个子任务")
        
        return "\n".join(all_tasks_status) if all_tasks_status else "当前无任务记录"
    
    def query_and_download(self, api_key: str = ""):
        final_api_key = builtins.get_api_key(api_key, "grsai")
        if not final_api_key: return self._create_error_result("未配置 Grsai API Key。")
        
        with banana_task_db.lock:
            tasks = banana_task_db.read_tasks()
        
        if not tasks: return self._create_error_result("当前没有任务记录。", is_text_output=True)

        api_client = GrsaiAsyncAPI(api_key=final_api_key)
        active_tasks = [(tid, tinfo) for tid, tinfo in tasks.items() if tinfo.get("status") not in ["succeeded", "failed", "downloaded"]]
        tasks_updated = False
        
        if active_tasks:
            print(f"[Banana Query] 正在刷新 {len(active_tasks)} 个活跃任务的状态...")
            for tid, tinfo in active_tasks:
                subtasks = tinfo.get("subtasks", [])
                if not subtasks: continue
                
                any_sub_running = any_sub_succeeded = subtask_updated = False
                
                for subtask in subtasks:
                    current_status = subtask.get("status", "running")
                    if current_status in ["succeeded", "failed"]:
                        if current_status == "succeeded": any_sub_succeeded = True
                        continue 
                        
                    api_task_id = subtask.get("api_task_id")
                    if not api_task_id: 
                        current_status = "failed"
                    else:
                        try:
                            response = api_client.make_request("/v1/draw/result", payload={"id": api_task_id})
                            query_data = response.get("data", {})
                            if query_data:
                                current_status = query_data.get("status", "running")
                                if current_status == "pending": current_status = "running"
                                
                                if query_data.get("results"):
                                    subtask["image_url"] = query_data["results"][0].get("url")
                                if current_status == "failed":
                                    subtask["failure_reason"] = query_data.get("failure_reason", "未知错误")
                        except Exception:
                            current_status = "running"

                    if subtask.get("status") != current_status:
                        subtask["status"] = current_status
                        subtask_updated = True
                    
                    if current_status == "running": any_sub_running = True
                    elif current_status == "succeeded": any_sub_succeeded = True
                
                final_main_status = "running" if any_sub_running else ("succeeded" if any_sub_succeeded else "failed")
                
                if final_main_status != tinfo.get("status") or subtask_updated:
                    tinfo["status"] = final_main_status
                    tinfo["subtasks"] = subtasks
                    banana_task_db.update_task(tid, tinfo)
                    tasks_updated = True

        if tasks_updated:
            tasks = banana_task_db.read_tasks()

        # 开始处理待下载的任务
        succeeded_candidates = [(tid, tinfo) for tid, tinfo in tasks.items() if tinfo.get("status") == "succeeded"]
        status_display_text = self._generate_display_status(tasks)
        
        if not succeeded_candidates:
            return {"ui": {"string": [status_display_text]}, "result": (torch.zeros((1, 1, 1, 3), dtype=torch.float32), status_display_text)}
        
        succeeded_candidates.sort(key=lambda x: x[1].get("submitted_at", "2999-01-01"))
        target_tid, target_tinfo = succeeded_candidates[0]
        
        try:
            pil_images = []
            subtasks = target_tinfo.get("subtasks", [])
            
            # 并发下载图片
            download_futures = {}
            with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                for sub in subtasks:
                    if sub.get("status") == "succeeded" and sub.get("image_url"):
                        future = executor.submit(download_image, sub.get("image_url"))
                        download_futures[future] = sub
                
                for future in concurrent.futures.as_completed(download_futures):
                    img = future.result()
                    if img: pil_images.append(img)
            
            if not pil_images:
                 return self._create_error_result(f"{target_tid} 图片下载失败 (部分成功任务)。", is_text_output=True)
            
            banana_task_db.update_task(target_tid, {"status": "downloaded"})
            tasks = banana_task_db.read_tasks()
            status_display_text = self._generate_display_status(tasks)

            total_subtasks = len(subtasks)
            success_count = len(pil_images)
            fail_count = total_subtasks - success_count
            credits_left = self._get_credits_balance(final_api_key)
            credits_str = str(credits_left) if credits_left >= 0 else 'N/A'

            result_title = "部分成功" if fail_count > 0 else "下载成功"
            count_info = f"成功: {success_count} | 失败: {fail_count}" if fail_count > 0 else f"共 {success_count} 张"
            final_msg = f"{result_title}: {target_tid} | {count_info} | 积分: {credits_str}\n{status_display_text}"
            
            return {"ui": {"string": [final_msg]}, "result": (safe_pil_batch_to_tensor(pil_images), final_msg)}
            
        except Exception as e:
            return self._create_error_result(f"下载过程出错: {e}", is_text_output=True)

# --- 节点注册 ---
NODE_CLASS_MAPPINGS = {
    "NanoBananaAsyncSubmit": NanoBananaAsyncSubmit,
    "NanoBananaAsyncBatchSubmit": NanoBananaAsyncBatchSubmit,
    "NanoBananaAsyncQuery": NanoBananaAsyncQuery,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NanoBananaAsyncSubmit": "🍌 Nano Banana 异步提交 (Grsai)",
    "NanoBananaAsyncBatchSubmit": "🍌 Nano Banana 异步批量提交 (Grsai)",
    "NanoBananaAsyncQuery": "🍌 Nano Banana 异步查询与下载 (Grsai)",
}