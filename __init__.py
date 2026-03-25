# ComfyUI-nkxx/__init__.py
import os
import importlib
import traceback
import builtins
import sys
import subprocess
import importlib.util

# ======================================================================
# 1. 自动依赖安装逻辑
# ======================================================================
def check_and_install_dependencies():
    required_packages = {
        "requests": "requests",
        "pandas": "pandas",
        "openpyxl": "openpyxl", 
        "yt-dlp": "yt_dlp",
        "opencv-python": "cv2",
        "aiohttp": "aiohttp"
    }

    print("--- [ComfyUI-nkxx] 正在检查核心依赖... ---")
    for package_name, import_name in required_packages.items():
        spec = importlib.util.find_spec(import_name)
        if spec is None:
            print(f"  > 检测到缺失库: {package_name}，正在自动安装...")
            try:
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', package_name])
            except subprocess.CalledProcessError:
                print(f"  > [警告] {package_name} 安装失败，请手动安装。")

check_and_install_dependencies()

# ======================================================================
# 2. 全局通用配置 & API Key 注入
# ======================================================================
# 可以在这里硬编码你的默认 Key，如果不填则优先读取节点内输入或系统环境变量
GRSAI_DEFAULT_API_KEY = ""
WUJIAI_DEFAULT_API_KEY = ""

def get_api_key(api_key_from_node: str, channel: str = "grsai") -> str:
    """
    智能获取 API Key。
    优先级: 1. 节点直填 -> 2. 系统环境变量 -> 3. __init__.py 默认配置
    channel: "grsai" 或 "wujiai"
    """
    if api_key_from_node and api_key_from_node.strip():
        return api_key_from_node.strip()
    
    if channel.lower() == "grsai":
        return os.getenv("GRSAI_KEY", GRSAI_DEFAULT_API_KEY).strip()
    elif channel.lower() == "wujiai":
        return os.getenv("WUJIAI_KEY", WUJIAI_DEFAULT_API_KEY).strip()
    
    return ""

# 注入到 builtins 以供子模块无缝调用
builtins.get_api_key = get_api_key

# ======================================================================
# 3. 智能节点加载逻辑 (支持根目录与子文件夹)
# ======================================================================
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

current_dir = os.path.dirname(__file__)

# 需要扫描的目录：根目录("."), grsai文件夹, wujiai文件夹
scan_dirs = [".", "grsai_nodes", "wujiai_nodes"]

for subdir in scan_dirs:
    target_dir = os.path.join(current_dir, subdir) if subdir != "." else current_dir
    if not os.path.exists(target_dir):
        continue
        
    for filename in os.listdir(target_dir):
        if filename.endswith(".py") and filename not in ["__init__.py", "utils.py"]:
            module_name = filename[:-3]
            # 构造相对导入路径，例如 ".tools" 或 ".grsai.banana_nodes"
            import_path = f".{module_name}" if subdir == "." else f".{subdir}.{module_name}"
            
            try:
                module = importlib.import_module(import_path, package=__name__)
                if hasattr(module, "NODE_CLASS_MAPPINGS"):
                    NODE_CLASS_MAPPINGS.update(module.NODE_CLASS_MAPPINGS)
                if hasattr(module, "NODE_DISPLAY_NAME_MAPPINGS"):
                    NODE_DISPLAY_NAME_MAPPINGS.update(module.NODE_DISPLAY_NAME_MAPPINGS)
                print(f"  > [ComfyUI-nkxx] 已加载: {import_path}")
            except Exception as e:
                print(f"  > [ComfyUI-nkxx] 加载失败 {import_path}:")
                traceback.print_exc()

# ======================================================================
# 4. 暴露给 ComfyUI 的核心接口
# ======================================================================
WEB_DIRECTORY = "./web"

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'WEB_DIRECTORY']