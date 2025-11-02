# -- ComfyUI-Nkxx/__init__.py --
import os
import importlib
import traceback
import builtins

# --- 全局通用配置 ---

# 在这里填入您的默认API Key。当节点中的api_key框为空时，会自动使用此Key。
# 例如: YOUR_DEFAULT_API_KEY = "sk-xxxxxxxxxxxxxxxxxxxxxxxx"
YOUR_DEFAULT_API_KEY = ""

# --- 全局通用函数 ---

def get_api_key(api_key_from_node: str) -> str:
    """
    根据优先级获取最终使用的API Key。
    这个函数将被注入到 builtins 中，所有节点文件都可以直接调用。
    """
    if api_key_from_node and api_key_from_node.strip():
        return api_key_from_node # 优先使用节点中填写的Key
    if YOUR_DEFAULT_API_KEY and YOUR_DEFAULT_API_KEY.strip():
        return YOUR_DEFAULT_API_KEY # 其次使用代码中预设的Key
    return None # 两处都为空

# 将通用函数注入到 builtins，使其全局可用
builtins.get_api_key = get_api_key


# --- 节点加载逻辑 ---

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

print("--- 正在加载自定义节点包：ComfyUI-Nkxx ---")
print(f"  > 正在配置全局函数...")

current_dir = os.path.dirname(__file__)
for filename in os.listdir(current_dir):
    if filename.endswith(".py") and filename != "__init__.py":
        module_name = filename[:-3]
        try:
            module = importlib.import_module(f".{module_name}", package=__name__)
            
            if hasattr(module, "NODE_CLASS_MAPPINGS"):
                NODE_CLASS_MAPPINGS.update(module.NODE_CLASS_MAPPINGS)
            if hasattr(module, "NODE_DISPLAY_NAME_MAPPINGS"):
                NODE_DISPLAY_NAME_MAPPINGS.update(module.NODE_DISPLAY_NAME_MAPPINGS)
            
            print(f"  > 已成功加载节点文件: {filename}")

        except Exception as e:
            print(f"  > 加载节点文件 {filename} 时发生错误:")
            print(traceback.format_exc())

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

print(f"--- ComfyUI-Nkxx 加载完成，共找到 {len(NODE_CLASS_MAPPINGS)} 个节点 ---")
