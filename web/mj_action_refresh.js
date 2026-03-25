import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

app.registerExtension({
    name: "Wujiai.MJ.ActionRefresh",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        // 只有当 ComfyUI 加载到我们写的动作节点时，才触发魔法
        if (nodeData.name === "MJActionSubmit") {
            // 保存原有的 onNodeCreated 钩子
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            
            nodeType.prototype.onNodeCreated = function () {
                // 先执行原有的逻辑
                const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
                
                // --- 魔法开始：在节点上动态添加一个按钮 ---
                this.addWidget("button", "🔄 刷新最新任务", "button", async () => {
                    try {
                        // 1. 发起请求：调用我们在 Python 里写好的接口
                        const response = await api.fetchApi("/wujiai_mj/get_recent_tasks");
                        if (!response.ok) {
                            throw new Error(`HTTP error! status: ${response.status}`);
                        }
                        const newTasks = await response.json();
                        
                        // 2. 找到节点上名字叫 "recent_task" 的下拉框
                        const widget = this.widgets.find(w => w.name === "recent_task");
                        
                        if (widget) {
                            // 3. 替换选项数据并重置选中项
                            widget.options.values = newTasks;
                            widget.value = newTasks[0]; 
                            
                            // [关键修复]：手动触发 widget 的回调函数，强制 ComfyUI 内部同步数据状态
                            if (widget.callback) {
                                widget.callback(widget.value, app.canvas, this, app.canvas.graph.canvas, undefined);
                            }
                            
                            // 4. 强制重绘画布，让用户立刻看到变化
                            app.graph.setDirtyCanvas(true, false);
                        }
                    } catch (error) {
                        console.error("[Wujiai MJ] 刷新任务列表失败:", error);
                        alert("刷新任务列表失败，请检查控制台报错或确认后端服务正常。");
                    }
                });
                return r;
            };
        }
    }
});