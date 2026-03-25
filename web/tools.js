import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";


// ======================================================================
// 1. API Key 管理器 (ApiKeyManager) - 纯净双拼记事本版 (修复缩放与空隙)
// ======================================================================
app.registerExtension({
    name: "Nkxx.ApiKeyManager",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "ApiKeyManager") {

            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                if (onNodeCreated) onNodeCreated.apply(this, arguments);

                // 1. 彻底隐藏自带的无用输入框，抹除它们占用的一切空间
                for (const w of this.widgets) {
                    if (w.name !== "active_count") {
                        w.type = "hidden";
                        w.hidden = true;
                        w.computeSize = () => [0, 0]; 
                        w.draw = () => {}; 
                    }
                }

                // 2. 拔除所有连线端点
                while (this.inputs && this.inputs.length > 0) this.removeInput(0);
                while (this.outputs && this.outputs.length > 0) this.removeOutput(0);

                // 3. 构造双拼输入框 DOM 容器
                const container = document.createElement("div");
                container.style.display = "flex";
                container.style.flexDirection = "column";
                container.style.gap = "6px";
                container.style.width = "100%";
                container.style.marginTop = "2px"; // 极限压缩顶部空隙
                
                container.addEventListener("pointerdown", e => e.stopPropagation());
                container.addEventListener("keydown", e => e.stopPropagation());

                this._uiRows = [];
                for (let i = 1; i <= 10; i++) {
                    const row = document.createElement("div");
                    row.style.display = "none"; 
                    row.style.gap = "8px";
                    row.style.width = "100%";
                    row.style.height = "26px"; 

                    const inputStyle = {
                        background: "var(--comfy-input-bg, #1a1c1d)",
                        color: "var(--input-text, #ececec)",
                        border: "1px solid var(--border-color, #444)",
                        borderRadius: "4px",
                        padding: "0 8px",
                        fontSize: "12px",
                        outline: "none",
                        minWidth: "0",
                        boxSizing: "border-box"
                    };

                    const nameInput = document.createElement("input");
                    nameInput.type = "text";
                    nameInput.placeholder = `分组 ${i}`;
                    Object.assign(nameInput.style, inputStyle);
                    nameInput.style.flex = "4";

                    const keyInput = document.createElement("input");
                    keyInput.type = "text";
                    keyInput.placeholder = "在此输入 API Key";
                    Object.assign(keyInput.style, inputStyle);
                    keyInput.style.flex = "6";

                    nameInput.addEventListener("focus", () => nameInput.style.borderColor = "#4caf50");
                    nameInput.addEventListener("blur", () => nameInput.style.borderColor = "var(--border-color, #444)");
                    keyInput.addEventListener("focus", () => keyInput.style.borderColor = "#4caf50");
                    keyInput.addEventListener("blur", () => keyInput.style.borderColor = "var(--border-color, #444)");

                    nameInput.addEventListener("input", () => {
                        const w = this.widgets.find(x => x.name === `name${i}`);
                        if (w) w.value = nameInput.value;
                    });
                    keyInput.addEventListener("input", () => {
                        const w = this.widgets.find(x => x.name === `api_key${i}`);
                        if (w) w.value = keyInput.value;
                    });

                    row.appendChild(nameInput);
                    row.appendChild(keyInput);
                    container.appendChild(row);

                    this._uiRows.push({ row, nameInput, keyInput });
                }

                this.masterDOMWidget = this.addDOMWidget("master_ui", "custom", container);

                // 强制插队排版
                const cw = this.widgets.find(w => w.name === "active_count");
                const dw = this.masterDOMWidget;
                this.widgets = this.widgets.filter(w => w !== cw && w !== dw);
                this.widgets.unshift(dw); 
                this.widgets.unshift(cw); 

                setTimeout(() => {
                    if (cw) {
                        const origCb = cw.callback;
                        cw.callback = (val) => {
                            if (origCb) origCb.call(cw, val);
                            this.updateUI();
                        };
                    }
                    this.updateUI();
                }, 50);
            };

            const onConfigure = nodeType.prototype.onConfigure;
            nodeType.prototype.onConfigure = function () {
                if (onConfigure) onConfigure.apply(this, arguments);
                setTimeout(() => this.updateUI(), 50);
            };
            
            // 【核心修复 1】：提供老老实实的固定最小值，彻底解除缩小限制！
            nodeType.prototype.computeSize = function(out) {
                const cw = this.widgets.find(w => w.name === "active_count");
                const count = cw ? cw.value : 1;
                
                // 极致压缩高度计算：标题栏(30) + 控制器(20) + (每行高32)
                const minHeight = LiteGraph.NODE_TITLE_HEIGHT + 24 + (count * 32);
                const minWidth = 280; // 绝对最小宽度
                
                if (out) {
                    out[0] = minWidth; // 告诉系统：我就算缩到 280 也没关系！
                    out[1] = minHeight;
                    return out;
                }
                return [minWidth, minHeight];
            };

            // 【核心修复 2】：暴力拦截调整大小事件！
            nodeType.prototype.onResize = function(size) {
                const computed = this.computeSize();
                size[0] = Math.max(size[0], computed[0]); // 宽度随你拉
                size[1] = computed[1];                    // 高度？想都别想，死死锁住！
            };

            nodeType.prototype.updateUI = function() {
                if (!this.widgets || !this._uiRows) return;
                
                const cw = this.widgets.find(w => w.name === "active_count");
                const count = cw ? cw.value : 1;

                for (let i = 0; i < 10; i++) {
                    const rowObj = this._uiRows[i];
                    const wName = this.widgets.find(w => w.name === `name${i + 1}`);
                    const wKey = this.widgets.find(w => w.name === `api_key${i + 1}`);
                    
                    if (wName && wName.value !== undefined) rowObj.nameInput.value = wName.value;
                    if (wKey && wKey.value !== undefined) rowObj.keyInput.value = wKey.value;

                    rowObj.row.style.display = (i < count) ? "flex" : "none";
                }

                if (this.masterDOMWidget) {
                    this.masterDOMWidget.computeSize = function() {
                        return [0, (count * 32)]; 
                    };
                }

                while (this.outputs && this.outputs.length > 0) this.removeOutput(0);

                // 应用尺寸：宽度保持用户拖拽的样子，高度强行“切胃”
                const computed = this.computeSize();
                this.size[0] = Math.max(this.size[0], computed[0]); 
                this.size[1] = computed[1]; // 强制削掉下巴！
                
                app.graph.setDirtyCanvas(true, true);
            };
        }
    }
});

// ======================================================================
// 2. 图像动态列表 (DynamicImageList)
// ======================================================================
app.registerExtension({
    name: "Nkxx.DynamicImageBatch.Final",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "DynamicImageList") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            
            nodeType.prototype.onNodeCreated = function () {
                const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
                
                this.updatePorts = function() {
                    if (!this.widgets) return;

                    // 【保留轻量清理】：防止加载老工作流时“幽灵按钮”复活
                    const btnIndex = this.widgets.findIndex(w => w.type === "button" || w.name === "update_button");
                    if (btnIndex !== -1) {
                        this.widgets.splice(btnIndex, 1);
                    }

                    const countWidget = this.widgets.find(w => w.name === "inputcount");
                    if (!countWidget) return;
                    
                    const targetCount = countWidget.value;
                    let currentCount = 0;
                    
                    if (this.inputs) {
                        for (let i = 0; i < this.inputs.length; i++) {
                            if (this.inputs[i].name.startsWith("image_")) {
                                currentCount++;
                            }
                        }
                    }

                    let portsChanged = false;

                    // 增减端点
                    if (targetCount > currentCount) {
                        for (let i = currentCount + 1; i <= targetCount; i++) {
                            this.addInput(`image_${i}`, "IMAGE");
                        }
                        portsChanged = true;
                    } 
                    else if (targetCount < currentCount) {
                        for (let i = currentCount; i > targetCount; i--) {
                            const idx = this.inputs.findIndex(inp => inp.name === `image_${i}`);
                            if (idx !== -1) {
                                this.removeInput(idx);
                            }
                        }
                        portsChanged = true;
                    }
                    
                    // 强制高度自适应（如果端点变化，或者刚才切除了旧按钮）
                    if (portsChanged || btnIndex !== -1) {
                        this.size[1] = 10; // 强行压扁
                        this.setSize([this.size[0], this.computeSize([this.size[0], 10])[1]]);
                        app.graph.setDirtyCanvas(true, true);
                    }
                };

                // 劫持数字改变事件
                setTimeout(() => { 
                    const countWidget = this.widgets?.find(w => w.name === "inputcount");
                    if (countWidget) {
                        const origCallback = countWidget.callback;
                        const node = this;
                        
                        countWidget.callback = function(value) {
                            if (origCallback) origCallback.apply(this, arguments);
                            node.updatePorts();
                        };
                    }
                    this.updatePorts(); 
                }, 50);

                return r;
            };

            // 处理加载已有工作流时的端口恢复
            const onConfigure = nodeType.prototype.onConfigure;
            nodeType.prototype.onConfigure = function () {
                const r = onConfigure ? onConfigure.apply(this, arguments) : undefined;
                setTimeout(() => {
                    if (this.updatePorts) this.updatePorts();
                }, 50);
                return r;
            };
        }
    }
});


// ======================================================================
// 3. 快速运行中控台 (GroupRunnerConsole)
// ======================================================================
app.registerExtension({
    name: "Nkxx.GroupRunnerConsole",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        
        if (nodeData.name === "GroupRunnerConsole") {
            
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                if (onNodeCreated) onNodeCreated.apply(this, arguments);
                
                this.groupsList = [];
                this.buttonRects = []; 
                this.lastScanTime = 0;
                
                // 开启原生允许缩放
                this.resizable = true; 

                this.scanGroupsFromGraph = function() {
                    const groups = app.graph._groups || [];
                    this.groupsList = groups.map(g => ({
                        title: g.title,
                        id: g.id,
                        color: g.color || "#4a4a4a", 
                        ref: g 
                    }));
                    
                    // 只在节点过小时撑开，绝不干涉用户的放大操作
                    const minWidth = 160;
                    const minHeight = 28 + this.groupsList.length * 32 + 4;
                    this.size[0] = Math.max(this.size[0] || minWidth, minWidth);
                    this.size[1] = Math.max(this.size[1] || minHeight, minHeight);
                };
                
                this.scanGroupsFromGraph();
            };

            // 专业级文本截断器 (超出部分变省略号)
            function truncateText(ctx, text, maxWidth) {
                if (ctx.measureText(text).width <= maxWidth) return text;
                const ellipsis = "...";
                const ellipsisWidth = ctx.measureText(ellipsis).width;
                let currentStr = "";
                for (let i = 0; i < text.length; i++) {
                    let tempStr = currentStr + text[i];
                    if (ctx.measureText(tempStr).width + ellipsisWidth > maxWidth) {
                        return currentStr + ellipsis;
                    }
                    currentStr = tempStr;
                }
                return text;
            }

            // 【核心修复 1】：老老实实地只告诉引擎“绝对底线”是多少，彻底解除单向死锁
            nodeType.prototype.computeSize = function(out) {
                const minWidth = 160; 
                const minHeight = 28 + (this.groupsList?.length || 0) * 32 + 4;
                if (out) {
                    out[0] = minWidth;
                    out[1] = minHeight;
                    return out;
                }
                return [minWidth, minHeight];
            };

            // 【核心修复 2】：在用户拖拽时，只拦截“比底线还小”的情况，变大完全自由
            nodeType.prototype.onResize = function(size) {
                const minWidth = 160;
                const minHeight = 28 + (this.groupsList?.length || 0) * 32 + 4;
                if (size[0] < minWidth) size[0] = minWidth;
                if (size[1] < minHeight) size[1] = minHeight;
            };

            nodeType.prototype.onDrawForeground = function (ctx, graphCanvas) {
                if (this.mode === 2 || this.mode === 4) return;

                const now = Date.now();
                if (now - this.lastScanTime > 3000) {
                    const oldLen = this.groupsList.length;
                    this.scanGroupsFromGraph();
                    if (oldLen !== this.groupsList.length) {
                        app.graph.setDirtyCanvas(true, true);
                    }
                    this.lastScanTime = now;
                }

                if (this.groupsList.length === 0) {
                    ctx.fillStyle = "#888";
                    ctx.textAlign = "center";
                    ctx.font = "12px Arial";
                    ctx.fillText("⚠️ 请添加 Group", this.size[0] / 2, this.size[1] / 2);
                    return;
                }

                const pillPadding = 10;
                const pillHeight = 24; 
                const pillRadius = pillHeight / 2; 
                let currentY = 28; 
                this.buttonRects = []; 

                // UI 宽度直接读取当前的物理宽度
                const pillWidth = this.size[0] - pillPadding * 2;
                const pillX = pillPadding;

                ctx.save();
                ctx.textBaseline = "middle";
                ctx.font = "13px Arial";

                for (const group of this.groupsList) {
                    const pillY = currentY;

                    ctx.fillStyle = "#2a2a2a"; 
                    ctx.beginPath();
                    ctx.roundRect(pillX, pillY, pillWidth, pillHeight, pillRadius); 
                    ctx.fill();
                    
                    ctx.strokeStyle = "#444";
                    ctx.lineWidth = 1;
                    ctx.stroke();

                    ctx.fillStyle = group.color;
                    ctx.beginPath();
                    ctx.arc(pillX + pillRadius + 2, pillY + pillHeight / 2, 5, 0, Math.PI * 2);
                    ctx.fill();

                    ctx.fillStyle = "#DDD";
                    ctx.textAlign = "left";
                    
                    const fullText = `▶ 运行: ${group.title}`;
                    const textStartX = pillX + pillRadius * 2 + 10; 
                    const maxTextWidth = pillWidth - (textStartX - pillX) - 10; 
                    
                    const displayText = truncateText(ctx, fullText, maxTextWidth);
                    ctx.fillText(displayText, textStartX, pillY + pillHeight / 2);

                    this.buttonRects.push({
                        rect: [pillX, pillY, pillWidth, pillHeight],
                        groupRef: group.ref
                    });

                    currentY += pillHeight + 8; 
                }
                ctx.restore();
            };

            // 保持点击与隔离运行逻辑不变
            nodeType.prototype.onMouseDown = function (e, localPos, graphCanvas) {
                if (this.mode === 2) return;
                for (const btn of this.buttonRects) {
                    const r = btn.rect;
                    if (localPos[0] >= r[0] && localPos[0] <= r[0] + r[2] &&
                        localPos[1] >= r[1] && localPos[1] <= r[1] + r[3]) {
                        runSpecificGroup(btn.groupRef);
                        graphCanvas.dirty_canvas = true;
                        return true; 
                    }
                }
                return false; 
            };

            const runSpecificGroup = (group) => {
                const requiredNodeIds = new Set();
                app.graph._nodes.forEach(n => {
                    if (n.pos[0] >= group.pos[0] && n.pos[0] <= group.pos[0] + group.size[0] &&
                        n.pos[1] >= group.pos[1] && n.pos[1] <= group.pos[1] + group.size[1]) {
                        requiredNodeIds.add(n.id);
                    }
                });

                if (requiredNodeIds.size === 0) {
                    alert(`❌ 组 [${group.title}] 内部没有包含任何有效节点！`);
                    return;
                }

                function getAncestors(node) {
                    if (node.inputs) {
                        for (const input of node.inputs) {
                            if (input.link) {
                                const link = app.graph.links[input.link];
                                if (link) {
                                    const originNode = app.graph.getNodeById(link.origin_id);
                                    if (originNode && !requiredNodeIds.has(originNode.id)) {
                                        requiredNodeIds.add(originNode.id);
                                        getAncestors(originNode);
                                    }
                                }
                            }
                        }
                    }
                }
                
                app.graph._nodes.forEach(n => {
                    if (requiredNodeIds.has(n.id)) getAncestors(n);
                });

                const originalModes = new Map();
                app.graph._nodes.forEach(n => {
                    originalModes.set(n.id, n.mode);
                    if (!requiredNodeIds.has(n.id)) {
                        n.mode = 2; 
                    } else if (n.mode === 2) {
                        n.mode = 0; 
                    }
                });

                app.queuePrompt(0, 1);

                setTimeout(() => {
                    app.graph._nodes.forEach(n => {
                        if (originalModes.has(n.id)) {
                            n.mode = originalModes.get(n.id);
                        }
                    });
                    app.graph.setDirtyCanvas(true, true);
                }, 100);
            };
        }
    }
});


// ======================================================================
// 4. 可选模型面板 (WujiaiModelBrowser) - 稳健原生美化版
// ======================================================================
app.registerExtension({
    name: "Nkxx.Wujiai.Tools.ModelBrowser",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "WujiaiModelBrowser") {
            
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                if (onNodeCreated) onNodeCreated.apply(this, arguments);

                const displayWidget = this.widgets.find((w) => w.name === "model_list_display");
                if (!displayWidget) return;

                // --- 1. 给原生文本框“整容”的 CSS ---
                const styleInput = (el) => {
                    if (!el || el.tagName !== "TEXTAREA") return;
                    el.style.backgroundColor = "rgba(15, 15, 15, 0.9)"; // 深色半透明背景
                    el.style.color = "#ececec";                       // 亮灰色文字
                    el.style.fontFamily = "'Consolas', 'Monaco', monospace"; 
                    el.style.fontSize = "14px";
                    el.style.lineHeight = "1.5";
                    el.style.padding = "12px";
                    el.style.borderRadius = "8px";
                    el.style.border = "1px solid #444";
                    el.style.outline = "none";
                    el.style.boxShadow = "inset 0 0 10px rgba(0,0,0,0.5)";
                    el.readOnly = true; // 只读模式，防止误触弹出输入法
                };

                let configTree = {};

                // --- 2. 纯文本模拟 Markdown 结构 ---
                this._updateDisplay = () => {
                    const catWidget = this.widgets.find(w => w.name === "category");
                    const currentCategory = catWidget ? catWidget.value : "";
                    
                    let text = "";
                    if (configTree["顶部公告"]) {
                        text += configTree["顶部公告"].join("\n") + "\n";
                        text += "━".repeat(25) + "\n"; // 使用特殊符号模拟分割线
                    }
                    if (configTree[currentCategory]) {
                        text += configTree[currentCategory].join("\n");
                    }

                    displayWidget.value = text;

                    // 高度自适应：每行约 21px + 底部边距
                    const lines = text.split("\n").length;
                    this.setSize([this.size[0], Math.max(160, lines * 21 + 80)]);
                };

                // --- 3. 稳健的 DOM 注入监听 ---
                // 利用 settimeout 确保在 DOM 挂载后立即注入样式
                const applyStyle = () => {
                    if (displayWidget.inputEl) {
                        styleInput(displayWidget.inputEl);
                        this._updateDisplay();
                    } else {
                        setTimeout(applyStyle, 50);
                    }
                };
                applyStyle();

                // 获取数据
                api.fetchApi("/nkxx/wujiai/models_config").then(r => r.json()).then(data => {
                    configTree = data;
                    this._updateDisplay();
                });
            };

            // 监听下拉框切换
            const onDrawBackground = nodeType.prototype.onDrawBackground;
            nodeType.prototype.onDrawBackground = function (ctx) {
                if (onDrawBackground) onDrawBackground.apply(this, arguments);
                const cat = this.widgets.find(w => w.name === "category");
                if (cat && this._lastCat !== cat.value) {
                    this._lastCat = cat.value;
                    if (this._updateDisplay) this._updateDisplay();
                }
            };
        }
    }
});