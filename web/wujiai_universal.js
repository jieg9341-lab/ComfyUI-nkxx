import { app } from "../../scripts/app.js";

app.registerExtension({
    // 修改了扩展名，利用 ComfyUI 的机制强制浏览器更新缓存
    name: "Nkxx.Wujiai.Universal.V4.Seedream.Update", 
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        // 在这里加入了新的两个批量节点
        const targetNodes = [
            "UniversalSyncGeneratorWujiai", 
            "UniversalAsyncSubmitWujiai",
            "UniversalAsyncBatchSubmitWujiai",
            "UniversalBatchDirWujiai"
        ];
        
        if (targetNodes.includes(nodeData.name)) {
            
            const onDrawBackground = nodeType.prototype.onDrawBackground;
            
            nodeType.prototype.onDrawBackground = function (ctx) {
                if (onDrawBackground) {
                    onDrawBackground.apply(this, arguments);
                }

                if (!this.widgets) return;

                const modelWidget = this.widgets.find(w => w.name === "model_name");
                const aspectWidget = this.widgets.find(w => w.name === "aspect_ratio");
                const sizeWidget = this.widgets.find(w => w.name === "image_size");
                const nWidget = this.widgets.find(w => w.name === "n");
                
                // 组图功能组件
                const seqModeWidget = this.widgets.find(w => w.name === "sequential_mode");
                const maxImgWidget = this.widgets.find(w => w.name === "max_images");

                if (!modelWidget || !aspectWidget) return;

                let currentModel = modelWidget.value || "";

                const modelInputIdx = this.inputs ? this.inputs.findIndex(inp => inp.name === "model_name") : -1;
                if (modelInputIdx !== -1) {
                    const modelInput = this.inputs[modelInputIdx];
                    if (modelInput && modelInput.link != null) {
                        const link = app.graph.links[modelInput.link];
                        if (link) {
                            const sourceNode = app.graph.getNodeById(link.origin_id);
                            if (sourceNode && sourceNode.widgets) {
                                const sourceWidget = sourceNode.widgets[link.origin_slot] || sourceNode.widgets[0];
                                if (sourceWidget) {
                                    currentModel = sourceWidget.value;
                                }
                            }
                        }
                    }
                }

                currentModel = currentModel.toLowerCase();
                
                let currentFamily = "gemini_or_default"; 
                if (currentModel.includes("seedream-5")) {
                    currentFamily = "seedream_5";
                } else if (currentModel.includes("seedream-4-5")) {
                    currentFamily = "seedream_4_5";
                } else if (currentModel.includes("seedream-4")) { // 修复：同时兼容 seedream-4 和 seedream-4-0
                    currentFamily = "seedream_4_0";
                } else if (currentModel.includes("grok") || currentModel.includes("gpt")) {
                    currentFamily = "strict_size";
                }

                let currentSeqMode = seqModeWidget ? seqModeWidget.value : "disabled";
                let stateKey = currentFamily + "_" + currentSeqMode;

                if (this._lastStateKey === stateKey) {
                    return; 
                }
                this._lastStateKey = stateKey; 

                let needsResize = false;

                const hideWidget = (widget) => {
                    if (widget && widget.type !== "hidden" && widget.type !== "converted-widget") {
                        widget.origType = widget.type;
                        widget.origComputeSize = widget.computeSize;
                        widget.type = "hidden";
                        widget.computeSize = () => [0, -4]; 
                        needsResize = true;
                    }
                };

                const showWidget = (widget, defaultType) => {
                    if (widget && widget.type === "hidden") {
                        widget.type = widget.origType || defaultType;
                        widget.computeSize = widget.origComputeSize || (() => [220, 20]);
                        needsResize = true;
                    }
                };

                if (currentFamily === "strict_size") {
                    const strictAspects = ["1:1", "2:3", "3:2", "auto"];
                    if (aspectWidget.options && aspectWidget.options.values.join() !== strictAspects.join()) {
                        aspectWidget.options.values = strictAspects;
                        if (!strictAspects.includes(aspectWidget.value)) aspectWidget.value = "1:1";
                        needsResize = true;
                    }
                    hideWidget(sizeWidget); 
                    showWidget(nWidget, "number"); 
                    hideWidget(seqModeWidget);
                    hideWidget(maxImgWidget);
                } 
                else if (currentFamily.startsWith("seedream")) {
                    // 加入了 auto 选项
                    const sdAspects = ["1:1", "4:3", "3:4", "16:9", "9:16", "3:2", "2:3", "21:9", "auto"];
                    if (aspectWidget.options && aspectWidget.options.values.join() !== sdAspects.join()) {
                        aspectWidget.options.values = sdAspects;
                        if (!sdAspects.includes(aspectWidget.value)) aspectWidget.value = "1:1";
                        needsResize = true;
                    }

                    showWidget(sizeWidget, "combo"); 
                    hideWidget(nWidget); 
                    
                    showWidget(seqModeWidget, "combo");
                    if (currentSeqMode === "auto") {
                        showWidget(maxImgWidget, "number");
                    } else {
                        hideWidget(maxImgWidget);
                    }

                    let sizeOptions = [];
                    if (currentFamily === "seedream_5") sizeOptions = ["2K", "3K"];
                    else if (currentFamily === "seedream_4_5") sizeOptions = ["2K", "4K"];
                    else if (currentFamily === "seedream_4_0") sizeOptions = ["1K", "2K", "4K"];

                    if (sizeWidget && sizeWidget.options) {
                        if (sizeWidget.options.values.join() !== sizeOptions.join()) {
                            sizeWidget.options.values = sizeOptions;
                            if (!sizeOptions.includes(sizeWidget.value)) sizeWidget.value = sizeOptions[0];
                            needsResize = true;
                        }
                    }
                }
                else {
                    const defAspects = ["1:1", "16:9", "9:16", "3:4", "4:3", "2:3", "3:2", "auto"];
                    if (aspectWidget.options && aspectWidget.options.values.join() !== defAspects.join()) {
                        aspectWidget.options.values = defAspects;
                        needsResize = true;
                    }
                    showWidget(sizeWidget, "combo"); 
                    hideWidget(nWidget); 
                    hideWidget(seqModeWidget);
                    hideWidget(maxImgWidget);
                    
                    if (sizeWidget && sizeWidget.options) {
                        const sizeOptions = ["默认", "1K", "2K", "4K"];
                        if (sizeWidget.options.values.join() !== sizeOptions.join()) {
                            sizeWidget.options.values = sizeOptions;
                            if (!sizeOptions.includes(sizeWidget.value)) sizeWidget.value = "默认";
                            needsResize = true;
                        }
                    }
                }

                if (needsResize) {
                    this.setSize([this.size[0], this.computeSize()[1]]);
                    app.graph.setDirtyCanvas(true, true);
                }
            };
        }
    }
});