# ComfyUI-Nkxx Custom Nodes

这是一个功能强大的 ComfyUI 自定义节点包，集成了 **GRSAI** 的多种服务。

本插件涵盖了**高质量图像生成 (Nano Banana, Sora-Image)**、**AI 视频生成 (Sora-2, Veo 3.1)** 以及 **多模态大模型 (Gemini)** 的工作流集成。特别针对**批量生产**、**异步任务管理**和**长视频/角色一致性**进行了深度优化。

## 📢 重要资源与支持 (Important)

* 🔑 **API Key 申请地址**: [**nkxx.grsai.com**](http://nkxx.grsai.com)
    > 本节点包依赖该 API 服务，请先前往注册申请 Key。
* 📺 **作者 Bilibili 主页**: [**点击关注**](https://space.bilibili.com/3546882187987924)
    > 欢迎大家关注我的B站账号并分享给更多的朋友使用！

---
## ✨ 核心功能亮点 (Features)

* **🍌 Nano Banana 图像生成**
    * 支持 **同步 (Sync)** 与 **异步 (Async)** 两种模式。
    * **超强批量处理**：支持读取 CSV/Excel 文件批量生成，支持文件夹内图片遍历处理。
    * **Pro 模型支持**：支持 2K/4K 高清分辨率，单次并发可达 14 张图。
    * **智能熔断**：内置积分保护和错误重试机制。
* **🤖 Grsai GPT Image (Sora-Image)**
    * 基于 `sora-image` 模型的新一代图像生成。
    * **多图参考**：支持上传最多 5 张参考图 (Ref Images) 进行风格/内容控制。
    * **高并发支持**：支持设置并发数 (Concurrency) 和单次变体数量 (Variants)。
* **🎬 Sora-2 视频生成**
    * **全功能支持**：文生视频、图生视频、视频续写 (Remix)。
    * **角色一致性**：支持通过上传视频或 PID 创建固定角色 (Character Consistency)。
        * **自动记录**：创建成功的角色信息会自动保存在插件目录下的 `character_library.txt` 文件中，方便随时调用。
    * **异步队列管理**：提交任务 -> 离线排队 -> 自动查询并下载，无需一直挂机等待。
    * **下载增强**：内置 `yt-dlp` 和 `curl` 双重下载保障，解决大文件下载失败问题。
* **🎥 Google Veo 3.1 视频**
    * 支持 **Veo 3.1** 模型。
    * **强控能力**：支持多张参考图 (Ref Image) 生成。
    * **首尾帧控制**：支持指定 First Frame 和 Last Frame 进行视频生成。
* **🧠 LLM & VLM 多模态**
    * 集成 **Gemini 2.5 / 3 Pro** 模型。
    * 支持纯文本对话及 **视觉理解 (VLM)**（图文对话）。

---

## 🛠️ 安装方法 (Installation)

1.  进入 ComfyUI 的 `custom_nodes` 目录：
    ```bash
    cd ComfyUI/custom_nodes/
    ```
2.  克隆本项目：
    ```bash
    git clone [https://github.com/jieg9341-lab/ComfyUI-nkxx.git](https://github.com/jieg9341-lab/ComfyUI-nkxx.git)
    ```
3.  依赖安装：
    * 插件启动时会自动检测并安装核心依赖 (`requests`, `pandas`, `openpyxl`, `yt-dlp` 等)。
    * 如果自动安装失败，请手动运行：
        ```bash
        pip install requests pandas openpyxl yt-dlp
        ```

---

## 🔑 配置 API Key (Configuration)

你可以通过以下三种方式配置 GRSAI 的 API Key（优先级由高到低）：

1.  **节点输入框**：在每个节点的 `api_key` 选项中直接填入。
2.  **全局配置 (推荐)**：打开 `__init__.py` 文件，修改 `YOUR_DEFAULT_API_KEY` 变量：
    ```python
    YOUR_DEFAULT_API_KEY = "sk-xxxxxxxxxxxxxxxxxxxxxxxx"
    ```
3.  **环境变量**：设置系统环境变量 `GRSAI_KEY`。

---

## 📦 节点详解 (Nodes Description)

### 1. 🖼️ 图像生成 (Image Generation)

| 节点名称 | 功能描述 |
| :--- | :--- |
| **Grsai Nano Banana** | 基础生图，支持 1-8 张并发，适用于快速测试。 |
| **Grsai Nano Banana Pro** | 专业版，支持 **2K/4K** 分辨率，最大支持 14 张并发。 |
| **Grsai Nano Banana Batch** | **批量神器**。读取 CSV/Excel 中的 Prompt 列，自动批量跑图。 |
| **Grsai GPT Image (Sora-Image)** | **[NEW]** 基于 Sora-Image 模型，支持 **5张参考图** 输入，支持设置生成变体数与并发数。 |
| **Grsai 文件夹批量处理** | 读取文件夹内的图片 + CSV 提示词，进行批量图生图/重绘操作，含熔断保护。 |

### 2. 🎬 Sora-2 (视频生成)

| 节点名称 | 功能描述 |
| :--- | :--- |
| **Sora2 Generator** | 标准生成节点，支持文生、图生、设置时长和比例。 |
| **Sora2 异步提交/批量提交** | 发送视频任务到云端队列，支持 CSV 批量提交视频任务。 |
| **Sora2 查询与下载** | 轮询云端任务状态，自动下载完成的视频 (支持断点续传/重试)。 |
| **Sora2 上传/PID 创建角色** | 用于保持角色一致性。上传本地视频或引用现有 PID 注册角色。<br>📝 **注意**：创建成功的角色ID和备注会自动追加写入到插件同级目录下的 `character_library.txt` 文件中。 |

### 3. 🎥 Veo 3.1 (视频生成)

| 节点名称 | 功能描述 |
| :--- | :--- |
| **Veo3.1 参考图生成** | 支持上传最多 3 张参考图来控制视频内容。 |
| **Veo3.1 首尾帧生成** | 支持指定视频的起始帧和结束帧，精准控制视频动态。 |

### 4. 🧠 LLM / 工具 (语言模型与辅助)

| 节点名称 | 功能描述 |
| :--- | :--- |
| **Grsai LLM (文本对话)** | 调用 Gemini 模型进行对话。 |
| **Grsai VLM (图文对话)** | 上传图片给 Gemini，让 AI 描述或分析图片内容。 |
| **Grsai LLM Writer** | 专门用于辅助写作，可批量生成 Prompt 并保存为 CSV 供绘画节点使用。 |
| **Save Image By Filename** | 智能保存节点，支持自定义文件名和目录结构。 |

---

## 📂 批量任务指南 (Batch Guide)

本插件对批量任务提供了极佳的支持，尤其是 **CSV/Excel** 联动。

### 数据文件格式
请准备一个 `.csv` 或 `.xlsx` 文件，格式非常简单，只需包含一个标题行（列名），下方填入提示词即可：

| prompt |
| :--- |
| 生成45度侧视图，半身特写 |
| 生成头肩特写图，正视图 |
| 生成头肩特写图，侧视图 |

**使用方法：**
1. 在 Excel 中第一行输入列名（例如 `prompt`）。
2. 在下方单元格中填入你需要批量生成的提示词。
3. 在 ComfyUI 批量节点的 `file_path` 中填入该文件的绝对路径。
4. 在 `column_name` 中填入你的列名（如 `prompt`）。

---

## 📝 更新日志

* **v1.x**: 初始版本，集成 Nano Banana 和 Sora2。
* **Update**: 新增 `Grsai GPT Image` 节点，新增功能齐全的异步香蕉工作流，优化 CSV 读取逻辑，增加角色库自动记录功能。

---

**Disclaimer**: This project is a third-party plugin for ComfyUI and is not officially affiliated with the model providers. Use responsibly.