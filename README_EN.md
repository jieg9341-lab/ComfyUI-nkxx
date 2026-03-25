# ComfyUI-nkxx (Na Ke Xing Xing API Node Pack)

This is a comprehensive AI node plugin pack built specifically for ComfyUI, deeply integrated with two powerful model ecosystems: **Grsai** and **Wujiai**. Whether you need high-end image generation, high-quality video generation, or large-model-based visual understanding and prompt generation, this pack lets you complete everything efficiently in one place inside ComfyUI.

## Author and API Access

* **Author Bilibili**: [Na Ke Xing Xing 188](https://space.bilibili.com/3546882187987924?spm_id_from=333.1007.0.0)  
  Follow for the latest tutorials and workflow updates.
* **Grsai Channel API**: [nkxx.grsai.ai](http://nkxx.grsai.ai)
* **Wujiai Channel API**: [Click here to register](https://wujiai.org/register?aff=vW2s)

---

## Installation Guide

### 1. Install the node pack

Place this folder (`ComfyUI-nkxx`) directly into your ComfyUI custom nodes directory:

`ComfyUI/custom_nodes/ComfyUI-nkxx`

Note: On the first restart of ComfyUI, the node pack will automatically detect and try to install required Python dependencies such as `requests`, `pandas`, and `yt-dlp`. If the automatic installation fails, please manually install the missing packages with `pip`.

### 2. Import the built-in workflows

This node pack includes a set of carefully prepared workflows from the author. Copy everything inside the `workflows` directory of this pack into ComfyUI's default user workflow folder:

`ComfyUI/user/default/workflows/`

After copying, refresh the ComfyUI page. You will then be able to load the built-in workflow sets such as `NaKeXingXing_workflows_grsai` and `NaKeXingXing_workflows_wujiai` directly from the `Load` panel or workflow list.

---

## Core Feature Modules

### Wujiai Channel

* **Universal Image Generator**: A single node system that unifies multiple models such as Gemini, Nano Banana, Grok, GPT, and Seedream. Supports single-image synchronous generation, asynchronous generation, powerful CSV/Excel batch processing, and local folder batch generation.
* **Midjourney Wujiai Edition**: Recreates the MJ workflow experience inside ComfyUI. Supports base image input, style reference (`--sref`), character reference (`--cref`), and includes **exclusive frontend button interaction support** so U/V/Reroll actions can be refreshed directly from the node panel. Also includes automatic 4-grid splitting.
* **Advanced Video Generation**:
  * **Sora2**: Supports image-to-video, text-to-video, and asynchronous batch submission.
  * **Veo 3.1**: Supports first/last frame control, multi-reference images, and 4K generation.
  * **Grok Video**: Supports `grok-video-3` and longer 10s / 15s video generation.

### Grsai Channel

* **Nano Banana Image Engine**: Supports nanobanana 1, 2, and Pro models, with up to 4K output and up to 14 reference images at once. Supports single-image workflows, CSV/Excel batch forms, and refined named export workflows.
* **Sora2 Video and Character Customization**: Supports video generation plus powerful character upload and character extraction from source video, with automatic local archiving of character IDs.
* **GPT Image (Sora-Image)**: Supports OpenAI's latest image model with compatible handling for streaming-style response data.
* **Veo 3.1 Video**: Supports both synchronous and asynchronous generation, with full support for first/last frame control and multi-reference image control.
* **VLM / LLM Visual and Language Assistant**: Integrates Gemini 2.5 / 3.1 series models. Can handle text-only chat, multi-image visual analysis, and is especially suitable for automatically generating Midjourney or image-generation prompts and saving them into CSV files.

### Exclusive Utility Toolkit

* **API Key Manager**: Visual global API key management with support for multiple keys and switching, plus polished built-in UI styling.
* **Quick Running Console (GroupRunner)**: Reads all Groups in a workflow and lets you run a specified Group with one click, greatly improving debugging efficiency for large workflows.
* **Dynamic Image List**: Dynamically increases or decreases image input ports based on a number, making flexible image merging easier.
* **Cloud-Synced Model List Updates**: Built-in remote model list synchronization, so the node pack can obtain the latest supported models without constantly updating the plugin code.

---

## Core Advantages and Technical Highlights

1. **Circuit Breaker Protection**
   * Includes intelligent request frequency detection. If ComfyUI's `Auto Queue` causes repeated submission of the same prompt within a short time, the system automatically triggers a circuit breaker to stop the requests and prevent accidental API credit or quota exhaustion.

2. **Enterprise-Grade Asynchronous Task Management**
   * Built on a thread-safe `TaskManager`, all asynchronous tasks are automatically saved into local JSON databases under the `data/` directory. You do not need to worry about crashes or restarts. You can resume checking progress and downloading results at any time.

3. **Multi-Threading and Robust Download Fallback**
   * The video downloader includes dual fallback mechanisms using `yt-dlp` and `curl`, helping maintain a high success rate even with large files or unstable network conditions.

4. **Seamless Batch Automation**
   * Deep support for reading `.csv`, `.xls`, and `.xlsx` files for batch image or video jobs.
   * Supports scanning entire local folders of images, combining them with CSV prompts for concurrent batch image-to-image generation, and automatically processing and saving outputs into the target directory.
