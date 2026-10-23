# LLM Image Workflow

一个把「一段文字」变成「一张配图」的工作流工具：识别意图 → 生成英文生图提示词 → 调用图像模型出图。界面基于 Gradio Blocks，左右分栏，左侧走提示词流程，右侧出图。

## 功能

1. **意图识别**：把输入分成三类——无意义内容 / 文章配图需求 / 直接绘图指令，并据此决定后续流程是否可走
2. **提示词生成**：调用 LLM 把中文文本转成可直接用于图像模型的英文提示词，带格式校验（纯英文、含风格词、长度 50–200）
3. **文生图 / 图生图**：有参考图时走图生图分支，参考图自动转 PNG、铺白底、缩放到 1024×1024
4. **响应解析**：兼容结构化 `image_url` 与 Markdown 图片两种返回格式

## 界面

左侧 60%：输入文本 → 识别意图 → System Prompt → 生成生图提示词 → 提示词编辑框
右侧 40%：参考图上传 → 生成图片按钮 → 结果预览

（截图待补充：`docs/screenshot.png`）

## 本地运行

```bash
pip install -r requirements.txt
```

设置环境变量（两个密钥分属不同服务）：

```bash
# 意图识别 / 提示词生成（Siliconflow）
export SILICONFLOW_API_KEY="sk-..."
# 图像生成（Nanobanana 兼容接口）
export NANOBANANA_API_KEY="..."
```

Windows PowerShell：

```powershell
$env:SILICONFLOW_API_KEY = "sk-..."
$env:NANOBANANA_API_KEY = "..."
```

启动：

```bash
python app.py
```

打开 http://127.0.0.1:7860

## 配置说明

| 变量 | 位置 | 用途 |
|---|---|---|
| `LLM_BASE_URL` | `app.py` 常量 | LLM 接口地址，默认 Siliconflow |
| `LLM_MODEL` | `app.py` 常量 | 默认 `Qwen/Qwen2.5-7B-Instruct` |
| `NANOBANANA_API_URL` | `app.py` 常量 | 图像生成接口地址 |
| `NANOBANANA_MODEL` | `app.py` 常量 | 默认 `gemini-2.5-flash-image` |
| `SILICONFLOW_API_KEY` | 环境变量 | LLM 鉴权 |
| `NANOBANANA_API_KEY` | 环境变量 | 图像模型鉴权 |

密钥一律通过环境变量注入，代码中不保存明文。

## 部署到 Hugging Face Spaces（免费）

1. 在 huggingface.co 用 GitHub 账号登录，New Space → SDK 选 Gradio
2. 上传 `app.py` 和 `requirements.txt`
3. Space 的 `Settings → Variables and secrets` 里添加上述两个 Key
4. 注意：公开 Space 会消耗你的模型额度，建议先用免费额度试，或只开放界面不配 Key

## 已知限制

- 意图识别对长文本偶有误判（模型判断力所限）
- 图像生成依赖第三方接口，返回格式可能变化
- 未配置密钥时界面可正常打开，但点击生成会提示配置缺失

## License

MIT
