# /// script
# dependencies = [
#  "gradio>=4.0.0",
#  "gradio_client>=1.0.0",
#  "pillow>=10.0.0",
#  "requests>=2.31.0",
#  "openai",
# ]
# ///
"""
板块 1 + 板块 2 + 板块 3 + 板块 4：LLM 驱动的文生图全流程工具

板块 1：固定左右分栏布局（左 60% / 右 40%），组件初始化状态。
板块 2：「识别意图」-> LLM 意图识别 -> 联动组件状态。
板块 3：「确认生成生图提示词」-> 二次调用 LLM -> 生成英文视觉提示词 -> 联动「生成图片」。
板块 4：「生成图片」-> 调用 Nanobanana API（文生图 / 图生图）-> 解析 Base64 -> 展示图片。
"""

import base64
import io
import os
import re
from typing import Optional, Tuple

import gradio as gr
import requests
from PIL import Image

# ============================================================
# Siliconflow LLM 配置（板块 2/3 共用）
# ============================================================
LLM_BASE_URL = "https://api.siliconflow.cn/v1"
LLM_API_KEY = os.environ.get("SILICONFLOW_API_KEY", "")
LLM_MODEL = "Qwen/Qwen2.5-7B-Instruct"
LLM_CHAT_URL = f"{LLM_BASE_URL}/chat/completions"

# ============================================================
# Nanobanana 图片生成 API 配置（板块 4）
# ============================================================
NANOBANANA_API_URL = "https://api.zyai.online/v1/chat/completions"
NANOBANANA_MODEL = "gemini-2.5-flash-image"
NANOBANANA_API_KEY = os.environ.get("NANOBANANA_API_KEY", "")  # 用户自行替换

# ============================================================
# 板块 2：意图识别模板与状态文案
# ============================================================
INTENT_PROMPT_TEMPLATE = """你需要识别用户输入文本的意图，仅返回以下 3 类结果中的一种（格式：数字 + 中文描述）：
1 = 无意义内容；2 = 文章 / 长文本配图需求；3 = 直接绘图指令。

用户输入：{user_input}

识别结果：
仅提取返回结果中的数字和描述，禁止额外内容。"""

INTENT_STATUS_UNKNOWN = "未识别意图"
INTENT_STATUS_1 = "1 = 无意义内容：无绘图需求"
INTENT_STATUS_2 = "2 = 文章 / 长文本配图需求：为输入内容生成配图"
INTENT_STATUS_3 = "3 = 直接绘图指令：根据指令生成图片"

# ============================================================
# 板块 3：生图提示词生成相关常量
# ============================================================
SYSTEM_PROMPT_DEFAULT = """你现在是一个创建NanoBanana画图提示词的助手。
需要根据我的内容处理，我这个图片的作用是能说明这一段在说什么，并且让大家知道这段话的上下结构就是整体说的是什么意思。
里面可能会类似PPT有一些讲解（如：左上角展示核心观点，右下角展示数据）。
设计风格要求：简约，Apple设计思维（Apple Design Philosophy）。
约束：请直接返回NanoBanana可用的英文提示词，不要返回任何解释、前缀或多余的废话。"""

PROMPT_GEN_TEMPERATURE = 0.7
PROMPT_GEN_MAX_TOKENS = 200
PROMPT_MIN_LEN = 50
PROMPT_MAX_LEN = 200

CONFIRM_BTN_LABEL = "确认生成生图提示词"
GENERATE_BTN_LABEL = "生成图片"
PROMPT_OK_SUFFIX = "提示词生成成功，可修改后生成图片"


# ============================================================
# 通用 LLM 调用
# ============================================================
def _chat_completion(messages, temperature: float = 0.1, max_tokens: int = 2048, timeout: int = 60) -> str:
    headers = {
        "Authorization": f"Bearer {LLM_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": LLM_MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    response = requests.post(LLM_CHAT_URL, headers=headers, json=payload, timeout=timeout)
    if response.status_code != 200:
        raise RuntimeError(f"接口返回 HTTP {response.status_code}：{response.text[:200]}")
    return response.json()["choices"][0]["message"]["content"]


# ============================================================
# 板块 2：意图识别
# ============================================================
def _parse_intent(raw_text: str) -> int:
    text = (raw_text or "").strip()
    match = re.search(r"([123])\s*[=＝]", text)
    if not match:
        match = re.search(r"(?m)^\s*([123])\b", text)
    if not match:
        match = re.search(r"\b([123])\b", text)
    if not match:
        raise ValueError(f"无法从返回内容中解析意图编号：{text[:100]}")
    return int(match.group(1))


def identify_intent(user_input: str):
    reset_state = (
        INTENT_STATUS_UNKNOWN,
        gr.update(value="", interactive=False),
        gr.update(value=CONFIRM_BTN_LABEL, interactive=False),
    )
    if not user_input or not user_input.strip():
        gr.Warning("请先输入需要识别的文本")
        return reset_state
    if not LLM_API_KEY.strip():
        gr.Error("未配置 LLM_API_KEY，请先设置环境变量 SILICONFLOW_API_KEY")
        return reset_state
    try:
        prompt = INTENT_PROMPT_TEMPLATE.replace("{user_input}", user_input.strip())
        raw_text = _chat_completion([{"role": "user", "content": prompt}], temperature=0.1)
        intent = _parse_intent(raw_text)
    except Exception as exc:
        gr.Error(f"意图识别失败：{exc}")
        return reset_state

    if intent == 1:
        return (
            INTENT_STATUS_1,
            gr.update(value="", interactive=False),
            gr.update(value=CONFIRM_BTN_LABEL, interactive=False),
        )
    if intent == 2:
        return (
            INTENT_STATUS_2,
            gr.update(value=SYSTEM_PROMPT_DEFAULT, interactive=True),
            gr.update(value=CONFIRM_BTN_LABEL, interactive=True),
        )
    return (
        INTENT_STATUS_3,
        gr.update(value=SYSTEM_PROMPT_DEFAULT, interactive=False),
        gr.update(value=CONFIRM_BTN_LABEL, interactive=True),
    )


# ============================================================
# 板块 3：生图提示词生成
# ============================================================
def clean_generated_prompt(raw: str) -> str:
    if not raw:
        return ""
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    text = lines[0] if lines else ""
    text = re.sub(r"^(?:Prompt|提示词)\s*[:：]\s*", "", text, flags=re.IGNORECASE)
    return text.strip().strip('"').strip("'").strip()


def enforce_prompt_constraints(text: str) -> str:
    if not text:
        return text
    if not re.search(r"(Apple Design Philosophy|Apple[- ]style)", text, re.IGNORECASE):
        text = text + ", Apple style"
    if "1024x1024" not in text:
        text = text + ", 1024x1024"
    return text.strip()


def validate_generated_prompt(text: str) -> Tuple[bool, str]:
    if not text:
        return False, "提示词为空"
    if re.search(r"[\u4e00-\u9fff]", text):
        return False, "提示词包含中文字符，必须为纯英文"
    if not re.search(r"(Apple Design Philosophy|Apple[- ]style)", text, re.IGNORECASE):
        return False, "提示词未包含 Apple Design Philosophy 或 Apple style"
    if "1024x1024" not in text:
        return False, "提示词未包含 1024x1024"
    if len(text) < PROMPT_MIN_LEN:
        return False, f"提示词长度仅 {len(text)} 字符，需 {PROMPT_MIN_LEN}-{PROMPT_MAX_LEN} 字符"
    if len(text) > PROMPT_MAX_LEN:
        return False, f"提示词长度 {len(text)} 字符，需 {PROMPT_MIN_LEN}-{PROMPT_MAX_LEN} 字符"
    return True, ""


def generate_prompt(user_input: str, system_prompt: str, current_status: str):
    fail_state = (
        gr.update(value=""),
        gr.update(value=GENERATE_BTN_LABEL, interactive=False),
        gr.update(value=current_status or INTENT_STATUS_UNKNOWN),
    )
    if not user_input or not user_input.strip():
        gr.Warning("输入文本为空，无法生成提示词")
        return fail_state
    if not LLM_API_KEY.strip():
        gr.Error("未配置 LLM_API_KEY，请先设置环境变量 SILICONFLOW_API_KEY")
        return fail_state

    final_system_prompt = (system_prompt or "").strip() or SYSTEM_PROMPT_DEFAULT
    messages = [
        {"role": "system", "content": final_system_prompt},
        {"role": "user", "content": f"请为以下内容生成视觉提示词：\n\n{user_input.strip()}"},
    ]
    try:
        raw = _chat_completion(messages, temperature=PROMPT_GEN_TEMPERATURE, max_tokens=PROMPT_GEN_MAX_TOKENS)
    except Exception as exc:
        gr.Error(f"提示词生成失败：{exc}")
        return fail_state

    if not raw or not raw.strip():
        gr.Error("提示词生成失败：LLM 返回内容为空，请重试")
        return fail_state

    prompt = enforce_prompt_constraints(clean_generated_prompt(raw))
    ok, reason = validate_generated_prompt(prompt)
    if not ok:
        gr.Error(f"提示词生成失败：{reason}")
        return fail_state

    base_status = (current_status or INTENT_STATUS_UNKNOWN).strip()
    new_status = base_status if PROMPT_OK_SUFFIX in base_status else f"{base_status}；{PROMPT_OK_SUFFIX}"
    return (
        gr.update(value=prompt),
        gr.update(value=GENERATE_BTN_LABEL, interactive=True),
        gr.update(value=new_status),
    )


def on_generation_prompt_change(value: str):
    has_text = bool(value and value.strip())
    return gr.update(value=GENERATE_BTN_LABEL, interactive=has_text)


# ============================================================
# 板块 4：Nanobanana 文生图 / 图生图
# ============================================================
def image_to_base64_data_uri(ref_image_path: str) -> str:
    """
    图片预处理：将参考图转为 Nanobanana 可用的 data URI。
    - 打开为 PIL Image
    - 透明通道转为白色背景
    - 缩放到 1024x1024
    - PNG 编码 + Base64
    返回：data:image/png;base64,...
    """
    img = Image.open(ref_image_path)

    # 透明通道 -> 白色背景
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        bg.paste(rgba, mask=rgba.split()[-1])
        img = bg.convert("RGB")
    else:
        img = img.convert("RGB")

    # 缩放到 1024x1024
    img = img.resize((1024, 1024), Image.LANCZOS)

    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def extract_base64_from_response(content) -> Optional[str]:
    """
    从 API 返回的 content 中提取图片 Base64。
    兼容：结构化列表 [{type:image_url, image_url:{url:...}}] 与 Markdown ![](data:...)
    """
    if not content:
        return None

    # 1. 结构化列表
    if isinstance(content, list):
        for part in reversed(content):
            if isinstance(part, dict):
                img_field = part.get("image_url") or part.get("image") or part.get("output_image")
                if isinstance(img_field, dict):
                    url = img_field.get("url", "")
                    if url.startswith("data:image/") and "," in url:
                        return url.split(",", 1)[1].strip()
        # 列表里没有结构化图片，拼文本找 Markdown
        text_parts = [
            str(p.get("text", ""))
            for p in content
            if isinstance(p, dict) and p.get("type") in ["text", "input_text"]
        ]
        content_str = "".join(text_parts)
    else:
        content_str = str(content)

    # 2. Markdown 格式
    pattern = re.compile(r"!\[.*?\]\((data:image/[^;]+;base64,[^)]+)\)", re.IGNORECASE)
    match = pattern.search(content_str)
    if match:
        data_url = match.group(1)
        if "," in data_url:
            return data_url.split(",", 1)[1].strip()

    return None


def base64_to_image(base64_str: str) -> Optional[Image.Image]:
    try:
        image_bytes = base64.b64decode(base64_str)
        return Image.open(io.BytesIO(image_bytes))
    except Exception:
        return None


def generate_image(prompt: str, ref_image_path: Optional[str]) -> Tuple[Optional[Image.Image], str]:
    """
    调用 Nanobanana API 生成图片。
    返回：(PIL.Image 或 None, intent_status 文案)
    """
    # 前置校验
    if not prompt or not prompt.strip():
        return None, "提示词为空，无法生成图片"
    if not NANOBANANA_API_KEY.strip():
        return None, "未配置 NANOBANANA_API_KEY，请先设置环境变量 NANOBANANA_API_KEY"

    # 构建 messages：文生图 / 图生图
    if ref_image_path:
        try:
            image_base64 = image_to_base64_data_uri(ref_image_path)
        except Exception as exc:
            return None, f"参考图预处理失败：{exc}"
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_base64}},
                ],
            }
        ]
    else:
        messages = [{"role": "user", "content": prompt}]

    payload = {
        "model": NANOBANANA_MODEL,
        "messages": messages,
        "stream": False,
    }
    headers = {
        "Authorization": f"Bearer {NANOBANANA_API_KEY}",
        "Content-Type": "application/json",
    }

    # 调用 API
    try:
        response = requests.post(
            NANOBANANA_API_URL, headers=headers, json=payload, timeout=120
        )
    except requests.exceptions.Timeout:
        return None, "API 调用超时，请稍后重试"
    except Exception as exc:
        return None, f"API 调用失败：{exc}"

    if response.status_code != 200:
        return None, f"API 返回 HTTP {response.status_code}：{response.text[:200]}"

    # 解析响应
    try:
        result = response.json()
        content = result["choices"][0]["message"]["content"]
    except Exception:
        return None, "无法解析 API 返回结果，请重试"

    base64_str = extract_base64_from_response(content)
    if not base64_str:
        # 模型可能只返回了文本（拒绝生成）
        text_preview = str(content)[:100] if content else "(空)"
        return None, f"未在响应中找到图片，模型返回：{text_preview}"

    output_image = base64_to_image(base64_str)
    if not output_image:
        return None, "Base64 解析失败，无法生成图片"

    return output_image, "图片生成成功"


# ============================================================
# 界面构建
# ============================================================
with gr.Blocks(title="LLM 驱动的文生图全流程工具") as demo:
    gr.Markdown("# LLM 驱动的文生图全流程工具")

    with gr.Row():
        # 左侧 60%：提示词生成流程区
        with gr.Column(scale=6):
            input_text = gr.Textbox(
                label="输入文本（教程段落 / 绘图指令）",
                lines=6,
                placeholder="请输入需要配图的教程文本或直接绘图指令...",
            )
            identify_intent_btn = gr.Button(value="识别意图", interactive=True)
            intent_status = gr.Textbox(
                label="意图类型 / 处理状态",
                lines=2,
                interactive=False,
                value=INTENT_STATUS_UNKNOWN,
            )
            system_prompt = gr.Textbox(
                label="System Prompt（仅文章配图意图可编辑）",
                lines=4,
                interactive=False,
                placeholder="LLM 生成提示词的约束规则...",
            )
            confirm_prompt_btn = gr.Button(value=CONFIRM_BTN_LABEL, interactive=False)
            generation_prompt = gr.Textbox(
                label="生图提示词（可编辑）",
                lines=3,
                interactive=True,
                value="",
                placeholder="生成的英文生图提示词将显示在此，支持手动修改...",
            )

        # 右侧 40%：Nanobanana 生图功能区
        with gr.Column(scale=4):
            ref_image = gr.Image(
                label="参考图（可选，图生图）",
                type="filepath",
                height=300,
                interactive=True,
            )
            generate_btn = gr.Button(value=GENERATE_BTN_LABEL, interactive=False)
            result_image = gr.Image(
                label="生成结果",
                type="pil",
                height=300,
                interactive=False,
            )

    # 板块 2
    identify_intent_btn.click(
        fn=identify_intent,
        inputs=[input_text],
        outputs=[intent_status, system_prompt, confirm_prompt_btn],
        api_name="identify_intent",
    )

    # 板块 3
    confirm_prompt_btn.click(
        fn=generate_prompt,
        inputs=[input_text, system_prompt, intent_status],
        outputs=[generation_prompt, generate_btn, intent_status],
        api_name="generate_prompt",
    )
    generation_prompt.change(
        fn=on_generation_prompt_change,
        inputs=[generation_prompt],
        outputs=[generate_btn],
    )

    # 板块 4：生成图片 -> Nanobanana API -> 展示结果
    generate_btn.click(
        fn=generate_image,
        inputs=[generation_prompt, ref_image],
        outputs=[result_image, intent_status],
        api_name="generate_image",
    )

if __name__ == "__main__":
    demo.launch()
