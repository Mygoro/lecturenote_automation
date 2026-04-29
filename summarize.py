"""
Claude API로 강의 텍스트를 3단계로 처리하는 모듈

[1단계] Haiku  - 텍스트 정제 (노이즈 제거, 실질 내용 보존)
[2단계] Haiku  - 중요 공지 추출 (과제/시험/출결/공지 → JSON)
[3단계] Sonnet - 강의 내용 요약 (정제된 텍스트 기반, 고정 형식)
"""

import json
from anthropic import Anthropic

# ────────────────────────────────────────────
# 프롬프트
# ────────────────────────────────────────────

CLEAN_PROMPT = """\
You are cleaning up a raw lecture transcript produced by speech-to-text software.

Tasks:
1. Remove filler words and meaningless repetitions (um, uh, repeated phrases due to STT errors).
2. Fix obvious STT misrecognitions where the intended word is clear from context.
3. Preserve ALL substantive content — do not summarize, shorten, or omit any actual lecture material.
4. Keep the original order and flow of the lecture.
5. Output plain text only. No headings, no bullet points.

Raw transcript:
"""

NOTICES_PROMPT = """\
You are extracting important announcements from a lecture transcript.

Extract ONLY items that fall into these categories:
- Assignment / project (include deadline if mentioned)
- Exam or quiz (include date/week if mentioned)
- Attendance policy or change
- Course schedule change
- Any other urgent notice from the instructor

Return a JSON array. Each item must have:
  "type": one of "assignment" | "exam" | "attendance" | "schedule_change" | "notice"
  "content": concise description in Korean (1–2 sentences)
  "deadline": date or week string if mentioned, otherwise null

If nothing is found, return an empty array: []

Lecture transcript:
"""

SUMMARY_PROMPT = """\
Act as a highly experienced academic note-taker and content structurer, specializing in transforming raw lecture transcripts into clear, student-friendly study guides for university undergraduates. Your expertise lies in distilling complex information into easily digestible, highly organized formats without sacrificing critical detail.

Your primary goal is to take a provided raw lecture transcript (강의 스크립트) and transform it into a structured, summarized, and exceptionally readable set of detailed notes. The final output must enhance readability for undergraduates, ensure *no critical information is omitted*, and strictly adhere to the specified structural and stylistic requirements.

**C — Context:**
You will be provided with a complete, raw lecture transcript, which may contain conversational filler, repetitions, or unorganized thoughts common in spoken discourse. The domain is general university-level academic subjects. Your task is to process this raw text and produce a polished, high-quality study aid designed for undergraduate students.

**O — Objective:**
Produce a comprehensive, structured, and detailed summary of the provided lecture transcript. The deliverable must be a standalone study guide, approximately 3-5 pages in length (equivalent to 1500-2500 words), that allows an undergraduate student to efficiently review and understand the lecture content without needing to refer back to the original transcript. Success means the summary is clear, complete, logically organized, and highly accessible, making complex topics easy to grasp.

**S — Style:**
Write in the style of meticulously organized, insightful, and student-friendly study notes crafted by a top-tier university student. The language should be precise yet approachable, focusing on clarity and retention.

**T — Tone:**
The tone should be consistently conversational and friendly, yet authoritative and informative. It should feel like a helpful peer explaining concepts, rather than a dry academic text. Use engaging language that encourages understanding.

**A — Audience:**
The target audience is undergraduate university students. They are intelligent but benefit greatly from well-structured information, clear explanations, and an approachable tone to aid their learning and exam preparation.

**R — Response Format:**
The summary must be structured as follows, presented entirely in Korean:

* **Main Headings (H1):** For major lecture topics.
* **Sub-Headings (H2):** For sub-topics within each major section.
* **Detailed Points:** Use a combination of bulleted lists (글머리 기호 •) and numbered lists (1., 2., 3.) for all key information, explanations, and sequential steps. Avoid long paragraphs; break information into concise, digestible points.
* **Key Examples/Case Studies/Analogies:** Integrate only the *most critical* examples, case studies, or analogies directly from the lecture transcript within the relevant sections to illustrate complex points. Clearly label these for easy identification (e.g., "예시:", "사례:", "비유:").
* **Technical Jargon:** Retain the original technical terms (전문 용어) as much as possible within the main text. However, *all* retained technical terms that might be unfamiliar to an undergraduate must be explained in a dedicated "용어 해설 (Glossary)" section at the *very end* of the summary. Do not explain them inline.

**Constraint Injection:**

1. **Do NOT omit any critical information or core concepts** from the original lecture transcript. Every significant point must be captured.
2. **Every major section must begin with a clear, concise heading (H1)**, followed by appropriate sub-headings (H2).
3. **Avoid generic introductory or concluding phrases.** Dive directly into the structured content.
4. **Ensure all explanations for technical jargon are grouped exclusively at the very end** under the "용어 해설 (Glossary)" section.
5. **Maintain a consistent conversational and friendly tone** throughout, avoiding overly academic or dry language.

**Chain-of-Thought:**
First, meticulously read through the entire lecture transcript to identify the overarching themes, main topics, and all supporting sub-topics. Second, extract every key detail, definition, example, and argument, ensuring nothing essential is missed. Third, logically structure this extracted information using a clear hierarchy of headings and sub-headings, preparing for bulleted and numbered lists. Fourth, rephrase the extracted content into concise, clear, and engaging points, adopting the specified conversational and friendly tone while preserving factual accuracy. Fifth, identify all technical terms and complex jargon, noting them for inclusion in the glossary. Finally, compile the comprehensive summary, integrate key examples, and create the "용어 해설" section at the end.

**Output Gating:**
Before delivering your final summary, meticulously verify the following:

1. **Completeness:** Is every core concept and critical detail from the original transcript present in the summary?
2. **Structure:** Does the summary effectively utilize H1, H2 headings, bulleted lists, and numbered lists as specified?
3. **Tone & Style:** Is the tone consistently conversational and friendly, and does the style reflect high-quality student notes?
4. **Examples:** Are key examples, case studies, or analogies integrated appropriately to clarify concepts?
5. **Jargon Handling:** Are all technical terms explained *only* in the dedicated "용어 해설 (Glossary)" section at the end?
6. **Length:** Is the summary approximately 3-5 pages in length (1500-2500 words)?

Begin processing the lecture transcript now.

Cleaned transcript:
"""


# ────────────────────────────────────────────
# 단계별 처리 함수
# ────────────────────────────────────────────

def _call(client: Anthropic, model: str, prompt: str, text: str, max_tokens: int) -> tuple[str, dict]:
    msg = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt + text}],
    )
    usage = {"input_tokens": msg.usage.input_tokens, "output_tokens": msg.usage.output_tokens}
    return msg.content[0].text, usage


def process_transcript(api_key: str, raw_transcript: str) -> tuple[str, list[dict], str, dict]:
    """
    3단계 파이프라인 실행
    반환: (정제된 텍스트, 공지 목록, 요약 텍스트, 모델별 토큰 사용량)
    """
    client = Anthropic(api_key=api_key)
    usage_by_model = {
        "haiku":  {"input_tokens": 0, "output_tokens": 0},
        "sonnet": {"input_tokens": 0, "output_tokens": 0},
    }

    # 1단계: 텍스트 정제 (Haiku)
    print("   [1/3] 텍스트 정제 중 (Haiku)...")
    cleaned, usage = _call(client, "claude-haiku-4-5-20251001", CLEAN_PROMPT, raw_transcript, max_tokens=16000)
    usage_by_model["haiku"]["input_tokens"]  += usage["input_tokens"]
    usage_by_model["haiku"]["output_tokens"] += usage["output_tokens"]

    # 2단계: 공지 추출 (Haiku)
    print("   [2/3] 중요 공지 추출 중 (Haiku)...")
    notices_raw, usage = _call(client, "claude-haiku-4-5-20251001", NOTICES_PROMPT, cleaned, max_tokens=2048)
    usage_by_model["haiku"]["input_tokens"]  += usage["input_tokens"]
    usage_by_model["haiku"]["output_tokens"] += usage["output_tokens"]

    notices = []
    try:
        text = notices_raw.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        notices = json.loads(text.strip())
        if not isinstance(notices, list):
            notices = []
    except Exception:
        notices = []

    # 3단계: 요약 (Sonnet)
    print("   [3/3] 강의 내용 요약 중 (Sonnet)...")
    summary, usage = _call(client, "claude-sonnet-4-6", SUMMARY_PROMPT, cleaned, max_tokens=8192)
    usage_by_model["sonnet"]["input_tokens"]  += usage["input_tokens"]
    usage_by_model["sonnet"]["output_tokens"] += usage["output_tokens"]

    return cleaned, notices, summary, usage_by_model
