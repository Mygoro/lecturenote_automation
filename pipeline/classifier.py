import anthropic
from pipeline.schema import Segment, LabeledSegment

MODEL = "claude-haiku-4-5-20251001"
TOKEN_BUDGET = 4000   # 배치당 최대 추정 토큰 수 (len(text)//3 기준)

CLASSIFY_TOOL = {
    "name": "classify_segments",
    "description": "Classify lecture transcript segments and extract announcements",
    "input_schema": {
        "type": "object",
        "properties": {
            "classifications": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "index": {"type": "integer"},
                        "label": {
                            "type": "string",
                            "enum": ["lecture_core", "announcement", "qa", "example", "smalltalk"],
                        },
                        "notices": {
                            "type": "array",
                            "description": "Populate only when label is 'announcement', otherwise []",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "type": {
                                        "type": "string",
                                        "enum": ["assignment", "exam", "attendance", "schedule_change", "notice"],
                                    },
                                    "content": {"type": "string"},
                                    "deadline": {"type": ["string", "null"]},
                                },
                                "required": ["type", "content", "deadline"],
                            },
                        },
                    },
                    "required": ["index", "label", "notices"],
                },
            }
        },
        "required": ["classifications"],
    },
}

SYSTEM_PROMPT = """\
You are classifying segments of a recorded university lecture.

Labels:
- lecture_core: main lecture content, explanations, theory
- announcement: a specific assignment, exam, attendance policy change, or schedule change \
with a concrete deadline or date
- qa: student questions and professor answers
- example: worked examples, case studies, demos
- smalltalk: greetings, off-topic, filler, pauses

Classify as 'announcement' ONLY when ALL of the following apply:
  - An assignment with an explicit deadline, OR an exam/quiz with a specific date, \
OR an attendance policy change, OR a class schedule change
  - The deadline or date is clearly stated (week number, day, or date)

Do NOT classify as 'announcement':
  - Behavioral instructions without a deadline ("conduct research", "find sources", \
"start early", "submit before groups are assigned")
  - Procedural guidance or process descriptions
  - Sub-items already covered by another announcement in the same or adjacent segment

Duplicate prevention:
  - If the same assignment appears across multiple segments, extract it ONCE — \
from the segment that contains the most specific information (deadline, grade weight).
  - Ignore repeated mentions in other segments.

For each segment assign exactly one label.
For 'announcement' segments only, extract notices (type, content, deadline).
For all other labels set notices to [].
Do not invent content. Base decisions solely on the provided text.\
"""


def _make_batches(segments: list[Segment]) -> list[list[Segment]]:
    """누적 토큰 예산 기반으로 세그먼트를 배치로 묶음"""
    batches: list[list[Segment]] = []
    current: list[Segment] = []
    current_tokens = 0
    for seg in segments:
        estimated = len(seg.text) // 3
        if current and current_tokens + estimated > TOKEN_BUDGET:
            batches.append(current)
            current = [seg]
            current_tokens = estimated
        else:
            current.append(seg)
            current_tokens += estimated
    if current:
        batches.append(current)
    return batches


def _call_haiku(client: anthropic.Anthropic, batch: list[Segment], out_usage: dict | None = None) -> list[dict]:
    user_content = "Classify each segment:\n\n"
    for seg in batch:
        user_content += f"[index={seg.index}]\n{seg.text}\n\n"

    response = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        temperature=0.2,
        system=SYSTEM_PROMPT,
        tools=[CLASSIFY_TOOL],
        tool_choice={"type": "tool", "name": "classify_segments"},
        messages=[{"role": "user", "content": user_content}],
    )

    if out_usage is not None:
        out_usage["input_tokens"]  += response.usage.input_tokens
        out_usage["output_tokens"] += response.usage.output_tokens

    for block in response.content:
        if block.type == "tool_use" and block.name == "classify_segments":
            return block.input["classifications"]

    raise RuntimeError("Haiku did not return classify_segments tool_use block")


def classify_chunks(segments: list[Segment], api_key: str, out_usage: dict | None = None) -> list[LabeledSegment]:
    client = anthropic.Anthropic(api_key=api_key)
    batches = _make_batches(segments)

    results: dict[int, dict] = {}
    for i, batch in enumerate(batches):
        print(f"  배치 {i+1}/{len(batches)} 분류 중 ({len(batch)}개 세그먼트)...")
        for c in _call_haiku(client, batch, out_usage):
            results[c["index"]] = c

    labeled = []
    for seg in segments:
        c = results.get(seg.index, {"label": "lecture_core", "notices": []})
        notices = c["notices"] if c["label"] == "announcement" else []
        labeled.append(LabeledSegment(
            index=seg.index,
            start=seg.start,
            end=seg.end,
            text=seg.text,
            label=c["label"],
            notices=notices,
        ))

    return labeled
