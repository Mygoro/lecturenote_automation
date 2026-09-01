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
                                        "enum": ["assignment", "exam", "attendance", "grading",
                                                 "logistics", "schedule_change", "notice"],
                                    },
                                    "content": {"type": "string"},
                                    "deadline": {"type": ["string", "null"]},
                                    "source_quote": {
                                        "type": "string",
                                        "description": "Verbatim supporting sentence(s) copied from the transcript",
                                    },
                                },
                                "required": ["type", "content", "deadline", "source_quote"],
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

SYSTEM_PROMPT = (
    "You are classifying segments of a recorded university lecture.\n"
    "\n"
    "Labels:\n"
    "- lecture_core: main lecture content, explanations, theory\n"
    "- announcement: course administration the student must act on or be graded by\n"
    "- qa: student questions and professor answers\n"
    "- example: worked examples, case studies, demos\n"
    "- smalltalk: greetings, off-topic, filler, pauses\n"
    "\n"
    "Classify as 'announcement' when the segment states ANY of the following.\n"
    "An explicit deadline is NOT required:\n"
    "  - assignment / exam / quiz: scope, format, date, weight\n"
    "  - grading: point allocation, weights, absolute vs relative grading, cutoffs\n"
    "  - attendance: how attendance is recorded, penalties, how many absences are\n"
    "    allowed, excused-absence rules and required paperwork, seating rules\n"
    "  - submission: what to submit, file format, naming, where to upload\n"
    "  - logistics: class channel (group chat, LMS), where materials are posted,\n"
    "    equipment to bring, room or schedule changes\n"
    "\n"
    "Do NOT classify as 'announcement':\n"
    "  - Motivational talk or study advice with no rule attached\n"
    "  - Restating subject matter already covered as lecture content\n"
    "\n"
    "Duplicate prevention:\n"
    "  - If the same rule appears in several segments, extract it ONCE, from the\n"
    "    segment carrying the most specific information (number, deadline, weight).\n"
    "\n"
    "Assign exactly one label per segment.\n"
    "For 'announcement' segments only, extract notices\n"
    "(type, content, deadline, source_quote):\n"
    "  - content: Korean, 1-2 sentences. Keep every number exactly as stated\n"
    "    (points, counts, weeks, how many absences are allowed).\n"
    "  - deadline: null when none is stated. Do not invent one.\n"
    "  - source_quote: copy the supporting sentence(s) from the transcript VERBATIM.\n"
    "    Never paraphrase, clean up, or translate it. It is used to verify you.\n"
    "For all other labels set notices to [].\n"
    "\n"
    "Accuracy rules - the transcript is auto-generated and contains recognition errors:\n"
    "  - Do not invent content. Base decisions solely on the provided text.\n"
    "  - If a sentence is garbled or self-contradictory, do NOT guess what it must\n"
    "    have meant. State only what is certain and append '(원문 불명확)' to content.\n"
    "  - NEVER flip the meaning of a rule to make it sound sensible. Attendance and\n"
    "    grading rules must be reported exactly as stated, even when they read oddly.\n"
)


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
        max_tokens=4096,
        temperature=0.2,
        system=SYSTEM_PROMPT,
        tools=[CLASSIFY_TOOL],
        tool_choice={"type": "tool", "name": "classify_segments"},
        messages=[{"role": "user", "content": user_content}],
    )

    if out_usage is not None:
        out_usage["input_tokens"]  += response.usage.input_tokens
        out_usage["output_tokens"] += response.usage.output_tokens

    if response.stop_reason == "max_tokens":
        print("  [warn] 분류 응답이 max_tokens에 걸렸습니다 - 일부 세그먼트가 누락될 수 있습니다")

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

    missing = [s.index for s in segments if s.index not in results]
    if missing:
        print(f"  [warn] 분류 결과 누락 {len(missing)}건 -> lecture_core로 폴백: {missing}")

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
