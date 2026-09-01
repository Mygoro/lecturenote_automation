import json

import anthropic
from pipeline.schema import LabeledSegment, Topic, Structure
from pipeline.segmenter import _fmt

MODEL = "claude-sonnet-5"

# strict=True 필수. 이걸 빼면 Sonnet 5가 topics를 리스트가 아니라
# {"topics": [...]} 객체 전체를 감싼 JSON 문자열로 반환한다 (9/9 재현).
# strict 모드는 maxItems를 거부하고 minItems는 0 또는 1만 허용하므로,
# 토픽 개수 상한은 SYSTEM_PROMPT로만 제약한다.
EXTRACT_TOOL = {
    "name": "extract_topics",
    "description": "Extract mutually exclusive topics, action items, and uncovered facts",
    "strict": True,
    "input_schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "topics": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "title": {"type": "string"},
                        "order": {"type": "integer"},
                        "source_chunks": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "description": "Segment indices for this topic. Must not overlap other topics.",
                        },
                        "time_range": {
                            "type": "string",
                            "description": "HH:MM:SS-HH:MM:SS",
                        },
                    },
                    "required": ["title", "order", "source_chunks", "time_range"],
                },
            },
            "action_items": {
                "type": "array",
                "items": {"type": "string"},
                "description": "학생이 실제로 수행해야 할 행동만. 한국어 한 줄씩.",
            },
            "uncovered_points": {
                "type": "array",
                "items": {"type": "string"},
                "description": "어느 토픽 source_chunks에도 담기지 않은 중요 사실. 한국어 한 줄씩.",
            },
        },
        "required": ["topics", "action_items", "uncovered_points"],
    },
}


def _unwrap(value, key: str):
    """tool_use 응답이 리스트 대신 JSON 문자열로 오는 경우를 방어한다.

    Sonnet 5는 특정 스키마에서 {"topics": [...]} 객체 전체를 문자열로 감싸
    반환하는 경우가 있다. strict=True로 예방하지만, 크론으로 무인 실행되는
    파이프라인이라 안전망을 남겨둔다.
    """
    if isinstance(value, str):
        parsed = json.loads(value)
        return parsed[key] if isinstance(parsed, dict) and key in parsed else parsed
    return value


SYSTEM_PROMPT = (
    "You are analyzing a university lecture transcript.\n"
    "\n"
    "Produce three things in one pass.\n"
    "\n"
    "1) topics - the lecture's subject matter\n"
    "   - You are told how many segments exist. NEVER produce more topics than\n"
    "     there are segments - every topic needs at least one segment of its own.\n"
    "     With 1-2 segments produce exactly ONE topic. Use fewer topics for a\n"
    "     short lecture; never pad the count. Maximum 8.\n"
    "   - Title each topic after what it actually covers. If one topic must carry\n"
    "     the whole lecture, title it after the dominant substance, not after the\n"
    "     opening remarks.\n"
    "   - Topics MUST be mutually exclusive. A segment index may appear in the\n"
    "     source_chunks of exactly ONE topic. Never repeat the same material under\n"
    "     two titles.\n"
    "   - Build topics from lecture_core and example segments only.\n"
    "   - Give each a concise descriptive title, ordered as they appear (order starts at 1).\n"
    "   - time_range spans the earliest start to the latest end among source_chunks.\n"
    "\n"
    "2) action_items - what the student must actually DO\n"
    "   - Concrete actions only: watch the week's videos, verify the online\n"
    "     attendance record, submit the 4 practice problems, bring equipment.\n"
    "   - Korean, one line each, imperative. Include numbers and deadlines when stated.\n"
    "   - Draw from every segment including announcements. Empty list if none.\n"
    "\n"
    "3) uncovered_points - the safety net\n"
    "   - After assigning source_chunks, scan the segments again and list every\n"
    "     concrete fact a student would be graded on or penalised for that is NOT\n"
    "     inside any topic's source_chunks.\n"
    "   - Especially: point allocations, attendance and absence rules, exam format,\n"
    "     submission rules, contact channels, seating rules.\n"
    "   - Korean, one line each. Empty list only if genuinely nothing was left out.\n"
    "\n"
    "Accuracy: do not invent topics or facts not present in the text. The transcript\n"
    "is auto-generated and may be garbled; never guess what a broken sentence meant.\n"
)


def _dedupe_chunks(topics: list[Topic]) -> list[Topic]:
    """토픽 간 source_chunks 중복 제거.

    프롬프트로 배타성을 요구하지만 모델이 어길 수 있어 코드로 한 번 더 강제한다.
    먼저 등장한(order가 앞선) 토픽이 해당 청크를 가져간다.
    """
    seen: set[int] = set()
    out: list[Topic] = []
    for t in sorted(topics, key=lambda x: x.order):
        keep = [c for c in t.source_chunks if c not in seen]
        dropped = len(t.source_chunks) - len(keep)
        if dropped:
            print(f"  [dedup] 토픽 '{t.title}' 중복 청크 {dropped}개 제거")
        if not keep:
            print(f"  [dedup] 토픽 '{t.title}' 고유 청크 없음 - 토픽 제외")
            continue
        seen.update(keep)
        t.source_chunks = keep
        out.append(t)
    for i, t in enumerate(out, start=1):
        t.order = i
    return out


def extract_structure(labeled_segments: list[LabeledSegment], api_key: str,
                      out_usage: dict | None = None) -> Structure:
    usable = [s for s in labeled_segments if s.label != "smalltalk"]
    if not usable:
        return Structure(topics=[])

    user_content = (f"Analyze these {len(usable)} lecture segments. "
                    f"Produce at most {min(len(usable), 8)} topics.\n\n")
    for seg in usable:
        user_content += (f"[index={seg.index}, label={seg.label}, "
                         f"{_fmt(seg.start)}-{_fmt(seg.end)}]\n{seg.text}\n\n")

    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model=MODEL,
        max_tokens=4096,
        system=SYSTEM_PROMPT,
        tools=[EXTRACT_TOOL],
        tool_choice={"type": "tool", "name": "extract_topics"},
        messages=[{"role": "user", "content": user_content}],
    )

    if out_usage is not None:
        out_usage["input_tokens"]  += response.usage.input_tokens
        out_usage["output_tokens"] += response.usage.output_tokens

    if response.stop_reason == "max_tokens":
        print("  [warn] 토픽 추출이 max_tokens에 걸렸습니다")

    for block in response.content:
        if block.type == "tool_use" and block.name == "extract_topics":
            d = block.input
            topics = [
                Topic(
                    title=t["title"],
                    order=t["order"],
                    source_chunks=t["source_chunks"],
                    time_range=t["time_range"],
                )
                for t in _unwrap(d["topics"], "topics")
            ]
            return Structure(
                topics=_dedupe_chunks(topics),
                action_items=list(_unwrap(d.get("action_items", []), "action_items")),
                uncovered_points=list(_unwrap(d.get("uncovered_points", []), "uncovered_points")),
            )

    raise RuntimeError("Sonnet did not return extract_topics tool_use block")
