import json

import anthropic
from pipeline.schema import LabeledSegment, Topic
from pipeline.segmenter import _fmt

MODEL = "claude-sonnet-5"

# strict=True 필수. 이걸 빼면 Sonnet 5가 topics를 리스트가 아니라
# {"topics": [...]} 객체 전체를 감싼 JSON 문자열로 반환한다 (9/9 재현).
# strict=True 또는 최상위 속성 추가로 해소되는 것까지 확인했고, 무엇이 방아쇠인지는
# 특정하지 못했다 (같은 모양인 EXTRACT_QA_TOOL은 strict 없이도 정상 동작한다).
# strict 모드는 maxItems를 거부하고 minItems는 0 또는 1만 허용하므로,
# 토픽 개수 상한은 SYSTEM_PROMPT로만 제약한다.
EXTRACT_TOOL = {
    "name": "extract_topics",
    "description": "Extract main topics from lecture core segments",
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
                            "description": "LabeledSegment index list that cover this topic",
                        },
                        "time_range": {
                            "type": "string",
                            "description": "HH:MM:SS-HH:MM:SS",
                        },
                    },
                    "required": ["title", "order", "source_chunks", "time_range"],
                },
            }
        },
        "required": ["topics"],
    },
}


def _unwrap(value, key: str):
    """tool_use 응답이 리스트 대신 JSON 문자열로 오는 경우를 방어한다.

    Sonnet 5는 최상위 배열 속성이 하나뿐인 스키마에서 {"topics": [...]} 객체
    전체를 문자열로 감싸 반환하는 경우가 있다. strict=True로 예방하지만,
    크론으로 무인 실행되는 파이프라인이라 안전망을 남겨둔다.
    """
    if isinstance(value, str):
        parsed = json.loads(value)
        return parsed[key] if isinstance(parsed, dict) and key in parsed else parsed
    return value


SYSTEM_PROMPT = """\
You are analyzing a university lecture transcript to extract its main topics.

Instructions:
- Identify 2 to 8 distinct topics covered in the lecture
- Each topic must be grounded in the provided segments (use source_chunks to cite segment indices)
- Assign a concise, descriptive title to each topic
- Order topics as they appear in the lecture (order starts at 1)
- time_range must span the earliest start to the latest end among source_chunks
- Do not invent topics not present in the text\
"""


def extract_structure(labeled_segments: list[LabeledSegment], api_key: str, out_usage: dict | None = None) -> list[Topic]:
    core = [s for s in labeled_segments if s.label == "lecture_core"]
    if not core:
        return []

    user_content = "Extract the main topics from these lecture segments:\n\n"
    for seg in core:
        user_content += f"[index={seg.index}, {_fmt(seg.start)}-{_fmt(seg.end)}]\n{seg.text}\n\n"

    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model=MODEL,
        # maxItems 상한이 사라지고 Sonnet 5가 토픽을 더 잘게 쪼개므로 1024에서 상향
        max_tokens=4096,
        system=SYSTEM_PROMPT,
        tools=[EXTRACT_TOOL],
        tool_choice={"type": "tool", "name": "extract_topics"},
        messages=[{"role": "user", "content": user_content}],
    )

    if out_usage is not None:
        out_usage["input_tokens"]  += response.usage.input_tokens
        out_usage["output_tokens"] += response.usage.output_tokens

    for block in response.content:
        if block.type == "tool_use" and block.name == "extract_topics":
            return [
                Topic(
                    title=t["title"],
                    order=t["order"],
                    source_chunks=t["source_chunks"],
                    time_range=t["time_range"],
                )
                for t in _unwrap(block.input["topics"], "topics")
            ]

    raise RuntimeError("Sonnet did not return extract_topics tool_use block")
