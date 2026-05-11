import anthropic
from pipeline.schema import LabeledSegment, Topic
from pipeline.segmenter import _fmt

MODEL = "claude-sonnet-4-6"

EXTRACT_TOOL = {
    "name": "extract_topics",
    "description": "Extract main topics from lecture core segments",
    "input_schema": {
        "type": "object",
        "properties": {
            "topics": {
                "type": "array",
                "minItems": 2,
                "maxItems": 8,
                "items": {
                    "type": "object",
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
        max_tokens=1024,
        temperature=0.2,
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
                for t in block.input["topics"]
            ]

    raise RuntimeError("Sonnet did not return extract_topics tool_use block")
