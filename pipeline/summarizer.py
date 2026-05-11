import anthropic
from pipeline.schema import (
    LabeledSegment, Topic,
    KeyPoint, TopicSummary, Notice, QASegment, SummaryResult,
)
from pipeline.segmenter import _fmt

MODEL = "claude-sonnet-4-6"

EMPHASIS_KEYWORDS = [
    "this is important", "key point", "remember", "critical",
    "make sure", "pay attention", "crucial", "essential",
]

SUMMARIZE_TOOL = {
    "name": "summarize_topic",
    "description": "Summarize one lecture topic with key points, emphasis, and concepts",
    "input_schema": {
        "type": "object",
        "properties": {
            "key_points": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "claim":        {"type": "string"},
                        "explanation":  {"type": "string"},
                        "source_quote": {"type": "string"},
                        "timestamp":    {"type": "string", "description": "HH:MM:SS"},
                    },
                    "required": ["claim", "explanation", "source_quote", "timestamp"],
                },
            },
            "important_emphasis":  {"type": "array", "items": {"type": "string"}},
            "concepts_introduced": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["key_points", "important_emphasis", "concepts_introduced"],
    },
}

EXTRACT_QA_TOOL = {
    "name": "extract_qa",
    "description": "Extract question-answer pairs from lecture Q&A segments",
    "input_schema": {
        "type": "object",
        "properties": {
            "qa_segments": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "question":  {"type": "string"},
                        "answer":    {"type": "string"},
                        "timestamp": {"type": "string", "description": "HH:MM:SS"},
                    },
                    "required": ["question", "answer", "timestamp"],
                },
            }
        },
        "required": ["qa_segments"],
    },
}

SUMMARIZE_SYSTEM = f"""\
You are summarizing a specific topic from a university lecture transcript.

Rules:
- Every key_point SHOULD include a source_quote (exact words from the transcript). \
If a direct quote is not available, set source_quote to empty string and still include the key_point.
- Every key_point MUST have a timestamp in HH:MM:SS format.
- important_emphasis: detect phrases like {', '.join(f'"{k}"' for k in EMPHASIS_KEYWORDS)}, \
or any concept repeated 3 or more times.
- concepts_introduced: list new technical terms or concepts mentioned for the first time.
- Do not invent content. Every claim must be traceable to the transcript.

Language rules (strictly follow):
- claim: English only — write as a concise English sentence matching the source material.
- explanation: Korean only — explain the claim in Korean for a Korean-speaking student.
- source_quote: English only — copy the professor's exact words from the transcript verbatim.
- timestamp: HH:MM:SS format.
- important_emphasis: Korean only — rephrase the emphasis point in Korean.
- concepts_introduced: English only — list the original English technical terms.\
"""

EXTRACT_QA_SYSTEM = """\
You are extracting student questions and professor answers from a lecture transcript.
For each Q&A exchange, record the question, the answer, and the timestamp (HH:MM:SS) \
where the question was asked.
Do not invent content.\
"""


def _build_segment_text(segments: list[LabeledSegment], indices: list[int]) -> str:
    idx_set = set(indices)
    parts = []
    for seg in segments:
        if seg.index in idx_set:
            parts.append(f"[{_fmt(seg.start)}]\n{seg.text}")
    return "\n\n".join(parts)


def _summarize_topic(
    client: anthropic.Anthropic,
    topic: Topic,
    labeled_segments: list[LabeledSegment],
    out_usage: dict | None = None,
) -> TopicSummary | None:
    content = _build_segment_text(labeled_segments, topic.source_chunks)
    word_count = len(content.split())
    base_msg = f"Topic: {topic.title}\n\nTranscript segments:\n\n{content}"

    if word_count < 50:
        print(f"  [skip] 토픽 '{topic.title}' 내용 부족 (words={word_count}) - 건너뜀")
        return None

    for attempt in range(3):
        retry_note = (
            "\n\nPrevious response was incomplete. key_points is required and must not be empty."
            if attempt > 0 else ""
        )
        user_msg = base_msg + retry_note

        response = client.messages.create(
            model=MODEL,
            max_tokens=4096,
            temperature=0.2,
            system=SUMMARIZE_SYSTEM,
            tools=[SUMMARIZE_TOOL],
            tool_choice={"type": "tool", "name": "summarize_topic"},
            messages=[{"role": "user", "content": user_msg}],
        )

        if out_usage is not None:
            out_usage["input_tokens"]  += response.usage.input_tokens
            out_usage["output_tokens"] += response.usage.output_tokens

        if response.stop_reason == "max_tokens":
            print(f"  [warn] 토픽 '{topic.title}' max_tokens(4096) 초과 - 8192로 재시도")
            response = client.messages.create(
                model=MODEL,
                max_tokens=8192,
                temperature=0.2,
                system=SUMMARIZE_SYSTEM,
                tools=[SUMMARIZE_TOOL],
                tool_choice={"type": "tool", "name": "summarize_topic"},
                messages=[{"role": "user", "content": user_msg}],
            )
            if out_usage is not None:
                out_usage["input_tokens"]  += response.usage.input_tokens
                out_usage["output_tokens"] += response.usage.output_tokens

        for block in response.content:
            if block.type == "tool_use" and block.name == "summarize_topic":
                d = block.input
                kps = d.get("key_points") or []
                if not kps:
                    print(f"  [warn] 토픽 '{topic.title}' key_points 누락 (attempt {attempt+1}/3)")
                    break
                return TopicSummary(
                    title=topic.title,
                    order=topic.order,
                    time_range=topic.time_range,
                    key_points=[
                        KeyPoint(
                            claim=kp["claim"],
                            explanation=kp["explanation"],
                            source_quote=kp["source_quote"],
                            timestamp=kp["timestamp"],
                        )
                        for kp in kps
                    ],
                    important_emphasis=d.get("important_emphasis", []),
                    concepts_introduced=d.get("concepts_introduced", []),
                )

    print(f"  [skip] 토픽 '{topic.title}' 3회 실패 - 결과에서 제외")
    return None


def _extract_qa(
    client: anthropic.Anthropic,
    qa_segs: list[LabeledSegment],
    out_usage: dict | None = None,
) -> list[QASegment]:
    if not qa_segs:
        return []

    content = "\n\n".join(f"[{_fmt(s.start)}]\n{s.text}" for s in qa_segs)
    response = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        temperature=0.2,
        system=EXTRACT_QA_SYSTEM,
        tools=[EXTRACT_QA_TOOL],
        tool_choice={"type": "tool", "name": "extract_qa"},
        messages=[{"role": "user", "content": content}],
    )

    if out_usage is not None:
        out_usage["input_tokens"]  += response.usage.input_tokens
        out_usage["output_tokens"] += response.usage.output_tokens

    for block in response.content:
        if block.type == "tool_use" and block.name == "extract_qa":
            return [
                QASegment(
                    question=q["question"],
                    answer=q["answer"],
                    timestamp=q["timestamp"],
                )
                for q in block.input["qa_segments"]
            ]

    raise RuntimeError("Sonnet did not return extract_qa block")


def build_summary(
    topics: list[Topic],
    labeled_segments: list[LabeledSegment],
    lecture_meta: dict,
    api_key: str,
    out_usage: dict | None = None,
) -> SummaryResult:
    client = anthropic.Anthropic(api_key=api_key)

    # 토픽별 요약 (토픽당 Sonnet 1회)
    topic_summaries = []
    for i, topic in enumerate(topics):
        print(f"  토픽 {i+1}/{len(topics)} 요약 중: {topic.title}")
        result = _summarize_topic(client, topic, labeled_segments, out_usage)
        if result is not None:
            topic_summaries.append(result)
    print(f"  요약 완료: {len(topic_summaries)}/{len(topics)} 토픽")

    # QA 추출 (qa 라벨 세그먼트 전체 → Sonnet 1회)
    qa_segs = [s for s in labeled_segments if s.label == "qa"]
    print(f"  QA 추출 중 ({len(qa_segs)}개 세그먼트)...")
    qa_segments = _extract_qa(client, qa_segs, out_usage)

    # 공지 수집 (classifier에서 이미 추출된 notices 그대로 사용)
    announcements = []
    for seg in labeled_segments:
        if seg.label == "announcement":
            for n in seg.notices:
                announcements.append(Notice(
                    type=n.get("type", "notice"),
                    content=n.get("content", ""),
                    deadline=n.get("deadline"),
                    source_quote="",
                    timestamp=_fmt(seg.start),
                ))

    # 전체 transcript (순서대로 이어붙임)
    full_transcript = "\n\n".join(
        s.text for s in sorted(labeled_segments, key=lambda x: x.index)
    )

    return SummaryResult(
        lecture_meta=lecture_meta,
        announcements=announcements,
        topics=topic_summaries,
        qa_segments=qa_segments,
        full_transcript=full_transcript,
    )
