import json
import re

import anthropic
from pipeline.schema import (
    LabeledSegment, Topic, Structure,
    KeyPoint, TopicSummary, Notice, QASegment, SummaryResult,
)
from pipeline.segmenter import _fmt

MODEL = "claude-sonnet-5"

SUMMARIZE_TOOL = {
    "name": "summarize_topic",
    "description": "Summarize one lecture topic with key points and concepts",
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
                    },
                    "required": ["claim", "explanation", "source_quote"],
                },
            },
            "concepts_introduced": {"type": "array", "items": {"type": "string"}},
            "summary_oneliner": {
                "type": "string",
                "description": "이 토픽의 핵심을 한국어 한 줄로 (30자 이내)",
            },
            "keywords_with_brief": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "keyword": {"type": "string"},
                        "brief":   {"type": "string"},
                    },
                    "required": ["keyword", "brief"],
                },
                "description": "핵심 키워드 3~5개, 각각 영어 원문 keyword + 한국어 brief (10~20자)",
            },
        },
        "required": ["key_points", "concepts_introduced", "summary_oneliner", "keywords_with_brief"],
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
                        "question": {"type": "string"},
                        "answer":   {"type": "string"},
                    },
                    "required": ["question", "answer"],
                },
            }
        },
        "required": ["qa_segments"],
    },
}

SUMMARIZE_SYSTEM = (
    "You are summarizing one topic from a university lecture transcript for a\n"
    "Korean student who will revise from your notes alone. The audio is NOT\n"
    "provided alongside, so the notes must stand on their own.\n"
    "\n"
    "Coverage - do not drop things:\n"
    "- Every concrete fact in the provided segments that a student could be tested\n"
    "  on or graded by must appear in some key_point. Numbers, rules, requirements,\n"
    "  definitions and named concepts all count.\n"
    "- Before you finish, re-read the segments and check nothing was left out.\n"
    "  Prefer merging related facts into one richer key_point over dropping them.\n"
    "\n"
    "Brevity - do not pad:\n"
    "- Do NOT restate the same fact as two key_points.\n"
    "- The summary must be substantially SHORTER than the transcript it came from.\n"
    "  A short segment deserves few key_points.\n"
    "\n"
    "Quoting:\n"
    "- source_quote must be copied from the transcript VERBATIM - exact characters,\n"
    "  no cleanup, no ellipsis, no translation. It is checked against the transcript\n"
    "  automatically and silently discarded if it does not match.\n"
    "- Quote a short contiguous span, not a stitched-together passage.\n"
    "- If no single span supports the point, set source_quote to an empty string.\n"
    "\n"
    "Accuracy - the transcript is auto-generated and contains recognition errors:\n"
    "- Do not invent content. Every claim must be traceable to the transcript.\n"
    "- If a sentence is garbled or self-contradictory, do NOT guess what it must\n"
    "  have meant, and NEVER flip its meaning to make it sound sensible. State only\n"
    "  what is certain and append '(원문 불명확)' to the explanation.\n"
    "- Rules about attendance, grading and deadlines are the highest risk. Report\n"
    "  them exactly as stated even when they read oddly.\n"
    "\n"
    "Terminology repair - explanation text only, never the quote:\n"
    "- Speech-to-text mangles technical terms. In explanation, concepts_introduced\n"
    "  and keywords, write the CORRECT term (e.g. 리스트/튜플/딕셔너리, 반복문,\n"
    "  파일 입출력, 바이브 코딩, 런어스). Leave source_quote exactly as transcribed.\n"
    "- Only repair a term when the intended word is unambiguous. Otherwise keep it\n"
    "  as-is and append '(원문 불명확)'.\n"
    "\n"
    "Language:\n"
    "- claim: English, one concise sentence.\n"
    "- explanation: Korean. This is the sentence the student actually reads.\n"
    "- source_quote: verbatim transcript text, whatever language it is in.\n"
    "- concepts_introduced: the correct term, original language.\n"
    "- summary_oneliner: Korean, 30자 이내.\n"
    "- keywords_with_brief: keyword는 원문 용어, brief는 한국어 10~20자.\n"
)

EXTRACT_QA_SYSTEM = (
    "You are extracting student questions and professor answers from a lecture\n"
    "transcript. Record the question and the answer in Korean.\n"
    "Do not invent content. If the exchange is not clearly a question and an\n"
    "answer, omit it.\n"
)


def _norm(text: str) -> str:
    """인용 대조용 정규화: 공백을 모두 제거해 줄바꿈/띄어쓰기 차이를 흡수한다."""
    return re.sub(r"\s+", "", text or "")


def _build_segment_text(segments: list[LabeledSegment], indices: list[int]) -> str:
    idx_set = set(indices)
    parts = []
    for seg in segments:
        if seg.index in idx_set:
            parts.append(seg.text)
    return "\n\n".join(parts)


def _unwrap(value, key: str):
    if isinstance(value, str):
        parsed = json.loads(value)
        return parsed[key] if isinstance(parsed, dict) and key in parsed else parsed
    return value


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
                kps = _unwrap(d.get("key_points") or [], "key_points")
                if not kps:
                    print(f"  [warn] 토픽 '{topic.title}' key_points 누락 (attempt {attempt+1}/3)")
                    break
                return TopicSummary(
                    title=topic.title,
                    order=topic.order,
                    time_range=topic.time_range,
                    key_points=[
                        KeyPoint(
                            claim=kp.get("claim", ""),
                            explanation=kp.get("explanation", ""),
                            source_quote=kp.get("source_quote", ""),
                        )
                        for kp in kps
                    ],
                    concepts_introduced=list(_unwrap(d.get("concepts_introduced", []), "concepts_introduced")),
                    summary_oneliner=d.get("summary_oneliner", ""),
                    keywords_with_brief=list(_unwrap(d.get("keywords_with_brief", []), "keywords_with_brief")),
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

    content = "\n\n".join(s.text for s in qa_segs)
    response = client.messages.create(
        model=MODEL,
        max_tokens=4096,
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
                QASegment(question=q.get("question", ""), answer=q.get("answer", ""))
                for q in _unwrap(block.input.get("qa_segments", []), "qa_segments")
            ]

    raise RuntimeError("Sonnet did not return extract_qa block")


def _verify_quotes(topics: list[TopicSummary], announcements: list[Notice], transcript: str) -> None:
    """인용이 전사문에 실제로 존재하는지 대조. 없으면 비운다.

    타임스탬프를 버린 대신 이 대조가 유일한 검증 수단이므로 코드로 강제한다.
    key_point 자체는 남기고 근거 없는 인용만 제거한다.
    """
    tnorm = _norm(transcript)
    kept = dropped = 0
    for t in topics:
        for kp in t.key_points:
            if not kp.source_quote:
                continue
            if _norm(kp.source_quote) in tnorm:
                kept += 1
            else:
                dropped += 1
                kp.source_quote = ""
    for n in announcements:
        if not n.source_quote:
            continue
        if _norm(n.source_quote) in tnorm:
            kept += 1
        else:
            dropped += 1
            n.source_quote = ""
    total = kept + dropped
    if total:
        print(f"  인용 대조: {kept}/{total} 원문 일치"
              + (f" - 불일치 {dropped}건 제거" if dropped else ""))


def _build_glossary(topics: list[TopicSummary]) -> list[dict]:
    """토픽별로 흩어진 키워드/개념을 중복 제거해 하나의 용어집으로 합친다."""
    seen: dict[str, str] = {}
    for t in sorted(topics, key=lambda x: x.order):
        for kw in (t.keywords_with_brief or []):
            k = (kw.get("keyword") or "").strip()
            if k and k not in seen:
                seen[k] = (kw.get("brief") or "").strip()
        for c in (t.concepts_introduced or []):
            c = c.strip()
            if c and c not in seen:
                seen[c] = ""
    return [{"keyword": k, "brief": v} for k, v in seen.items()]


def build_summary(
    structure: Structure,
    labeled_segments: list[LabeledSegment],
    lecture_meta: dict,
    api_key: str,
    out_usage: dict | None = None,
) -> SummaryResult:
    client = anthropic.Anthropic(api_key=api_key)
    topics = structure.topics

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
                    source_quote=n.get("source_quote", ""),
                ))

    # 전체 transcript (순서대로 이어붙임)
    full_transcript = "\n\n".join(
        s.text for s in sorted(labeled_segments, key=lambda x: x.index)
    )

    _verify_quotes(topic_summaries, announcements, full_transcript)

    summary_chars = sum(
        len(kp.explanation) + len(kp.source_quote)
        for t in topic_summaries for kp in t.key_points
    )
    if full_transcript:
        print(f"  본문 압축률: {summary_chars/len(full_transcript):.0%} "
              f"(전사 {len(full_transcript):,}자 -> 요약 본문 {summary_chars:,}자)")

    return SummaryResult(
        lecture_meta=lecture_meta,
        announcements=announcements,
        topics=topic_summaries,
        qa_segments=qa_segments,
        full_transcript=full_transcript,
        action_items=structure.action_items,
        uncovered_points=structure.uncovered_points,
        glossary=_build_glossary(topic_summaries),
    )
