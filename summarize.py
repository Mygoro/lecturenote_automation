"""
강의 텍스트 처리 진입점.
segment → classify → structure → summarize 파이프라인을 순서대로 호출한다.
"""

from pipeline.segmenter import segment_transcript
from pipeline.classifier import classify_chunks
from pipeline.structurer import extract_structure
from pipeline.summarizer import build_summary
from pipeline.schema import SummaryResult


def _make_usage() -> dict:
    return {"input_tokens": 0, "output_tokens": 0}


def _summary_to_markdown(result: SummaryResult) -> str:
    """SummaryResult.topics → 마크다운 요약 텍스트"""
    lines = []
    for t in result.topics:
        lines.append(f"## {t.order}. {t.title}  ({t.time_range})")
        for kp in t.key_points:
            lines.append(f"- **{kp.claim}**")
            lines.append(f"  {kp.explanation}")
            lines.append(f"  > \"{kp.source_quote}\" ({kp.timestamp})")
        if t.important_emphasis:
            lines.append("\n**강조 포인트**")
            for e in t.important_emphasis:
                lines.append(f"- {e}")
        if t.concepts_introduced:
            lines.append("\n**핵심 개념**")
            for c in t.concepts_introduced:
                lines.append(f"- {c}")
        lines.append("")
    return "\n".join(lines)


def process_transcript(api_key: str, raw_transcript) -> tuple[str, list[dict], str, dict]:
    """
    파이프라인 진입점. main.py 인터페이스 유지.

    raw_transcript: transcribe_audio() 반환값 (TranscriptionVerbose 객체 또는 dict)
    반환: (full_transcript, notices, summary_markdown, usage_by_model)
      usage_by_model = {"haiku": {...}, "sonnet": {...}}
    """
    usage_by_model = {
        "haiku":  _make_usage(),
        "sonnet": _make_usage(),
    }

    # 1단계: 청크 분할
    print("   [1/4] 청크 분할 중...")
    segments = segment_transcript(raw_transcript)
    print(f"   {len(segments)}개 세그먼트 생성")

    # 2단계: 분류 + 공지 추출 (Haiku)
    print("   [2/4] 세그먼트 분류 중 (Haiku)...")
    labeled = classify_chunks(segments, api_key, out_usage=usage_by_model["haiku"])

    # 3단계: 토픽 추출 (Sonnet)
    print("   [3/4] 토픽 구조 추출 중 (Sonnet)...")
    topics = extract_structure(labeled, api_key, out_usage=usage_by_model["sonnet"])

    # 4단계: 토픽별 요약 + QA 추출 (Sonnet)
    print("   [4/4] 요약 생성 중 (Sonnet)...")
    lecture_meta = {"course": "", "week": "", "duration": "", "language": ""}
    result: SummaryResult = build_summary(topics, labeled, lecture_meta, api_key, out_usage=usage_by_model["sonnet"])

    # 반환값 변환 (기존 인터페이스 유지)
    notices = [
        {"type": n.type, "content": n.content, "deadline": n.deadline}
        for n in result.announcements
    ]
    summary_markdown = _summary_to_markdown(result)

    return result.full_transcript, notices, summary_markdown, usage_by_model
