from pipeline.schema import Segment

GAP_THRESHOLD_SEC = 3       # 청크 경계: 무음 3초 이상
MIN_CHUNK_SEC = 5 * 60      # 최소 청크 길이: 5분
MAX_CHUNK_SEC = 15 * 60     # 최대 청크 길이: 15분


def _fmt(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _extract_raw_segments(verbose_json) -> list[dict]:
    """Whisper API 응답(객체 또는 dict)에서 segments 추출"""
    if isinstance(verbose_json, dict):
        segs = verbose_json.get("segments", [])
    else:
        segs = getattr(verbose_json, "segments", []) or []

    result = []
    for seg in segs:
        if isinstance(seg, dict):
            result.append({"start": float(seg["start"]), "end": float(seg["end"]), "text": seg["text"]})
        else:
            result.append({"start": float(seg.start), "end": float(seg.end), "text": seg.text})
    return result


def segment_transcript(verbose_json) -> list[Segment]:
    raw = _extract_raw_segments(verbose_json)
    if not raw:
        return []

    # 1단계: gap 기반 그룹 분리
    groups: list[list[dict]] = []
    current = [raw[0]]
    for seg in raw[1:]:
        if seg["start"] - current[-1]["end"] >= GAP_THRESHOLD_SEC:
            groups.append(current)
            current = [seg]
        else:
            current.append(seg)
    groups.append(current)

    # 2단계: 5분 미만 그룹을 다음 그룹과 합침
    merged: list[list[dict]] = []
    i = 0
    while i < len(groups):
        g = groups[i]
        duration = g[-1]["end"] - g[0]["start"]
        if duration < MIN_CHUNK_SEC and i + 1 < len(groups):
            groups[i + 1] = g + groups[i + 1]
        else:
            merged.append(g)
        i += 1

    # 3단계: 15분 초과 그룹 강제 분할
    final: list[list[dict]] = []
    for g in merged:
        if g[-1]["end"] - g[0]["start"] <= MAX_CHUNK_SEC:
            final.append(g)
            continue
        chunk_start = g[0]["start"]
        current_chunk: list[dict] = []
        for seg in g:
            current_chunk.append(seg)
            if seg["end"] - chunk_start >= MAX_CHUNK_SEC:
                final.append(current_chunk)
                current_chunk = []
                chunk_start = seg["end"]
        if current_chunk:
            final.append(current_chunk)

    # 4단계: Segment 객체 변환
    return [
        Segment(
            index=idx,
            start=group[0]["start"],
            end=group[-1]["end"],
            text=" ".join(s["text"].strip() for s in group).strip(),
        )
        for idx, group in enumerate(final)
    ]
