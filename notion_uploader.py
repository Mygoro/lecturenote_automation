"""
Notion API로 강의 노트 데이터베이스에 페이지를 생성하는 모듈

실제 DB 속성:
- 제목 (title)
- 강의일 (date)
- 과목 (select) — 폴더명이 자동으로 과목명이 됨
- 요약 (text) — 요약 첫 줄을 간략히 저장
- 태그 (multi_select) — 자동요약, AI요약 태그 자동 부여
"""

from __future__ import annotations
import re
from datetime import datetime, date as date_type
from typing import TYPE_CHECKING
from notion_client import Client

if TYPE_CHECKING:
    from pipeline.schema import SummaryResult


def _calc_week_number(date_str: str, semester_start: str) -> int:
    """학기 시작일 기준으로 N주차 계산 (1주차부터 시작)"""
    recording_date = datetime.strptime(date_str, "%Y-%m-%d").date()
    start_date     = datetime.strptime(semester_start, "%Y-%m-%d").date()
    delta = (recording_date - start_date).days
    return max(1, delta // 7 + 1)


def _extract_week_from_filename(file_name: str) -> int | None:
    """파일명에서 'N주차' 패턴을 추출. 패턴이 없으면 None.

    예: "AI Agents 9주차.aac" → 9, "FW 7주차(1).m4a" → 7
    """
    if not file_name:
        return None
    m = re.search(r"(\d+)주차", file_name)
    return int(m.group(1)) if m else None


def _resolve_title(
    client: Client,
    database_id: str,
    lecture_name: str,
    week_num: int,
    file_name: str = "",
) -> str:
    """
    같은 과목+주차 기존 페이지 수를 확인해 제목 결정
    - 기존 0개 → "[과목] N주차"
    - 기존 1개 (부 번호 없음) → 기존 페이지에 "1부" 소급 추가, 새 페이지 "2부"
    - 기존에 이미 부 번호 있음 → count+1 부

    주차는 file_name 파싱을 먼저 시도하고, 실패 시 인자로 받은 week_num을 사용.
    """
    parsed_week = _extract_week_from_filename(file_name)
    if parsed_week is not None:
        week_num = parsed_week

    base = f"[{lecture_name}] {week_num}주차" if lecture_name else f"{week_num}주차"

    # 같은 과목+주차 페이지 조회
    if lecture_name:
        filter_obj = {
            "and": [
                {"property": "과목", "select": {"equals": lecture_name}},
                {"property": "제목", "title": {"contains": f"{week_num}주차"}},
            ]
        }
    else:
        filter_obj = {"property": "제목", "title": {"contains": f"{week_num}주차"}}

    results = client.databases.query(
        database_id=database_id,
        filter=filter_obj,
    ).get("results", [])

    count = len(results)

    if count == 0:
        return base  # 처음 업로드 → 부 번호 없음

    # 기존 페이지 중 "부"가 없는 것을 찾아 "1부" 소급 추가
    for page in results:
        title_blocks = page.get("properties", {}).get("제목", {}).get("title", [])
        existing_title = "".join(t.get("text", {}).get("content", "") for t in title_blocks)
        if "부" not in existing_title:
            client.pages.update(
                page_id=page["id"],
                properties={"제목": {"title": [{"text": {"content": existing_title + " 1부"}}]}},
            )
            print(f"  기존 페이지 소급 수정: '{existing_title}' → '{existing_title} 1부'")

    return f"{base} {count + 1}부"


MAX_LEN = 1900  # Notion 블록 텍스트 최대 길이


def _rich(content: str) -> list[dict]:
    """마크다운 인라인 볼드(**text**)를 Notion rich_text 배열로 변환"""
    import re
    parts = re.split(r'\*\*(.+?)\*\*', content)
    result = []
    for i, part in enumerate(parts):
        if not part:
            continue
        is_bold = (i % 2 == 1)
        # MAX_LEN 초과 시 분할
        while len(part) > MAX_LEN:
            result.append({
                "type": "text",
                "text": {"content": part[:MAX_LEN]},
                "annotations": {"bold": is_bold},
            })
            part = part[MAX_LEN:]
        result.append({
            "type": "text",
            "text": {"content": part},
            "annotations": {"bold": is_bold},
        })
    return result if result else [{"type": "text", "text": {"content": ""}}]


def _heading(text: str, level: int) -> dict:
    t = f"heading_{level}"
    return {"object": "block", "type": t, t: {"rich_text": _rich(text)}}


def _divider() -> dict:
    return {"object": "block", "type": "divider", "divider": {}}


def _bullet(text: str) -> dict:
    return {"object": "block", "type": "bulleted_list_item",
            "bulleted_list_item": {"rich_text": _rich(text[:MAX_LEN])}}


def _paragraph(text: str) -> dict:
    return {"object": "block", "type": "paragraph",
            "paragraph": {"rich_text": _rich(text[:MAX_LEN])}}


def _callout(text: str, emoji: str = "📢") -> dict:
    return {"object": "block", "type": "callout",
            "callout": {"rich_text": _rich(text[:MAX_LEN]), "icon": {"type": "emoji", "emoji": emoji}}}


def _callout_ex(text: str, emoji: str = "📢", color: str = "default", children: list | None = None) -> dict:
    block = {
        "object": "block",
        "type": "callout",
        "callout": {
            "rich_text": _rich(text[:MAX_LEN]),
            "icon": {"type": "emoji", "emoji": emoji},
            "color": color,
        }
    }
    if children:
        block["callout"]["children"] = children[:100]
    return block


def _toggle(title: str, children: list[dict]) -> list[dict]:
    """
    Notion toggle 블록. children은 별도 append_blocks로 추가해야 해서
    toggle 본체와 children 목록을 tuple로 반환하지 않고,
    toggle 블록만 반환 후 page_id로 나중에 채움.
    → 단순화를 위해 toggle 내부엔 최대 100개만 직접 넣음.
    """
    return [{
        "object": "block",
        "type": "toggle",
        "toggle": {
            "rich_text": _rich(title),
            "children": children[:100],
        }
    }]


def _markdown_to_blocks(text: str) -> list[dict]:
    """
    Claude 출력 마크다운 → Notion 블록 변환
    지원 형식:
      # H1 / ## H2 / ### H3
      - item / * item / *   item (공백 다수 포함)
          1. numbered (들여쓰기 무시, bullet로 처리)
      **bold** 인라인 처리
    """
    import re
    BULLET_RE  = re.compile(r'^(\s*[-*])\s+')   # -나 * 로 시작하는 bullet
    NUMBER_RE  = re.compile(r'^\s*\d+\.\s+')    # 1. 2. 숫자 목록

    blocks = []
    for line in text.split("\n"):
        line = line.rstrip()
        if not line:
            continue

        # 헤딩
        if line.startswith("### "):
            blocks.append(_heading(line[4:].strip(), 3))
        elif line.startswith("## "):
            blocks.append(_heading(line[3:].strip(), 2))
        elif line.startswith("# "):
            blocks.append(_heading(line[2:].strip(), 1))

        # bullet (- / * / *   )
        elif BULLET_RE.match(line):
            content = BULLET_RE.sub("", line).strip()
            blocks.append({
                "object": "block", "type": "bulleted_list_item",
                "bulleted_list_item": {"rich_text": _rich(content)},
            })

        # 숫자 목록 (numbered list)
        elif NUMBER_RE.match(line):
            content = NUMBER_RE.sub("", line).strip()
            blocks.append({
                "object": "block", "type": "numbered_list_item",
                "numbered_list_item": {"rich_text": _rich(content)},
            })

        # 구분선
        elif line.strip() in ("---", "***", "___"):
            blocks.append(_divider())

        # 일반 단락
        else:
            blocks.append({
                "object": "block", "type": "paragraph",
                "paragraph": {"rich_text": _rich(line)},
            })

    return blocks


def _quote(text: str) -> dict:
    return {"object": "block", "type": "quote",
            "quote": {"rich_text": _rich(text[:MAX_LEN])}}


def _paragraph_gray(text: str) -> dict:
    return {
        "object": "block",
        "type": "paragraph",
        "paragraph": {
            "rich_text": [{"type": "text", "text": {"content": text[:MAX_LEN]}, "annotations": {"color": "gray"}}]
        }
    }


def _todo(text: str) -> dict:
    return {"object": "block", "type": "to_do",
            "to_do": {"rich_text": _rich(text[:MAX_LEN]), "checked": False}}


def _summary_result_to_blocks(result: SummaryResult) -> list[dict]:
    """SummaryResult → Notion 블록 목록

    구성 순서: 할 일 체크리스트 → 목차 → 공지 → 토픽 본문 → Q&A → 용어집 → 미반영 항목.
    타임스탬프는 렌더링하지 않는다. 음성 파일이 노트와 함께 제공되지 않으므로
    검증에도 복습에도 쓸 수 없고, 청크 시작점이라 값 자체도 부정확하다.
    근거 표시는 전사문과 대조를 마친 인용문이 담당한다.
    """
    blocks = []

    # ── 1. 이번 주 할 일 (최상단)
    if result.action_items:
        blocks.append(_heading("✅ 이번 주 할 일", 2))
        for a in result.action_items:
            blocks.append(_todo(a))
        blocks.append(_divider())

    # ── 2. 목차 (토픽별 한 줄 핵심)
    topics_sorted = [t for t in sorted(result.topics, key=lambda x: x.order) if t.key_points]
    if len(topics_sorted) > 1:
        toc = []
        for t in topics_sorted:
            line = f"**{t.order}. {t.title}**"
            if t.summary_oneliner:
                line += f" — {t.summary_oneliner}"
            toc.append(_bullet(line))
        blocks.append(_callout_ex("**목차**", emoji="🗂️", color="gray_background", children=toc))
        blocks.append(_divider())

    # ── 3. 공지 (yellow callout) — 인용이 있으면 근거로 함께 표시
    if result.announcements:
        blocks.append(_heading("📢 공지 · 운영 규정", 2))
        for n in result.announcements:
            content = n.content
            if n.deadline:
                content += f"  (기한: {n.deadline})"
            children = []
            if n.source_quote:
                children.append(_paragraph_gray(f'"{n.source_quote}"'))
                emoji, color = "📢", "yellow_background"
            else:
                # 전사문과 대조되는 인용이 없는 공지. 타임스탬프를 버린 이상 대조가
                # 유일한 검증 수단이므로, 검증되지 않은 공지는 사실처럼 보이면 안 된다.
                children.append(_paragraph_gray("원문 대조 실패 — 직접 확인하세요"))
                emoji, color = "❓", "red_background"
            blocks.append(_callout_ex(f"**{content}**", emoji=emoji,
                                      color=color, children=children))
        blocks.append(_divider())

    # ── 4. 토픽 본문
    for t in topics_sorted:
        blocks.append(_heading(f"🎯 {t.title}", 2))
        if t.summary_oneliner:
            blocks.append(_callout_ex(f"**핵심**: {t.summary_oneliner}",
                                      emoji="💡", color="blue_background"))
        for kp in t.key_points:
            blocks.append(_paragraph(f"**{kp.explanation}**"))
            if kp.source_quote:
                blocks.append(_quote(f'"{kp.source_quote}"'))
        blocks.append(_divider())

    # ── 5. Q&A toggle
    if result.qa_segments:
        qa_blocks = []
        for qa in result.qa_segments:
            qa_blocks.append(_paragraph(f"Q: {qa.question}"))
            qa_blocks.append(_paragraph(f"A: {qa.answer}"))
        blocks.extend(_toggle("Q&A", qa_blocks))

    # ── 6. 용어집 (토픽별로 흩어진 키워드/개념을 하나로 통합)
    if result.glossary:
        gloss = []
        for g in result.glossary:
            kw, brief = g.get("keyword", ""), g.get("brief", "")
            gloss.append(_bullet(f"**{kw}** — {brief}" if brief else f"**{kw}**"))
        blocks.extend(_toggle(f"📖 용어집 ({len(result.glossary)}개)", gloss))

    # ── 7. 요약에 반영되지 않은 항목 (structurer의 누락 점검 결과)
    if result.uncovered_points:
        unc = [_bullet(u) for u in result.uncovered_points]
        blocks.append(_callout_ex(
            "**요약에 반영되지 않은 항목** — 원문에는 있으나 위 토픽에 담기지 않았습니다",
            emoji="🔎", color="orange_background", children=unc))

    return blocks


def _notices_to_blocks(notices: list[dict]) -> list[dict]:
    """공지 목록 → callout 블록 목록"""
    if not notices:
        return []

    EMOJI_MAP = {
        "assignment": "📝",
        "exam": "📋",
        "attendance": "✅",
        "schedule_change": "📅",
        "notice": "📢",
    }
    blocks = [_heading("중요 공지", level=1), _divider()]
    for n in notices:
        emoji = EMOJI_MAP.get(n.get("type", "notice"), "📢")
        content = n.get("content", "")
        deadline = n.get("deadline")
        if deadline:
            content += f"  (기한: {deadline})"
        blocks.append(_callout(content, emoji))
    blocks.append(_divider())
    return blocks


def _plain_text_blocks(text: str) -> list[dict]:
    """원본 텍스트 → paragraph 블록 목록 (toggle 내부용)"""
    blocks = []
    for line in text.split("\n"):
        line = line.rstrip()
        if not line:
            continue
        while len(line) > MAX_LEN:
            blocks.append(_paragraph(line[:MAX_LEN]))
            line = line[MAX_LEN:]
        blocks.append(_paragraph(line))
    return blocks


def _append_in_batches(client: Client, page_id: str, blocks: list[dict]):
    """100개 단위로 나눠서 블록 추가"""
    for i in range(0, len(blocks), 100):
        client.blocks.children.append(block_id=page_id, children=blocks[i:i+100])


def upload_to_notion(
    token: str,
    database_id: str,
    file_name: str,
    transcript: str,
    summary: str,
    notices: list[dict] = None,
    lecture_name: str = "",
    created_time: str = "",
    semester_start: str = "2026-03-03",
    summary_result: SummaryResult | None = None,
    drive_file_id: str = "",
) -> str:
    """
    Notion 강의 노트 DB에 페이지 생성
    반환: 생성된 Notion 페이지 URL
    """
    client = Client(auth=token)
    if notices is None:
        notices = []

    # 날짜 파싱
    if created_time:
        try:
            dt = datetime.fromisoformat(created_time.replace("Z", "+00:00"))
            date_str = dt.strftime("%Y-%m-%d")
        except Exception:
            date_str = datetime.now().strftime("%Y-%m-%d")
    else:
        date_str = datetime.now().strftime("%Y-%m-%d")

    # 주차 계산 + 중복 여부 확인해서 제목 결정
    # week_num은 폴백값 — _resolve_title이 file_name 파싱을 우선 시도
    week_num = _calc_week_number(date_str, semester_start)
    title = _resolve_title(client, database_id, lecture_name, week_num, file_name)
    print(f"  페이지 제목: {title}")

    # 페이지 속성
    properties = {
        "제목": {"title": [{"text": {"content": title}}]},
        "강의일": {"date": {"start": date_str}},
        "태그": {"multi_select": [{"name": "자동요약"}, {"name": "AI요약"}]},
    }

    # DB 목록에서 내용이 보이도록 토픽별 한 줄 핵심을 이어붙여 넣는다
    if summary_result is not None:
        oneliners = [t.summary_oneliner for t in
                     sorted(summary_result.topics, key=lambda x: x.order)
                     if t.summary_oneliner]
        if oneliners:
            properties["요약"] = {
                "rich_text": [{"text": {"content": " · ".join(oneliners)[:1900]}}]
            }
    if lecture_name:
        properties["과목"] = {"select": {"name": lecture_name}}
    if drive_file_id:
        properties["파일 ID"] = {"rich_text": [{"text": {"content": drive_file_id}}]}

    # 본문 블록 구성
    if summary_result is not None:
        # 새 구조: SummaryResult 기반
        content_blocks = _summary_result_to_blocks(summary_result)
    else:
        # 구 구조: 마크다운 문자열 기반 (하위 호환)
        content_blocks = (
            _notices_to_blocks(notices)
            + [_heading("강의 요약", level=1), _divider()]
            + _markdown_to_blocks(summary)
            + [_divider()]
        )
    transcript_blocks = _plain_text_blocks(transcript)

    # 원본 텍스트는 toggle로 접어서 숨김 (toggle 자체만 첫 배치에 포함)
    toggle_block = {
        "object": "block",
        "type": "toggle",
        "toggle": {
            "rich_text": _rich("원본 텍스트 (Whisper 변환)"),
            "children": transcript_blocks[:100],
        }
    }

    first_blocks = content_blocks + [toggle_block]

    # 페이지 생성 (첫 100개)
    try:
        response = client.pages.create(
            parent={"database_id": database_id},
            properties=properties,
            children=first_blocks[:100],
        )
    except Exception as e:
        if "파일 ID" in str(e) and "not a property" in str(e):
            print("  [warn] 파일 ID 속성 없음 - 해당 속성 제외하고 재시도")
            properties.pop("파일 ID", None)
            response = client.pages.create(
                parent={"database_id": database_id},
                properties=properties,
                children=first_blocks[:100],
            )
        else:
            raise
    page_id  = response["id"]
    page_url = response["url"]

    # 요약 나머지 블록 추가 (100개 초과 시)
    remaining = first_blocks[100:]
    if remaining:
        _append_in_batches(client, page_id, remaining)

    # toggle 내부 원본 텍스트 나머지 추가 (100개 초과 시)
    # toggle 블록 ID 조회
    if len(transcript_blocks) > 100:
        toggle_children = client.blocks.children.list(block_id=page_id).get("results", [])
        toggle_id = None
        for b in reversed(toggle_children):
            if b.get("type") == "toggle":
                toggle_id = b["id"]
                break
        if toggle_id:
            _append_in_batches(client, toggle_id, transcript_blocks[100:])

    return page_url
