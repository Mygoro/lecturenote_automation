"""
강의 녹음 자동 처리 메인 스크립트

실행 방법:
  python main.py

Windows 작업 스케줄러 등록 방법은 README 참고
"""

import os
import sys
import traceback
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Windows CMD 한글/이모지 출력 깨짐 방지
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
if sys.stderr.encoding != "utf-8":
    sys.stderr.reconfigure(encoding="utf-8")

# .env 파일 로드 (시스템 환경변수보다 .env 파일을 우선 적용)
load_dotenv(Path(__file__).parent / ".env", override=True)

from notion_client import Client as NotionClient
from drive_monitor import get_drive_service, get_new_audio_files, download_file
from transcribe import transcribe_audio
from summarize import process_transcript
from notion_uploader import upload_to_notion

# ────────────────────────────────────────────
# 설정
# ────────────────────────────────────────────
BASE_DIR         = Path(__file__).parent
TEMP_DIR         = BASE_DIR / "temp_audio"
CREDENTIALS_FILE = BASE_DIR / "google_credentials.json"
LOCK_FILE        = BASE_DIR / ".running.lock"

OPENAI_API_KEY      = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY   = os.getenv("ANTHROPIC_API_KEY")
NOTION_TOKEN        = os.getenv("NOTION_TOKEN")
NOTION_DATABASE_ID  = os.getenv("NOTION_DATABASE_ID")
DRIVE_FOLDER_ID            = os.getenv("DRIVE_FOLDER_ID")
EXTRACURRICULAR_FOLDER_ID  = os.getenv("EXTRACURRICULAR_FOLDER_ID")  # 비교과 폴더 (선택)
SEMESTER_START             = os.getenv("SEMESTER_START", "2026-03-02")
MAX_FILES_PER_RUN          = int(os.environ.get("MAX_FILES_PER_RUN", "1"))

# 과목별 Whisper 언어 코드 (미지정 시 자동 감지)
# key: Drive 폴더명 (lecture_name), value: Whisper language 코드
COURSE_LANGUAGES: dict[str, str] = {
    "RDQM": "en",
}


def load_processed() -> set:
    """Notion DB의 파일 ID 속성을 조회해서 처리 완료된 Drive 파일 ID set 반환"""
    client = NotionClient(auth=NOTION_TOKEN)
    processed_ids = set()
    has_more = True
    start_cursor = None

    while has_more:
        kwargs = {"database_id": NOTION_DATABASE_ID}
        if start_cursor:
            kwargs["start_cursor"] = start_cursor
        response = client.databases.query(**kwargs)

        for page in response.get("results", []):
            prop = page.get("properties", {}).get("파일 ID", {})
            rich_text = prop.get("rich_text", [])
            if rich_text:
                fid = rich_text[0].get("text", {}).get("content", "")
                if fid:
                    processed_ids.add(fid)

        has_more = response.get("has_more", False)
        start_cursor = response.get("next_cursor")

    return processed_ids


def check_env():
    """필수 환경변수 확인"""
    missing = []
    for var in ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "NOTION_TOKEN", "NOTION_DATABASE_ID", "DRIVE_FOLDER_ID"]:
        if not os.getenv(var) or "여기에" in (os.getenv(var) or ""):
            missing.append(var)

    if missing:
        print("❌ 다음 환경변수가 설정되지 않았습니다:")
        for var in missing:
            print(f"   - {var}")
        print("\n📝 automation/.env 파일을 열어 API 키를 입력해주세요.")
        sys.exit(1)

    if not CREDENTIALS_FILE.exists():
        print("❌ google_credentials.json 파일이 없습니다.")
        print("   README의 'Google Drive API 설정' 섹션을 참고해주세요.")
        sys.exit(1)


# ────────────────────────────────────────────
# API 단가
# ────────────────────────────────────────────
WHISPER_COST_PER_MIN = 0.006                        # $0.006 / 분
HAIKU_IN   =  0.8 / 1_000_000                       # $0.80  / 1M input
HAIKU_OUT  =  4.0 / 1_000_000                       # $4.00  / 1M output
SONNET_IN  =  2.0 / 1_000_000                       # $2.00  / 1M input  (Sonnet 5)
SONNET_OUT = 10.0 / 1_000_000                       # $10.00 / 1M output (Sonnet 5)
USD_TO_KRW = 1380


def print_cost_report(duration_minutes: float, usage_by_model: dict):
    """API 사용량 및 비용 출력 (모델별 분리)"""
    whisper_cost = duration_minutes * WHISPER_COST_PER_MIN

    h = usage_by_model.get("haiku",  {"input_tokens": 0, "output_tokens": 0})
    s = usage_by_model.get("sonnet", {"input_tokens": 0, "output_tokens": 0})

    haiku_cost  = h["input_tokens"] * HAIKU_IN  + h["output_tokens"] * HAIKU_OUT
    sonnet_cost = s["input_tokens"] * SONNET_IN + s["output_tokens"] * SONNET_OUT
    claude_cost = haiku_cost + sonnet_cost
    total_usd   = whisper_cost + claude_cost

    print(f"\n  ┌──────────────────────────────────────────┐")
    print(f"  │            API 사용량 리포트              │")
    print(f"  ├──────────────────────────────────────────┤")
    if duration_minutes > 0:
        print(f"  │ Whisper  {duration_minutes:5.1f}분  →  ${whisper_cost:.4f} ({whisper_cost*USD_TO_KRW:.0f}원)")
    else:
        print(f"  │ Whisper  (길이 측정 불가)")
    print(f"  │ Haiku    {h['input_tokens']:,}in / {h['output_tokens']:,}out  →  ${haiku_cost:.4f} ({haiku_cost*USD_TO_KRW:.0f}원)")
    print(f"  │ Sonnet   {s['input_tokens']:,}in / {s['output_tokens']:,}out  →  ${sonnet_cost:.4f} ({sonnet_cost*USD_TO_KRW:.0f}원)")
    print(f"  ├──────────────────────────────────────────┤")
    print(f"  │ 합계     ${total_usd:.4f} ({total_usd*USD_TO_KRW:.0f}원)")
    print(f"  └──────────────────────────────────────────┘")


def process_file(service, file_info: dict):
    """단일 파일 전체 처리 (다운로드 → 변환 → 요약 → 노션 업로드)"""
    file_id   = file_info["id"]
    file_name = file_info["name"]
    created   = file_info.get("createdTime", "")

    print(f"\n{'='*50}")
    print(f"📁 파일: {file_name}")
    print(f"{'='*50}")

    local_path = None
    try:
        # 1. 다운로드
        print("⬇️  다운로드 중...")
        TEMP_DIR.mkdir(exist_ok=True)
        local_path = download_file(service, file_id, file_name, str(TEMP_DIR))
        print(f"   완료: {local_path}")

        # 2. 음성 → 텍스트
        print("🎤 Whisper로 텍스트 변환 중...")
        language = COURSE_LANGUAGES.get(file_info.get("lecture_name", ""))
        if language:
            print(f"   언어 지정: {language}")
        transcript, duration_minutes = transcribe_audio(OPENAI_API_KEY, local_path, language=language)
        print(f"   완료 ({len(transcript.text if hasattr(transcript, 'text') else transcript['text'])}자, {duration_minutes:.1f}분)")

        # 3. 정제 → 공지 추출 → 요약
        print("🤖 Claude 3단계 처리 중...")
        cleaned, notices, summary, claude_usage, summary_result = process_transcript(ANTHROPIC_API_KEY, transcript)
        print(f"   완료 (공지 {len(notices)}건, 요약 {len(summary)}자)")

        # 비용 리포트
        print_cost_report(duration_minutes, claude_usage)

        # 4. 노션 업로드
        print("📓 Notion에 업로드 중...")
        page_url = upload_to_notion(
            token=NOTION_TOKEN,
            database_id=NOTION_DATABASE_ID,
            file_name=file_name,
            transcript=cleaned,
            summary=summary,
            notices=notices,
            lecture_name=file_info.get("lecture_name", ""),
            created_time=created,
            semester_start=SEMESTER_START,
            drive_file_id=file_id,
            summary_result=summary_result,
        )
        print(f"   완료: {page_url}")
        print(f"✅ 처리 완료: {file_name}")

    except Exception as e:
        print(f"❌ 오류 발생 ({file_name}): {e}")
        traceback.print_exc()
        raise

    finally:
        # 임시 파일 삭제
        if local_path and Path(local_path).exists():
            Path(local_path).unlink()
            print("🗑️  임시 파일 삭제 완료")


def main():
    # 중복 실행 방지 (수동/자동 모두 적용)
    if LOCK_FILE.exists():
        print("⚠️  이미 실행 중인 작업이 있습니다. 종료합니다.")
        print(f"   (락 파일: {LOCK_FILE})")
        print("   이전 작업이 비정상 종료된 경우 .running.lock 파일을 수동으로 삭제하세요.")
        sys.exit(0)

    LOCK_FILE.write_text(str(os.getpid()))
    try:
        _main()
    finally:
        LOCK_FILE.unlink(missing_ok=True)


def _main():
    print(f"\n🔍 강의 녹음 자동화 시작 [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]")

    # 환경변수 확인
    check_env()

    # Notion DB에서 처리 완료 파일 ID 로드
    print("\n📋 Notion DB에서 처리 이력 확인 중...")
    processed_ids = load_processed()
    print(f"   기존 처리 파일: {len(processed_ids)}개")

    # Google Drive 연결
    print("\n📡 Google Drive 연결 중...")
    service = get_drive_service(str(CREDENTIALS_FILE))
    print("   연결 완료")

    # 새 파일 확인
    print(f"\n🔎 새 음성 파일 확인 중...")
    new_files = get_new_audio_files(service, DRIVE_FOLDER_ID, processed_ids)

    if not new_files:
        print("   새 파일 없음. 종료합니다.")
        return

    print(f"   새 파일 {len(new_files)}개 발견!")

    # 각 파일 처리
    success_count = 0
    for file_info in new_files[:MAX_FILES_PER_RUN]:
        try:
            process_file(service, file_info)
            success_count += 1
        except Exception:
            print(f"   이 파일은 건너뜁니다.")
            continue

    print(f"\n🎉 완료! {success_count}/{len(new_files)}개 파일 처리됨")


if __name__ == "__main__":
    main()
