"""
OpenAI Whisper API로 음성 파일을 텍스트로 변환하는 모듈
- Whisper API 파일 크기 제한: 25MB
- 처리 순서:
  1. 미지원 형식(aac 등) → 원본 음질 유지하며 mp3 변환
  2. 25MB 이하 → 바로 전송
  3. 25MB 초과 → 32kbps 모노 압축 (보통 여기서 해결)
  4. 압축 후에도 초과 → 청크 분할
"""

import os
import subprocess
import shutil
from pathlib import Path
from openai import OpenAI

# Whisper API 파일 크기 제한 (25MB, 여유분 포함)
MAX_FILE_SIZE_BYTES = 24 * 1024 * 1024

# Whisper API가 지원하는 확장자
WHISPER_SUPPORTED = {".flac", ".m4a", ".mp3", ".mp4", ".mpeg", ".mpga", ".oga", ".ogg", ".wav", ".webm"}


def _check_ffmpeg():
    if not shutil.which("ffmpeg"):
        raise RuntimeError(
            "ffmpeg가 설치되어 있지 않습니다.\n"
            "https://ffmpeg.org/download.html 에서 설치 후 PATH에 추가해주세요."
        )


def _convert_to_mp3(audio_path: str) -> str:
    """
    미지원 형식(aac 등)을 mp3로 변환. 음질 유지 (128kbps 스테레오).
    크기 감소가 목적이 아닌 순수 형식 변환용.
    """
    _check_ffmpeg()
    out_path = str(Path(audio_path).with_suffix("")) + "_converted.mp3"
    cmd = [
        "ffmpeg", "-y",
        "-i", audio_path,
        "-b:a", "128k",      # 128kbps (음질 유지)
        "-f", "mp3",
        out_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg 변환 실패:\n{result.stderr}")
    return out_path


def _compress_to_mp3(audio_path: str) -> str:
    """
    ffmpeg로 32kbps 모노 mp3 압축. 25MB 초과 시 용량 감소 목적.
    1시간 강의 기준 약 14MB → 청크 분할 없이 한 번에 처리 가능.
    """
    _check_ffmpeg()
    out_path = str(Path(audio_path).with_suffix("")) + "_compressed.mp3"
    cmd = [
        "ffmpeg", "-y",
        "-i", audio_path,
        "-ac", "1",          # 모노
        "-ar", "16000",      # 16kHz (Whisper 권장 샘플레이트)
        "-b:a", "32k",       # 32kbps
        "-f", "mp3",
        out_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg 압축 실패:\n{result.stderr}")
    return out_path


def get_audio_duration_minutes(audio_path: str) -> float:
    """ffprobe로 음성 파일 길이(분) 반환. ffmpeg 미설치 시 0 반환."""
    if not shutil.which("ffprobe"):
        return 0.0
    import json as _json
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", audio_path],
        capture_output=True, text=True, encoding="utf-8", errors="replace"
    )
    try:
        info = _json.loads(result.stdout)
        return float(info["format"]["duration"]) / 60
    except Exception:
        return 0.0


def transcribe_audio(api_key: str, audio_path: str, language: str = "en") -> tuple[str, float]:
    """
    음성 파일 → 텍스트 변환
    반환: (텍스트, 음성 길이(분))
    1. 원본 파일이 25MB 이하면 바로 전송
    2. 초과 시 ffmpeg로 32kbps 모노 압축 (빠름, 보통 여기서 해결됨)
    3. 압축 후에도 초과 시 청크 분할 (긴 강의 대비 안전망)
    """
    client = OpenAI(api_key=api_key, timeout=300.0)  # 5분 타임아웃
    temp_files = []

    try:
        # 음성 길이 측정 (원본 기준)
        duration_minutes = get_audio_duration_minutes(audio_path)

        # 미지원 형식이면 음질 유지하며 mp3로 변환
        ext = Path(audio_path).suffix.lower()
        if ext not in WHISPER_SUPPORTED:
            print("  미지원 형식 → mp3 변환 중 (음질 유지)...")
            audio_path = _convert_to_mp3(audio_path)
            temp_files.append(audio_path)

        file_size = os.path.getsize(audio_path)
        print(f"  파일 크기: {file_size / 1024 / 1024:.1f}MB")

        # 25MB 이하: 바로 전송
        if file_size <= MAX_FILE_SIZE_BYTES:
            return _transcribe_single(client, audio_path, language), duration_minutes

        # 25MB 초과: 압축 시도
        print("  25MB 초과 → 압축 중 (32kbps 모노)...")
        compressed = _compress_to_mp3(audio_path)
        temp_files.append(compressed)

        compressed_size = os.path.getsize(compressed)
        print(f"  압축 후 크기: {compressed_size / 1024 / 1024:.1f}MB")

        if compressed_size <= MAX_FILE_SIZE_BYTES:
            return _transcribe_single(client, compressed, language), duration_minutes

        # 압축 후에도 초과 시 청크 분할 (매우 긴 강의)
        print("  압축 후에도 초과 → 청크 분할 변환...")
        return _transcribe_chunked(client, compressed, language), duration_minutes

    finally:
        for f in temp_files:
            if Path(f).exists():
                Path(f).unlink()


def _remove_repetition(text: str) -> str:
    """
    Whisper hallucination 제거: 문장/어구가 연속으로 반복되는 패턴을 감지해 제거.

    동작 방식:
    - 텍스트를 문장 단위로 분리
    - 동일 문장이 연속 3회 이상 반복되면 1회만 남기고 이후 내용 잘라냄
    - 단어 단위 반복도 추가 처리
    """
    import re

    # 1단계: 문장 단위 반복 제거 (마침표, 줄바꿈 기준)
    sentences = re.split(r'(?<=[.!?\n])\s*', text)
    cleaned = []
    repeat_count = 0

    for i, sentence in enumerate(sentences):
        s = sentence.strip()
        if not s:
            continue
        # 직전 문장과 동일하면 반복 카운트 증가
        if cleaned and s == cleaned[-1].strip():
            repeat_count += 1
            if repeat_count >= 2:
                # 3회 이상 반복 시 이후 모두 제거
                break
        else:
            repeat_count = 0
            cleaned.append(sentence)

    result = " ".join(cleaned).strip()

    # 2단계: 짧은 어구/단어 반복 제거 (예: "감사합니다 감사합니다 감사합니다...")
    # 동일 어구가 5회 이상 연속되면 1회만 남김
    result = re.sub(r'(.{2,30}?)(\s*\1){4,}', r'\1', result)

    return result


def _transcribe_single(client: OpenAI, audio_path: str, language: str, retries: int = 3) -> str:
    """단일 파일 변환. 타임아웃 시 최대 3회 재시도."""
    import time
    for attempt in range(retries):
        try:
            with open(audio_path, "rb") as audio_file:
                transcript = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    language=language,
                    response_format="text",
                    timeout=300,
                )
            return _remove_repetition(transcript)
        except Exception as e:
            if attempt < retries - 1 and "timed out" in str(e).lower():
                wait = 10 * (attempt + 1)
                print(f"  타임아웃 → {wait}초 후 재시도 ({attempt+1}/{retries-1})...")
                time.sleep(wait)
            else:
                raise


def _transcribe_chunked(client: OpenAI, audio_path: str, language: str) -> str:
    """
    25MB 초과 파일을 ffmpeg로 10분 단위 분할 후 변환
    """
    import json as _json

    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", audio_path],
        capture_output=True, text=True, encoding="utf-8", errors="replace"
    )
    try:
        total_seconds = float(_json.loads(result.stdout)["format"]["duration"])
    except Exception:
        raise RuntimeError("ffprobe로 파일 길이를 읽지 못했습니다.")

    chunk_seconds = 10 * 60  # 10분 (20분에서 줄임 — 청크당 업로드 크기 감소)
    starts = list(range(0, int(total_seconds), chunk_seconds))
    total_chunks = len(starts)

    temp_dir = Path(audio_path).parent
    full_transcript = []
    chunk_files = []

    try:
        for i, start in enumerate(starts):
            chunk_path = str(temp_dir / f"chunk_{i}_{Path(audio_path).stem}.mp3")
            cmd = [
                "ffmpeg", "-y", "-i", audio_path,
                "-ss", str(start), "-t", str(chunk_seconds),
                "-ac", "1", "-ar", "16000", "-b:a", "32k",
                "-f", "mp3", chunk_path
            ]
            r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
            if r.returncode != 0:
                raise RuntimeError(f"청크 분할 실패 (청크 {i}): {r.stderr}")
            chunk_files.append(chunk_path)

        for i, chunk_path in enumerate(chunk_files):
            print(f"  청크 {i+1}/{total_chunks} 변환 중...")
            text = _transcribe_single(client, chunk_path, language)
            full_transcript.append(text)

    finally:
        for f in chunk_files:
            if Path(f).exists():
                Path(f).unlink()

    return "\n".join(full_transcript)
