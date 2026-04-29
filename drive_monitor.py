"""
Google Drive 폴더를 모니터링해서 새 음성 파일을 감지하는 모듈
"""

import os
import json
from pathlib import Path
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from google.oauth2 import service_account
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
import google.auth
import io

# OAuth2 스코프
SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]

# 지원하는 음성/영상 파일 형식
SUPPORTED_FORMATS = {
    "audio/mpeg",        # .mp3
    "audio/mp4",         # .m4a
    "audio/wav",         # .wav
    "audio/x-wav",
    "audio/webm",        # .webm
    "audio/ogg",         # .ogg
    "video/mp4",         # .mp4 (영상에서 음성 추출)
    "audio/x-m4a",
    "audio/aac",         # .aac
    "audio/x-aac",       # .aac (일부 기기)
}

SUPPORTED_EXTENSIONS = {".mp3", ".m4a", ".wav", ".webm", ".ogg", ".mp4", ".mpeg", ".aac"}


def get_drive_service(credentials_path: str):
    """Google Drive API 서비스 객체 생성 (OAuth2 방식)"""
    creds = None
    token_path = Path(credentials_path).parent / "token.json"

    # 기존 토큰 로드
    if token_path.exists():
        from google.oauth2.credentials import Credentials
        creds = Credentials.from_authorized_user_file(str(token_path), SCOPES)

    # 토큰 없거나 만료된 경우 재인증
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(credentials_path, SCOPES)
            creds = flow.run_local_server(port=0)

        # 토큰 저장 (다음 실행 시 재사용)
        with open(token_path, "w") as f:
            f.write(creds.to_json())

    return build("drive", "v3", credentials=creds)


def _list_items_in_folder(service, folder_id: str) -> list[dict]:
    """폴더 안의 모든 항목(파일 + 하위 폴더) 반환"""
    query = f"'{folder_id}' in parents and trashed = false"
    results = service.files().list(
        q=query,
        fields="files(id, name, mimeType, createdTime, size)",
        orderBy="createdTime desc",
        pageSize=100,
    ).execute()
    return results.get("files", [])


def get_new_audio_files(service, folder_id: str, processed_ids: set) -> list[dict]:
    """
    지정 폴더 및 하위 폴더를 재귀 탐색해서 새 음성 파일 목록 반환
    반환: [{"id": ..., "name": ..., "mimeType": ..., "createdTime": ..., "lecture_name": ...}, ...]
    - lecture_name: 파일이 위치한 하위 폴더명 (최상위 폴더 직속이면 빈 문자열)
    """
    new_files = []
    _scan_folder(service, folder_id, processed_ids, new_files, lecture_name="")
    return new_files


def _scan_folder(service, folder_id: str, processed_ids: set, result: list, lecture_name: str):
    """폴더를 재귀 탐색해서 음성 파일 수집"""
    items = _list_items_in_folder(service, folder_id)

    for item in items:
        mime = item.get("mimeType", "")
        ext = Path(item["name"]).suffix.lower()

        if mime == "application/vnd.google-apps.folder":
            # 하위 폴더면 그 폴더명을 강의명으로 넘겨서 재귀 탐색
            _scan_folder(service, item["id"], processed_ids, result, lecture_name=item["name"])

        else:
            # 음성 파일 여부 확인
            is_supported = (mime in SUPPORTED_FORMATS or ext in SUPPORTED_EXTENSIONS)
            if is_supported and item["id"] not in processed_ids:
                item["lecture_name"] = lecture_name  # 폴더명 = 강의명
                result.append(item)


def download_file(service, file_id: str, file_name: str, dest_dir: str) -> str:
    """Google Drive에서 파일을 로컬로 다운로드. 저장 경로 반환"""
    dest_path = os.path.join(dest_dir, file_name)

    request = service.files().get_media(fileId=file_id)
    with open(dest_path, "wb") as f:
        downloader = MediaIoBaseDownload(f, request)
        done = False
        while not done:
            status, done = downloader.next_chunk()
            if status:
                print(f"  다운로드 중... {int(status.progress() * 100)}%")

    return dest_path
