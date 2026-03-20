from __future__ import annotations

import errno
import gc
import io
import random
import ssl
import time
from pathlib import Path

import torch

try:
    import psutil
except ImportError:  # pragma: no cover - optional fallback
    psutil = None

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials as UserCredentials
from google.oauth2.service_account import Credentials as ServiceAccountCredentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload


GOOGLE_DRIVE_SCOPES = ["https://www.googleapis.com/auth/drive.file"]


def _is_transient_network_error(exc: BaseException) -> bool:
    """True for errors that often succeed on retry (TLS drops, resets, 5xx, rate limits)."""
    if isinstance(exc, (ssl.SSLError, ConnectionError, TimeoutError, BrokenPipeError)):
        return True
    if isinstance(exc, OSError):
        transient = {errno.ECONNRESET, errno.ETIMEDOUT, errno.EPIPE}
        for _name in ("ECONNABORTED", "ENOTCONN"):
            if hasattr(errno, _name):
                transient.add(getattr(errno, _name))
        if exc.errno is not None and exc.errno in transient:
            return True
    if isinstance(exc, HttpError):
        status = int(getattr(getattr(exc, "resp", None), "status", 0) or 0)
        if status in (408, 429, 500, 502, 503, 504):
            return True
    return False


class DriveUploader:
    """Upload files to Google Drive using either OAuth or a service account."""

    def __init__(
        self,
        folder_id: str,
        *,
        auth_mode: str = "oauth",
        credentials_json: str | Path | None = None,
        token_json: str | Path | None = None,
    ):
        self.folder_id = folder_id
        self.auth_mode = auth_mode
        self.credentials_json = str(credentials_json) if credentials_json is not None else None
        self.token_json = Path(token_json) if token_json is not None else None
        creds = self._load_credentials()
        self.service = build("drive", "v3", credentials=creds, cache_discovery=False)

    def _load_credentials(self):
        if self.auth_mode == "service-account":
            if self.credentials_json is None:
                raise ValueError("credentials_json is required for service-account auth")
            return ServiceAccountCredentials.from_service_account_file(
                self.credentials_json,
                scopes=GOOGLE_DRIVE_SCOPES,
            )

        if self.auth_mode != "oauth":
            raise ValueError(f"Unknown auth_mode: {self.auth_mode}")

        if self.token_json is not None and self.token_json.exists():
            creds = UserCredentials.from_authorized_user_file(str(self.token_json), GOOGLE_DRIVE_SCOPES)
        else:
            if self.credentials_json is None:
                raise ValueError("credentials_json is required for oauth auth")
            flow = InstalledAppFlow.from_client_secrets_file(self.credentials_json, GOOGLE_DRIVE_SCOPES)
            try:
                creds = flow.run_console()
            except AttributeError:
                creds = flow.run_local_server(port=0, open_browser=True)
            if self.token_json is not None:
                self.token_json.write_text(creds.to_json(), encoding="utf-8")

        if creds.expired and creds.refresh_token:
            creds.refresh(Request())
            if self.token_json is not None:
                self.token_json.write_text(creds.to_json(), encoding="utf-8")
        return creds

    def upload_file(
        self,
        local_path: str | Path,
        remote_name: str | None = None,
        *,
        max_retries: int = 8,
        base_delay_s: float = 2.0,
    ) -> str:
        """
        Upload with retries around resumable chunks. SSLEOFError / connection drops
        during long uploads are common; each retry restarts the resumable session.
        """
        path = Path(local_path)
        if not path.exists():
            raise FileNotFoundError(path)
        if max_retries < 1:
            raise ValueError("max_retries must be >= 1")

        name = remote_name or path.name

        for attempt in range(max_retries):
            try:
                metadata = {
                    "name": name,
                    "parents": [self.folder_id],
                }
                media = MediaFileUpload(str(path), mimetype="application/octet-stream", resumable=True)
                request = self.service.files().create(
                    body=metadata,
                    media_body=media,
                    fields="id,name",
                    supportsAllDrives=True,
                )

                response = None
                while response is None:
                    _status, response = request.next_chunk()
                return response["id"]
            except Exception as e:
                if attempt >= max_retries - 1 or not _is_transient_network_error(e):
                    raise
                delay = base_delay_s * (2**attempt) + random.uniform(0, 1.0)
                time.sleep(delay)

    def download_file(
        self,
        file_id: str,
        local_path: str | Path,
        *,
        max_retries: int = 8,
        base_delay_s: float = 2.0,
    ) -> Path:
        """Download a Drive file by ID to a local path. Returns the path."""
        if max_retries < 1:
            raise ValueError("max_retries must be >= 1")
        path = Path(local_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        for attempt in range(max_retries):
            try:
                request = self.service.files().get_media(fileId=file_id, supportsAllDrives=True)
                fh = io.BytesIO()
                downloader = MediaIoBaseDownload(fh, request, chunksize=1024 * 1024)
                done = False
                while not done:
                    _status, done = downloader.next_chunk()
                path.write_bytes(fh.getvalue())
                return path
            except Exception as e:
                if attempt >= max_retries - 1 or not _is_transient_network_error(e):
                    raise
                delay = base_delay_s * (2**attempt) + random.uniform(0, 1.0)
                time.sleep(delay)

    def list_files_in_folder(self, folder_id: str | None = None) -> list[dict]:
        """List files in a Drive folder. Returns list of dicts with 'id' and 'name'."""
        folder_id = folder_id or self.folder_id
        result = (
            self.service.files()
            .list(
                q=f"'{folder_id}' in parents",
                pageSize=1000,
                fields="files(id,name)",
                supportsAllDrives=True,
                includeItemsFromAllDrives=True,
            )
            .execute()
        )
        return result.get("files", [])


def _cpu_free_gb() -> float | None:
    if psutil is None:
        return None
    return psutil.virtual_memory().available / (1024**3)


def _gpu_free_gb(device: str | None) -> float | None:
    if device is None or not torch.cuda.is_available():
        return None
    if device == "cuda":
        idx = torch.cuda.current_device()
    elif device.startswith("cuda:"):
        idx = int(device.split(":", 1)[1])
    else:
        return None
    free_bytes, _total_bytes = torch.cuda.mem_get_info(idx)
    return free_bytes / (1024**3)


def ensure_memory_budget(
    *,
    device: str | None,
    min_cpu_free_gb: float = 1.0,
    min_gpu_free_gb: float = 1.0,
    poll_seconds: int = 5,
) -> None:
    """
    Wait until enough CPU/GPU memory is available for generation.

    This is intentionally conservative and also clears caches before each check.
    """
    while True:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        cpu_free = _cpu_free_gb()
        gpu_free = _gpu_free_gb(device)

        cpu_ok = cpu_free is None or cpu_free >= min_cpu_free_gb
        gpu_ok = gpu_free is None or gpu_free >= min_gpu_free_gb

        if cpu_ok and gpu_ok:
            return

        time.sleep(max(1, poll_seconds))

