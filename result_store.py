# result_store.py
"""
마지막 분석 결과 스냅샷의 저장·복원을 담당합니다.

Streamlit Cloud에서 '재접속해도 결과가 보이게' 하려면 두 가지를
분리해서 봐야 합니다.

1) 세션(session_state)은 브라우저 탭이 닫히거나 앱이 잠들면 사라집니다.
   -> 앱이 시작될 때 저장소에서 읽어와 복원해야 합니다.

2) 컨테이너 파일시스템은 재배포(deploy)와 재시작 때 초기화됩니다.
   -> 로컬 파일만으로는 '영구 보관'이 보장되지 않습니다.
      그래서 같은 스냅샷을 ① 로컬(gzip), ② 외부 HTTP 저장소,
         ③ 사용자가 직접 내려받기/올리기 로 모두 다룰 수 있게 합니다.

스냅샷 형식(버전 9)
    {
      "version": 9,
      "saved_at": "2026-11-19T16:40:00+09:00",
      "complete": true | false,          # 실행이 끝까지 갔는지
      "period": ["2023-11-19", "2026-11-19"],
      "meta": {"recent_signal_days": 7, "recent_signal_limit": 3},
      "records": [...],
      "results": [
         {
           "stock_name", "ticker", "latest", "messages",
           "image_filename",              # img/ 폴더의 원본 PNG 이름
           "preview": {mime,width,height,bytes,data(b64)},  # 저용량 JPEG
           "weekly_signals", "combined_signals", "metrics", "trades"
         }
      ]
    }

차트를 원본 PNG로 통째로 넣지 않는 이유:
    300dpi 5패널 PNG는 장당 1~3MB이고 base64로 33% 더 커집니다.
    20종목이면 수십 MB -> Streamlit Cloud(1GB)에서 저장·복원이
    느려지거나 실패합니다. 그래서 스냅샷에는 폭 1200px JPEG
    미리보기만 담고, 원본은 img/ 폴더 파일을 그대로 참조합니다.
"""

from __future__ import annotations

import base64
import gzip
import io
import json
import math
import os
import uuid
from datetime import date, datetime
from numbers import Integral, Real
from pathlib import Path

import pandas as pd
from PIL import Image

SNAPSHOT_VERSION = 16

# 미리보기 설정 (스냅샷 1건이 수 MB를 넘지 않도록)
PREVIEW_MAX_WIDTH = 1200
PREVIEW_QUALITIES = (82, 75, 68, 60, 52, 44)
PREVIEW_TARGET_BYTES = 320_000

REQUIRED_LATEST_KEYS = {
    "Weekly 신호",
    "Weekly+Monthly 신호",
    "Weekly 신호일",
    "Weekly+Monthly 신호일",
    "신호 조회 일수",
    "PER",
    "PER 분기",
}


class SnapshotError(ValueError):
    """저장 결과를 복원할 수 없을 때 발생합니다(사용자에게 보여줄 메시지)."""


# =========================================================
# JSON 변환
# =========================================================

def json_safe(value):
    """pandas/numpy 값과 결측값을 엄격한 JSON 값으로 바꿉니다."""
    if value is None or value is pd.NA or value is pd.NaT:
        return None

    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}

    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]

    if isinstance(value, (datetime, date, pd.Timestamp)):
        return value.isoformat()

    if isinstance(value, bool):
        return value

    if isinstance(value, Integral):
        return int(value)

    if isinstance(value, Real):
        number = float(value)
        return number if math.isfinite(number) else None

    if isinstance(value, str):
        return value

    if hasattr(value, "item"):  # numpy scalar
        return json_safe(value.item())

    raise TypeError(
        "JSON으로 저장할 수 없는 자료형입니다: "
        f"{type(value).__name__}"
    )


def frame_to_json(frame):
    """
    DataFrame을 인덱스·dtype까지 보존하는 JSON(dict)으로 바꿉니다.

    dtype을 함께 저장하는 이유:
        to_json/read_json 왕복에서 60.0 -> 60(int)처럼
        표시 형식이 바뀌어 승률 '60.0%'가 '60%'로 보이는 문제를 막습니다.
    """
    if frame is None or not isinstance(frame, pd.DataFrame):
        return None

    return {
        "data": json.loads(
            frame.to_json(
                orient="split",
                date_format="iso",
                force_ascii=False,
            )
        ),
        "dtypes": {
            str(column): str(dtype)
            for column, dtype in frame.dtypes.items()
        },
    }


def json_to_frame(data):
    """frame_to_json의 결과를 DataFrame으로 되돌립니다(구형식도 허용)."""
    if not data:
        return pd.DataFrame()

    # 신형식 {"data": {split}, "dtypes": {...}}
    # 구형식 {columns, index, data(list)} — 'data' 키가 리스트이므로 구분 필요
    if isinstance(data, dict) and isinstance(data.get("data"), dict):
        split = data["data"]
        dtypes = data.get("dtypes") or {}
    else:  # 이전 형식(plain split)
        split = data
        dtypes = {}

    if not split:
        return pd.DataFrame()

    frame = pd.read_json(
        io.StringIO(json.dumps(split, ensure_ascii=False)),
        orient="split",
    )

    for column, dtype in dtypes.items():
        if column not in frame.columns:
            continue

        try:
            frame[column] = frame[column].astype(dtype)
        except (TypeError, ValueError):
            pass  # 캐스팅 불가하면 pandas 추론값을 그대로 씁니다.

    return frame


# =========================================================
# 차트 미리보기
# =========================================================

def make_preview(png_bytes, max_width=PREVIEW_MAX_WIDTH):
    """PNG를 스냅샷용 JPEG 미리보기로 줄입니다."""
    if not png_bytes:
        return None

    with Image.open(io.BytesIO(png_bytes)) as image:
        image.load()

        if image.mode != "RGB":
            image = image.convert("RGB")

        if image.width > max_width:
            ratio = max_width / float(image.width)
            height = max(1, int(round(image.height * ratio)))
            image = image.resize((max_width, height), Image.LANCZOS)

        chosen = b""

        for quality in PREVIEW_QUALITIES:
            buffer = io.BytesIO()
            image.save(
                buffer,
                format="JPEG",
                quality=quality,
                optimize=True,
            )
            chosen = buffer.getvalue()

            if len(chosen) <= PREVIEW_TARGET_BYTES:
                break

        width, height = image.size

    return {
        "mime": "image/jpeg",
        "width": width,
        "height": height,
        "bytes": len(chosen),
        "data": base64.b64encode(chosen).decode("ascii"),
    }


# =========================================================
# 스냅샷 생성 / 복원
# =========================================================

def build_snapshot(
    results,
    records,
    period,
    saved_at=None,
    complete=False,
    meta=None,
    source="app",
):
    """분석 결과 목록을 저장 가능한 스냅샷(dict)으로 만듭니다."""
    if not results:
        raise SnapshotError("저장할 분석 결과가 없습니다.")

    saved_at = saved_at or datetime.now().astimezone().isoformat(
        timespec="seconds"
    )

    items = []

    for result in results:
        items.append({
            "stock_name": result["stock_name"],
            "ticker": result["ticker"],
            "latest": json_safe(result.get("latest") or {}),
            "nomad_stats": json_safe(result.get("nomad_stats") or {}),
            "nomad_sell_stats": json_safe(result.get("nomad_sell_stats") or {}),
            "double_stats": json_safe(result.get("double_stats") or {}),
            "ma_stats": json_safe(result.get("ma_stats") or {}),
            "messages": json_safe(result.get("messages") or []),
            "image_filename": result.get("image_filename"),
            "preview": make_preview(result.get("png_bytes")),
            "weekly_signals": frame_to_json(result.get("weekly_signals")),
            "combined_signals": frame_to_json(result.get("combined_signals")),
            "metrics": frame_to_json(result.get("metrics")),
            "trades": {
                strategy: frame_to_json(frame)
                for strategy, frame in (result.get("trades") or {}).items()
            },
        })

    return {
        "version": SNAPSHOT_VERSION,
        "saved_at": saved_at,
        "source": str(source),          # "app" | "batch"
        "complete": bool(complete),
        "period": list(period) if period else None,
        "meta": dict(meta or {}),
        "records": json_safe(list(records or [])),
        "results": items,
    }


def restore_snapshot(payload, image_dir=None):
    """
    스냅샷을 세션에서 쓰는 결과 목록으로 되돌립니다.

    차트는 ① img/ 폴더의 원본 PNG를 먼저 찾고,
             ② 없으면 스냅샷에 든 JPEG 미리보기를 사용합니다.
    """
    if not isinstance(payload, dict):
        raise SnapshotError("저장 결과가 비어 있습니다.")

    version = payload.get("version")

    if version != SNAPSHOT_VERSION:
        raise SnapshotError(
            f"저장 결과 형식이 다릅니다(저장 버전 {version}, "
            f"현재 {SNAPSHOT_VERSION})."
        )

    items = payload.get("results") or []

    if not items:
        raise SnapshotError("저장 결과에 종목 데이터가 없습니다.")

    meta = payload.get("meta") or {}
    expected_days = meta.get("recent_signal_days")
    expected_limit = meta.get("recent_signal_limit")

    results = []
    notes = []

    preview_count = 0

    for item in items:
        latest = item.get("latest") or {}
        missing = REQUIRED_LATEST_KEYS - set(latest)

        if missing:
            raise SnapshotError(
                "저장 결과에 최신 신호·PER 정보가 없습니다"
                f"(누락: {', '.join(sorted(missing))})."
            )

        if expected_days and latest.get("신호 조회 일수") != expected_days:
            raise SnapshotError(
                "저장 결과의 신호 조회 기간이 현재 설정과 다릅니다."
            )

        if (
            expected_limit
            and latest.get("신호 개수 제한") is not None
            and latest.get("신호 개수 제한") != expected_limit
        ):
            raise SnapshotError(
                "저장 결과의 신호 개수 설정이 현재와 다릅니다."
            )

        image_bytes = None
        image_mime = "image/png"
        image_name = item.get("image_filename")
        image_note = None
        is_preview = False

        if image_dir is not None and image_name:
            candidate = Path(image_dir) / str(image_name)

            if candidate.is_file():
                image_bytes = candidate.read_bytes()

        if image_bytes is None:
            preview = item.get("preview")

            if preview and preview.get("data"):
                image_bytes = base64.b64decode(preview["data"])
                image_mime = preview.get("mime") or "image/jpeg"
                is_preview = True
                preview_count += 1
                image_note = (
                    "원본 차트 파일(img/)이 없어 압축 미리보기로 표시합니다. "
                    "원본이 필요하면 '분석 실행'을 다시 눌러 주세요."
                )

                if image_name:
                    image_name = str(
                        Path(str(image_name)).with_suffix(".jpg")
                    )

        results.append({
            "stock_name": item["stock_name"],
            "ticker": item["ticker"],
            "latest": latest,
            "nomad_stats": item.get("nomad_stats") or {},
            "nomad_sell_stats": item.get("nomad_sell_stats") or {},
            "double_stats": item.get("double_stats") or {},
            "ma_stats": item.get("ma_stats") or {},
            "messages": item.get("messages") or [],
            "image_filename": item.get("image_filename"),
            "image_download_name": image_name,
            "image_mime": image_mime,
            "image_is_preview": is_preview,
            "image_note": image_note,
            "png_bytes": image_bytes,
            "weekly_signals": json_to_frame(item.get("weekly_signals")),
            "combined_signals": json_to_frame(item.get("combined_signals")),
            "metrics": json_to_frame(item.get("metrics")),
            "trades": {
                strategy: json_to_frame(frame_data)
                for strategy, frame_data in (item.get("trades") or {}).items()
            },
        })

    if not results:
        raise SnapshotError("복원할 수 있는 종목 결과가 없습니다.")

    if not payload.get("complete", True):
        notes.append(
            f"마지막 실행이 끝나기 전에 저장된 부분 결과입니다"
            f"(종목 {len(results)}개)."
        )

    if preview_count:
        notes.append(
            f"{preview_count}개 종목의 차트는 원본 PNG가 없어 "
            "미리보기로 표시합니다."
        )

    return {
        "results": results,
        "records": payload.get("records") or [],
        "period": payload.get("period"),
        "saved_at": payload.get("saved_at"),
        "source": payload.get("source") or "app",
        "complete": bool(payload.get("complete", True)),
        "notes": notes,
    }


# =========================================================
# 직렬화
# =========================================================

def dumps(payload):
    """스냅샷을 gzip 압축 JSON 바이트로 만듭니다."""
    text = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    )
    return gzip.compress(text.encode("utf-8"), compresslevel=6)


def loads(blob):
    """gzip 또는 일반 JSON 바이트를 스냅샷(dict)으로 읽습니다."""
    if isinstance(blob, str):
        blob = blob.encode("utf-8")

    if not isinstance(blob, (bytes, bytearray)):
        raise SnapshotError("저장 파일 형식을 읽을 수 없습니다.")

    if bytes(blob[:2]) == b"\x1f\x8b":
        blob = gzip.decompress(blob)

    return json.loads(bytes(blob).decode("utf-8"))


# =========================================================
# 저장소 구현
# =========================================================

class LocalStore:
    """컨테이너 파일시스템에 gzip 스냅샷을 원자적으로 기록합니다.

    주의: Streamlit Cloud는 재배포·재시작 시 이 파일이 사라집니다.
    같은 컨테이너가 살아 있는 동안의 '재접속 복원'은 이 저장소로 충분합니다.
    """

    backend = "local"

    def __init__(self, path):
        self.path = Path(path)
        self.last_size = 0

    def save(self, payload):
        data = dumps(payload)

        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.parent / f".{self.path.name}.{uuid.uuid4().hex}.tmp"

        try:
            temporary.write_bytes(data)
            os.replace(temporary, self.path)
        finally:
            temporary.unlink(missing_ok=True)

        self.last_size = len(data)
        return len(data)

    def load(self):
        if not self.path.is_file():
            return None

        self.last_size = self.path.stat().st_size
        return loads(self.path.read_bytes())

    def download_bytes(self):
        if not self.path.is_file():
            return None

        return self.path.read_bytes()

    def describe(self):
        return f"로컬 파일 · output/{self.path.name}"


class HttpStore:
    """외부 HTTP 저장소(S3 presigned URL, Supabase Storage 등)에 보관합니다.

    st.secrets 예시:
        [result_store]
        backend = "http"
        url = "https://.../last_analysis.json.gz"
        token = "..."      # Authorization: Bearer (선택)
    """

    backend = "http"

    def __init__(self, url, token=None, timeout=20):
        self.url = str(url)
        self.token = token
        self.timeout = timeout
        self.last_size = 0

    def _headers(self):
        headers = {"Content-Type": "application/gzip"}

        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"

        return headers

    def _requests(self):
        import requests  # 지연 import: 로컬 모드에서는 필요 없습니다.

        return requests

    def save(self, payload):
        requests = self._requests()
        data = dumps(payload)

        response = requests.put(
            self.url,
            data=data,
            headers=self._headers(),
            timeout=self.timeout,
        )
        response.raise_for_status()

        self.last_size = len(data)
        return len(data)

    def load(self):
        requests = self._requests()

        response = requests.get(
            self.url,
            headers=self._headers(),
            timeout=self.timeout,
        )

        if response.status_code == 404:
            return None

        response.raise_for_status()

        self.last_size = len(response.content)
        return loads(response.content)

    def download_bytes(self):
        requests = self._requests()

        response = requests.get(
            self.url,
            headers=self._headers(),
            timeout=self.timeout,
        )

        if response.status_code == 404:
            return None

        response.raise_for_status()
        return response.content

    def describe(self):
        return "외부 HTTP 저장소"


class GitHubStore:
    """
    GitHub 저장소에 커밋된 스냅샷을 읽습니다(읽기 전용).

    왜 필요한가
        Actions가 결과를 저장소에 커밋해도 Streamlit Cloud가
        항상 즉시 재배포되는 것은 아닙니다(수동 재시작·브랜치 불일치 등).
        그러면 앱은 옛 파일을 계속 보여 줍니다.
        이 저장소를 쓰면 앱이 깨어날 때마다 저장소의 최신 스냅샷을
        직접 가져오므로, 재배포를 기다리지 않고 결과를 볼 수 있습니다.

    설정 예시 (.streamlit/secrets.toml)
        [result_store]
        backend = "github"
        repo    = "owner/repo"
        path    = "output/last_analysis.json.gz"
        branch  = "main"
        token   = "ghp_..."     # private 저장소면 필요
    """

    backend = "github"

    def __init__(
        self,
        repo,
        path="output/last_analysis.json.gz",
        branch="main",
        token=None,
        api_base="https://api.github.com",
        timeout=20,
    ):
        self.repo = str(repo).strip().strip("/")
        self.path = str(path).lstrip("/")
        self.branch = str(branch or "main")
        self.token = token
        self.api_base = api_base.rstrip("/")
        self.timeout = timeout
        self.last_size = 0

    @property
    def raw_url(self):
        return f"{self.api_base}/repos/{self.repo}/contents/{self.path}"

    def _session(self):
        import requests  # 지연 import

        session = requests.Session()
        headers = {
            "Accept": "application/vnd.github.raw",
            "X-GitHub-Api-Version": "2022-11-28",
        }

        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"

        session.headers.update(headers)
        return session

    def fetch_bytes(self):
        """스냅샷 원본 바이트를 가져옵니다(없으면 None)."""
        session = self._session()

        response = session.get(
            self.raw_url,
            params={"ref": self.branch},
            timeout=self.timeout,
        )

        if response.status_code == 404:
            return None

        response.raise_for_status()

        content = response.content
        self.last_size = len(content)
        return content

    def load(self):
        content = self.fetch_bytes()
        return loads(content) if content else None

    def download_bytes(self):
        return self.fetch_bytes()

    def save(self, payload):
        # 배치는 git commit으로 올립니다. 앱에서는 로컬에 저장합니다.
        raise SnapshotError(
            "GitHub 저장소는 읽기 전용입니다(결과는 배치가 커밋합니다)."
        )

    def describe(self):
        return f"GitHub {self.repo} · {self.branch}:{self.path}"


class FallbackStore:
    """
    여러 저장소를 함께 사용합니다.

    - 읽기: 둘 다 읽어 '더 최신(saved_at)' 스냅샷을 씁니다.
    - 쓰기: 로컬에 저장합니다(원격은 읽기 전용).
    """

    backend = "fallback"

    def __init__(self, stores, writer=None):
        self.stores = [store for store in stores if store is not None]
        self.writer = writer or self.stores[-1]
        self.last_size = 0

    @staticmethod
    def _saved_at(payload):
        return str((payload or {}).get("saved_at") or "")

    def load(self):
        newest = None

        for store in self.stores:
            try:
                payload = store.load()

            except Exception:
                continue  # 한 저장소 장애가 복원을 막지 않게

            if not payload:
                continue

            if newest is None or self._saved_at(payload) > self._saved_at(newest):
                newest = payload

        if newest is not None:
            self.last_size = len(dumps(newest))

        return newest

    def save(self, payload):
        return self.writer.save(payload)

    def download_bytes(self):
        for store in self.stores:
            try:
                data = store.download_bytes()

            except Exception:
                continue

            if data:
                return data

        return None

    def describe(self):
        return " · ".join(store.describe() for store in self.stores)


def make_store(base_dir, secrets=None, filename="last_analysis.json.gz"):
    """
    설정에 맞는 저장소를 만듭니다.

    backend = "local"   (기본) output/ 아래 파일
    backend = "http"    외부 HTTP 저장소(읽기/쓰기)
    backend = "github"  저장소에 커밋된 스냅샷 읽기 + 로컬 쓰기
    """
    config = {}

    try:
        container = (secrets or {}).get("result_store")

        if container:
            config = dict(container)
    except Exception:
        config = {}

    backend = str(config.get("backend", "local")).strip().lower()
    local = LocalStore(Path(base_dir) / "output" / filename)

    if backend in {"github", "repo", "git"} and config.get("repo"):
        remote = GitHubStore(
            repo=config["repo"],
            path=config.get("path") or f"output/{filename}",
            branch=config.get("branch") or "main",
            token=config.get("token"),
        )
        return FallbackStore([remote, local], writer=local)

    if backend in {"http", "https", "s3", "supabase", "webdav"}:
        url = config.get("url")

        if url:
            return HttpStore(url, token=config.get("token"))

    return local
