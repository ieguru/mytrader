# notifier.py
"""
배치 결과를 텔레그램으로 보냅니다.

설계 원칙
- 텔레그램 장애가 배치를 실패시키지 않습니다(호출부에서 예외를 흡수).
- 메시지는 4096자 제한이 있어 자동 분할합니다.
- HTML parse_mode 사용: MarkdownV2는 이스케이프 규칙이 까다로워
  종목명에 특수문자가 들어오면 조용히 실패합니다. HTML은
  &, <, > 세 가지만 escape하면 됩니다.
- 429(Too Many Requests)는 retry_after만큼 기다렸다가 재시도합니다.
- 신호가 없는 날에도 요약을 보냅니다. '조용한 날 = 장애'를
  구분할 수 없게 되기 때문입니다.

환경변수 / secrets
    TELEGRAM_BOT_TOKEN   BotFather가 준 토큰
    TELEGRAM_CHAT_ID     받을 chat_id (개인/그룹)
    TELEGRAM_DISABLE=1   알림 끄기(테스트용)
"""

from __future__ import annotations

import html
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

TELEGRAM_LIMIT = 4096
SAFE_LIMIT = 3800          # 분할 여유(헤더/푸터 자리)
DEFAULT_API_BASE = "https://api.telegram.org"

BUY_LABEL = "매수"
SELL_LABEL = "매도"


class TelegramError(RuntimeError):
    """텔레그램 API가 ok=false 또는 예상 밖 응답을 준 경우."""


# =========================================================
# 설정
# =========================================================

@dataclass
class TelegramConfig:
    token: str = ""
    chat_id: str = ""
    api_base: str = DEFAULT_API_BASE
    timeout: int = 20
    retries: int = 3
    max_photos: int = 3

    @property
    def configured(self):
        return bool(self.token and self.chat_id)


def config_from_env(environ, api_base=None):
    """환경변수에서 설정을 읽습니다."""
    return TelegramConfig(
        token=str(environ.get("TELEGRAM_BOT_TOKEN") or "").strip(),
        chat_id=str(environ.get("TELEGRAM_CHAT_ID") or "").strip(),
        api_base=api_base or DEFAULT_API_BASE,
        max_photos=int(environ.get("TELEGRAM_MAX_PHOTOS") or 3),
    )


def notification_disabled(environ):
    value = str(environ.get("TELEGRAM_DISABLE") or "").strip().lower()
    return value in {"1", "true", "yes", "on"}


# =========================================================
# 메시지 조립
# =========================================================

def esc(value):
    """HTML parse_mode용 escape."""
    return html.escape(str(value), quote=False)


def format_price(value):
    try:
        number = float(value)

        if number != number:  # NaN
            return "-"

        return f"{number:,.0f}"
    except (TypeError, ValueError):
        return "-"


def format_per(value):
    try:
        number = float(value)

        if number != number:
            return "-"

        return f"{number:.1f}"
    except (TypeError, ValueError):
        return "-"


def build_summary(records, run_date=None, title="일일 종목 분석"):
    """
    배치 실행 내역(records)으로 텔레그램 요약 HTML을 만듭니다.

    records 항목( batch_daily.py가 만드는 형식 )
        종목명, 종목코드, 상태(완료/실패),
        기준일, 신호, 신호일, 종가, PER, 차트, 경고
    """
    records = list(records or [])
    done = [r for r in records if r.get("상태") == "완료"]
    failed = [r for r in records if r.get("상태") == "실패"]

    buys = [r for r in done if r.get("신호") == BUY_LABEL]
    sells = [r for r in done if r.get("신호") == SELL_LABEL]
    quiet = [
        r for r in done
        if r.get("신호") not in (BUY_LABEL, SELL_LABEL)
    ]

    # 데이터 기준일: 가장 많이 등장한 값(휴장일이면 직전 거래일)
    basis = ""
    if done:
        counts = {}
        for row in done:
            key = str(row.get("기준일") or "")
            counts[key] = counts.get(key, 0) + 1
        basis = max(counts, key=counts.get)

    header = f"📊 <b>{esc(title)}</b>"

    if run_date:
        header += f" · {esc(run_date)}"

    lines = [header]

    if basis:
        lines.append(f"데이터 기준일 <b>{esc(basis)}</b>")

    lines.append(
        f"완료 <b>{len(done)}</b> / 실패 <b>{len(failed)}</b>"
        f" / 전체 <b>{len(records)}</b>"
    )
    lines.append("")

    def signal_block(rows, icon, label):
        if not rows:
            return

        lines.append(f"{icon} <b>{label} {len(rows)}종목</b>")

        for row in rows:
            name = esc(row.get("종목명", "?"))
            price = format_price(row.get("종가"))
            per = format_per(row.get("PER"))
            day = str(row.get("신호일") or "")
            tail = f" · PER {per}" if per != "-" else ""
            when = f" ({esc(day)})" if day else ""
            lines.append(
                f"• <b>{name}</b> — {price}원{tail}{when}"
            )

        lines.append("")

    signal_block(buys, "🟢", "매수 신호")
    signal_block(sells, "🔴", "매도 신호")

    if not buys and not sells:
        lines.append("⚪ 오늘 새 매매 신호는 없습니다.")
        lines.append("")

    if quiet:
        lines.append(f"⚪ 신호 없음 {len(quiet)}종목")

    # PER 계산 가능 종목 수(재무 엑셀 유무 파악용)
    per_ready = sum(
        1 for row in done
        if format_per(row.get("PER")) != "-"
    )
    lines.append(
        f"📄 PER 산출 {per_ready}/{len(done)}종목"
        " (엑셀 없는 종목은 목표가·PER 제외)"
    )

    warned = [r for r in done if str(r.get("경고") or "").strip()]
    if warned:
        lines.append(f"⚠️ 경고 {len(warned)}종목")

    if failed:
        lines.append("")
        lines.append(f"❌ <b>실패 {len(failed)}종목</b>")

        for row in failed[:5]:
            reason = str(row.get("경고") or "")[:120]
            lines.append(
                f"• {esc(row.get('종목명', '?'))}: {esc(reason)}"
            )

        if len(failed) > 5:
            lines.append(f"• 외 {len(failed) - 5}종목")

    return "\n".join(lines)


def split_message(text, limit=SAFE_LIMIT):
    """4096자 제한에 맞춰 줄 단위로 나눕니다."""
    if len(text) <= limit:
        return [text]

    chunks = []
    current = ""

    for line in text.split("\n"):
        # 한 줄이 limit보다 길면 강제로 자릅니다.
        while len(line) > limit:
            if current:
                chunks.append(current)
                current = ""

            chunks.append(line[:limit])
            line = line[limit:]

        candidate = f"{current}\n{line}" if current else line

        if len(candidate) > limit:
            chunks.append(current)
            current = line
        else:
            current = candidate

    if current:
        chunks.append(current)

    total = len(chunks)

    return [
        f"{chunk}\n\n<i>({index}/{total})</i>"
        for index, chunk in enumerate(chunks, start=1)
    ]


# =========================================================
# 전송
# =========================================================

@dataclass
class TelegramNotifier:
    config: TelegramConfig
    session: object = None            # requests.Session 주입(테스트용)
    sleep: object = field(default=time.sleep)
    sent: list = field(default_factory=list)

    def __post_init__(self):
        if self.session is None:
            import requests

            self.session = requests.Session()

    def _url(self, method):
        return f"{self.config.api_base}/bot{self.config.token}/{method}"

    def _handle(self, response):
        try:
            payload = response.json()
        except ValueError:
            raise TelegramError(
                f"응답을 해석할 수 없습니다(HTTP {response.status_code})"
            ) from None

        if payload.get("ok"):
            return payload.get("result")

        description = payload.get("description", "알 수 없는 오류")
        error_code = payload.get("error_code", response.status_code)
        raise TelegramError(f"{error_code}: {description}")

    def _request(self, method, json_body=None, files=None, data=None):
        """429/5xx는 재시도하고, 4xx는 즉시 실패합니다."""
        last_error = None

        for attempt in range(1, self.config.retries + 1):
            try:
                response = self.session.post(
                    self._url(method),
                    json=json_body,
                    files=files,
                    data=data,
                    timeout=self.config.timeout,
                )
            except Exception as error:      # 네트워크 오류
                last_error = TelegramError(
                    f"네트워크 오류: {type(error).__name__}"
                )

                if attempt < self.config.retries:
                    self.sleep(2 * attempt)

                continue

            if response.status_code == 429:
                retry_after = 3

                try:
                    retry_after = int(
                        response.json()
                        .get("parameters", {})
                        .get("retry_after", 3)
                    )
                except (ValueError, TypeError):
                    pass

                last_error = TelegramError("429: 요청이 너무 많습니다")

                if attempt < self.config.retries:
                    logger.warning(
                        "텔레그램 429 — %s초 후 재시도", retry_after
                    )
                    self.sleep(retry_after)

                continue

            if response.status_code >= 500:
                last_error = TelegramError(
                    f"{response.status_code}: 서버 오류"
                )

                if attempt < self.config.retries:
                    self.sleep(2 * attempt)

                continue

            return self._handle(response)

        raise last_error or TelegramError("전송에 실패했습니다")

    # -----------------------------------------------------
    # 공개 API
    # -----------------------------------------------------

    def find_chat_ids(self):
        """최근 메시지에서 chat_id 후보를 찾습니다(설정 도우미)."""
        updates = self._request("getUpdates", json_body={"limit": 20})

        found = []

        for update in updates or []:
            message = (
                update.get("message")
                or update.get("channel_post")
                or update.get("edited_message")
                or {}
            )
            chat = message.get("chat") or {}
            chat_id = chat.get("id")

            if chat_id is None:
                continue

            label = (
                chat.get("title")
                or chat.get("username")
                or chat.get("first_name")
                or ""
            )
            item = (str(chat_id), str(label))

            if item not in found:
                found.append(item)

        return found

    def send_message(self, text):
        """HTML 메시지를 (필요하면 여러 개로) 보냅니다."""
        if not self.config.configured:
            raise TelegramError("텔레그램 설정(토큰/chat_id)이 없습니다.")

        results = []

        for chunk in split_message(text):
            results.append(
                self._request(
                    "sendMessage",
                    json_body={
                        "chat_id": self.config.chat_id,
                        "text": chunk,
                        "parse_mode": "HTML",
                        "disable_web_page_preview": True,
                    },
                )
            )

        self.sent.append(("message", len(results)))
        return results

    def send_photo(self, photo, caption=""):
        """차트 이미지를 캡션과 함께 보냅니다."""
        if not self.config.configured:
            raise TelegramError("텔레그램 설정(토큰/chat_id)이 없습니다.")

        if isinstance(photo, (str, Path)):
            path = Path(photo)

            if not path.is_file():
                raise TelegramError(f"차트 파일이 없습니다: {path.name}")

            payload = (path.name, path.read_bytes(), "image/png")
        else:
            payload = ("chart.png", bytes(photo), "image/png")

        result = self._request(
            "sendPhoto",
            data={
                "chat_id": self.config.chat_id,
                "caption": caption[:1024],
                "parse_mode": "HTML",
            },
            files={"photo": payload},
        )

        self.sent.append(("photo", 1))
        return result


def photo_caption(row):
    """차트 사진에 붙일 짧은 캡션."""
    name = esc(row.get("종목명", "?"))
    signal = esc(row.get("신호") or "신호 없음")
    price = format_price(row.get("종가"))
    per = format_per(row.get("PER"))
    day = esc(row.get("신호일") or "")

    caption = f"<b>{name}</b> — {signal}\n종가 {price}원"

    if per != "-":
        caption += f" · PER {per}"

    if day:
        caption += f"\n신호일 {day}"

    return caption[:1024]


# =========================================================
# 배치 연동
# =========================================================

def notify_daily(
    records,
    run_date=None,
    base_dir=None,
    config=None,
    send_photos=True,
    notifier=None,
):
    """
    배치 결과를 텔레그램으로 보냅니다.

    Returns:
        dict — {"sent": bool, "messages": int, "photos": int,
                "error": str|None, "skipped": str|None}
    """
    result = {
        "sent": False,
        "messages": 0,
        "photos": 0,
        "error": None,
        "skipped": None,
    }

    config = config or TelegramConfig()

    if not config.configured:
        result["skipped"] = "텔레그램 설정이 없습니다."
        return result

    notifier = notifier or TelegramNotifier(config)

    text = build_summary(records, run_date=run_date)

    try:
        sent_messages = notifier.send_message(text)
        result["messages"] = len(sent_messages)
        result["sent"] = True

    except Exception as error:
        result["error"] = str(error)
        return result

    # 신호가 발생한 종목의 차트를 최대 N장 첨부
    if send_photos and config.max_photos > 0:
        candidates = [
            row for row in records or []
            if row.get("상태") == "완료"
            and row.get("신호") in (BUY_LABEL, SELL_LABEL)
            and row.get("차트")
        ]

        for row in candidates[: config.max_photos]:
            path = (Path(base_dir) if base_dir else Path(".")) / str(
                row["차트"]
            )

            try:
                notifier.send_photo(path, photo_caption(row))
                result["photos"] += 1

            except Exception as error:
                logger.warning(
                    "차트 전송 실패(%s): %s",
                    row.get("종목명"),
                    error,
                )

    return result
