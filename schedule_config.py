# schedule_config.py
"""
분석 실행 스케줄(주기·시각)을 관리합니다.

주기(period)
    hourly  매시간
    daily   매일
    weekly  매주 (요일 지정)

시간대
    모든 시각은 Asia/Seoul(KST, UTC+9, 서머타임 없음) 기준으로 저장합니다.

왜 '게이트' 방식인가
    GitHub Actions의 cron은 YAML에 고정값으로만 쓸 수 있어
    앱에서 시각을 바꿔도 워크플로가 따라오지 못합니다.
    그래서 워크플로는 매시간(:05) 한 번씩 깨어나 이 설정을 읽고,
    '지금이 실행할 시간인가'만 판정한 뒤 실제 분석을 돌립니다.
    (schedule_gate.py)

    - 일/주 주기: 시(hour) 단위로 판정합니다.
      GitHub cron은 수 분~수십 분 지연될 수 있어 '분'까지 맞추면
      실행을 놓치기 때문입니다. 분 값은 고정 cron을 쓸 때만 의미가 있습니다.
    - 같은 슬롯(예: 2026-11-19 20시)에서 두 번 실행되지 않도록
      마지막 실행 슬롯을 상태 파일에 기록합니다.
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

KST = ZoneInfo("Asia/Seoul")

CONFIG_VERSION = 1

PERIODS = ("hourly", "daily", "weekly")

PERIOD_LABELS = {
    "hourly": "매시간",
    "daily": "매일",
    "weekly": "매주",
}

WEEKDAY_NAMES = ("월", "화", "수", "목", "금", "토", "일")

DEFAULT_CONFIG = {
    "version": CONFIG_VERSION,
    "enabled": True,
    "period": "daily",
    "hour": 20,
    "minute": 1,
    "weekday": 0,          # 0=월 … 6=일 (주기=weekly일 때만 사용)
    "notify": True,        # 텔레그램 알림 사용
    "timezone": "Asia/Seoul",
    "updated_at": None,
    "note": "",
}


class ScheduleConfigError(ValueError):
    """설정 값이 올바르지 않을 때 발생합니다."""


# =========================================================
# 검증 / 정규화
# =========================================================

def _to_int(value, name, low, high):
    try:
        number = int(value)
    except (TypeError, ValueError):
        raise ScheduleConfigError(
            f"'{name}' 값이 숫자가 아닙니다: {value!r}"
        ) from None

    if not low <= number <= high:
        raise ScheduleConfigError(
            f"'{name}' 값은 {low}~{high} 범위여야 합니다: {number}"
        )

    return number


def _to_bool(value, default=True):
    if value is None:
        return default

    if isinstance(value, bool):
        return value

    text = str(value).strip().lower()

    if text in {"1", "true", "yes", "on", "y"}:
        return True

    if text in {"0", "false", "no", "off", "n", ""}:
        return False

    return default


def normalize_config(raw):
    """설정 dict를 검증해 새 dict로 돌려줍니다."""
    if raw is None:
        raw = {}

    if not isinstance(raw, dict):
        raise ScheduleConfigError("설정 형식이 올바르지 않습니다.")

    period = str(raw.get("period", DEFAULT_CONFIG["period"])).strip().lower()

    if period not in PERIODS:
        raise ScheduleConfigError(
            "주기는 hourly / daily / weekly 중 하나여야 합니다: "
            f"{period!r}"
        )

    timezone = str(
        raw.get("timezone") or DEFAULT_CONFIG["timezone"]
    ).strip()

    if timezone != "Asia/Seoul":
        # 저장은 KST 기준으로만 다룹니다(혼동 방지).
        timezone = "Asia/Seoul"

    config = {
        "version": CONFIG_VERSION,
        "enabled": _to_bool(raw.get("enabled"), True),
        "period": period,
        "hour": _to_int(raw.get("hour", DEFAULT_CONFIG["hour"]), "시", 0, 23),
        "minute": _to_int(
            raw.get("minute", DEFAULT_CONFIG["minute"]), "분", 0, 59
        ),
        "weekday": _to_int(
            raw.get("weekday", DEFAULT_CONFIG["weekday"]), "요일", 0, 6
        ),
        "notify": _to_bool(raw.get("notify"), True),
        "timezone": timezone,
        "updated_at": raw.get("updated_at"),
        "note": str(raw.get("note") or "")[:200],
    }

    return config


# =========================================================
# 저장 / 읽기
# =========================================================

def load_config(path, missing_ok=True):
    """설정 파일을 읽습니다. 없으면 기본값을 돌려줍니다."""
    path = Path(path)

    if not path.is_file():
        if missing_ok:
            return dict(DEFAULT_CONFIG)
        raise ScheduleConfigError(
            f"스케줄 설정 파일이 없습니다: {path.name}"
        )

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ScheduleConfigError(
            f"스케줄 설정을 읽지 못했습니다: {type(error).__name__}"
        ) from None

    return normalize_config(raw)


def save_config(path, config, touch=True):
    """설정을 원자적으로 저장합니다."""
    config = normalize_config(config)

    if touch:
        config["updated_at"] = datetime.now(KST).isoformat(
            timespec="seconds"
        )

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"

    try:
        temporary.write_text(
            json.dumps(config, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)

    return config


# =========================================================
# 판정 / 표시
# =========================================================

def as_kst(moment=None):
    """datetime을 KST aware 값으로 바꿉니다."""
    if moment is None:
        return datetime.now(KST)

    if isinstance(moment, datetime):
        if moment.tzinfo is None:
            return moment.replace(tzinfo=KST)

        return moment.astimezone(KST)

    raise ScheduleConfigError("시각 형식이 올바르지 않습니다.")


def describe(config):
    """'매일 20:01 (KST)' 같은 사람용 요약."""
    config = normalize_config(config)
    period = config["period"]

    if period == "hourly":
        return f"매시간 {config['minute']:02d}분 (KST)"

    clock = f"{config['hour']:02d}:{config['minute']:02d}"

    if period == "daily":
        return f"매일 {clock} (KST)"

    return (
        f"매주 {WEEKDAY_NAMES[config['weekday']]}요일 "
        f"{clock} (KST)"
    )


def slot_key(config, moment=None):
    """
    같은 실행을 두 번 하지 않기 위한 슬롯 키.

    일/주 주기는 '시' 단위로 판정하므로 슬롯도 날짜 기준입니다.
    """
    config = normalize_config(config)
    moment = as_kst(moment)

    if config["period"] == "hourly":
        return moment.strftime("%Y-%m-%dT%H")

    return moment.strftime("%Y-%m-%d")


def is_due(config, moment=None):
    """
    '지금 실행할 시간인가'를 판정합니다(슬롯 중복은 고려하지 않음).

    Returns:
        (due: bool, reason: str)
    """
    config = normalize_config(config)
    moment = as_kst(moment)

    if not config["enabled"]:
        return False, "스케줄이 꺼져 있습니다."

    period = config["period"]

    if period == "hourly":
        # 매시간 주기이므로 항상 실행 대상입니다(슬롯은 시 단위).
        return True, "매시간 주기"

    if period == "daily":
        if moment.hour != config["hour"]:
            return (
                False,
                f"오늘 실행 시각은 {config['hour']:02d}시입니다"
                f"(현재 {moment.hour:02d}시).",
            )

        return True, "매일 주기 — 실행 시각 도달"

    if period == "weekly":
        if moment.weekday() != config["weekday"]:
            return (
                False,
                f"{WEEKDAY_NAMES[config['weekday']]}요일에 실행합니다"
                f"(오늘은 {WEEKDAY_NAMES[moment.weekday()]}요일).",
            )

        if moment.hour != config["hour"]:
            return (
                False,
                f"실행 시각은 {config['hour']:02d}시입니다"
                f"(현재 {moment.hour:02d}시).",
            )

        return True, "매주 주기 — 실행 시각 도달"

    return False, "알 수 없는 주기입니다."


def is_due_with_state(config, moment=None, last_slot=None):
    """슬롯 중복까지 고려한 최종 판정."""
    config = normalize_config(config)
    moment = as_kst(moment)

    due, reason = is_due(config, moment)

    if not due:
        return False, reason, slot_key(config, moment)

    slot = slot_key(config, moment)

    if last_slot == slot:
        return (
            False,
            f"이미 이 슬롯({slot})에서 실행했습니다.",
            slot,
        )

    return True, reason, slot


def next_runs(config, moment=None, count=3):
    """다음 실행 예정 시각을 count개 계산합니다."""
    config = normalize_config(config)
    moment = as_kst(moment)
    results = []

    if not config["enabled"]:
        return results

    period = config["period"]
    minute = config["minute"]

    if period == "hourly":
        cursor = moment.replace(minute=minute, second=0, microsecond=0)

        if cursor <= moment:
            cursor += timedelta(hours=1)

        for _ in range(count):
            results.append(cursor)
            cursor += timedelta(hours=1)

        return results

    step = timedelta(days=1)
    cursor = moment.replace(
        hour=config["hour"], minute=minute, second=0, microsecond=0
    )

    if cursor <= moment:
        cursor += step

    if period == "daily":
        for _ in range(count):
            results.append(cursor)
            cursor += step

        return results

    # weekly: 지정 요일로 맞춘 뒤 7일 간격
    while cursor.weekday() != config["weekday"] or cursor <= moment:
        cursor += step

    for _ in range(count):
        results.append(cursor)
        cursor += timedelta(days=7)

    return results


# =========================================================
# cron 변환 (고정 cron을 쓰고 싶을 때)
# =========================================================

def cron_parts(config):
    """
    KST 설정을 UTC 기준 cron 필드로 변환합니다.

    Returns:
        dict — minute, hour, weekday(또는 None), day_shift
    """
    config = normalize_config(config)

    # 2026-01-05는 월요일 → weekday(0=월)를 그대로 더하면 원하는 요일
    moment = datetime(
        2026,
        1,
        5 + config["weekday"],
        config["hour"],
        config["minute"],
        tzinfo=KST,
    )

    utc = moment.astimezone(ZoneInfo("UTC"))
    day_shift = (utc.date() - moment.date()).days

    return {
        "minute": utc.minute,
        "hour": utc.hour,
        "weekday": (utc.weekday() + 1) % 7,  # cron: 0=일요일
        "day_shift": day_shift,
        "kst": moment,
        "utc": utc,
    }


def cron_expression(config):
    """정확한 시각으로 고정 cron을 쓸 때의 표현식(UTC)."""
    config = normalize_config(config)
    parts = cron_parts(config)

    if config["period"] == "hourly":
        return f"{parts['minute']} * * * *"

    if config["period"] == "daily":
        return f"{parts['minute']} {parts['hour']} * * *"

    return f"{parts['minute']} {parts['hour']} * * {parts['weekday']}"


def utc_note(config):
    """KST → UTC 변환 설명 문장."""
    config = normalize_config(config)

    if config["period"] == "hourly":
        return (
            f"매시간 {config['minute']:02d}분 → cron "
            f"\"{config['minute']} * * * *\""
        )

    parts = cron_parts(config)
    kst = parts["kst"]
    utc = parts["utc"]

    note = (
        f"KST {kst:%Y-%m-%d(%a) %H:%M} = "
        f"UTC {utc:%Y-%m-%d(%a) %H:%M}"
    )

    if parts["day_shift"]:
        note += f" (UTC 날짜가 {parts['day_shift']:+d}일)"

    return note
