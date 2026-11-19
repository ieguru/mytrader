# per_status.py
"""
PER 계산을 app.py / batch_daily.py가 함께 사용합니다.

PER = 종가 × 상장주식수 ÷ (4Q_income × 100,000,000)

단위
    상장주식수: 주
    4Q_income: 억원

원칙
- 기준일까지 '종료된' 분기 중 가장 최근 행만 사용합니다.
  (엑셀에 미래 분기가 들어 있어도 PER에 쓰지 않습니다)
- 가장 최근 종료 분기의 값이 비었으면 더 오래된 분기로 대체하지 않습니다.
"""

from __future__ import annotations

import math
import re

import pandas as pd

PER_QUARTER_PATTERN = r"\d{4}\.[1-4]Q"

PER_KEYS = (
    "PER",
    "PER 분기",
    "PER 상장주식수",
    "PER 4Q_income",
    "PER 설명",
)


def to_finite_number(value):
    """숫자 또는 쉼표 포함 문자열을 유한한 float로 변환합니다."""
    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    if isinstance(value, bool):
        return None

    try:
        number = float(
            str(value).replace(",", "").strip()
            if isinstance(value, str)
            else value
        )
    except (TypeError, ValueError, OverflowError):
        return None

    return number if math.isfinite(number) else None


def empty_per_status():
    return {
        "PER": None,
        "PER 분기": None,
        "PER 상장주식수": None,
        "PER 4Q_income": None,
        "PER 설명": "",
    }


def read_per_financial(excel_path, stock_name):
    """
    PER 계산용 재무 표를 읽습니다.

    Returns:
        DataFrame — 분기, 상장주식수, 4Q_income, _분기종료일
        (_분기종료일 오름차순)
    """
    with pd.ExcelFile(excel_path) as workbook:
        if stock_name not in workbook.sheet_names:
            raise ValueError(
                f"엑셀에 '{stock_name}' 시트가 없습니다."
            )

        frame = pd.read_excel(
            workbook,
            sheet_name=stock_name,
        )

    frame = frame.dropna(how="all").copy()
    frame.columns = [
        str(column).strip()
        for column in frame.columns
    ]

    if frame.columns.duplicated().any():
        raise ValueError(
            "PER 재무 데이터에 중복된 컬럼이 있습니다."
        )

    required = {"분기", "상장주식수", "4Q_income"}
    missing = required - set(frame.columns)

    if missing:
        raise ValueError(
            "PER 계산에 필요한 엑셀 컬럼이 없습니다: "
            + ", ".join(sorted(missing))
        )

    frame["분기"] = (
        frame["분기"]
        .astype("string")
        .str.strip()
    )

    valid_quarter = frame["분기"].str.fullmatch(
        PER_QUARTER_PATTERN,
        na=False,
    )

    if not valid_quarter.all():
        raise ValueError(
            "PER 재무 데이터의 분기는 "
            "'2026.1Q' 형식이어야 합니다."
        )

    if frame["분기"].duplicated().any():
        raise ValueError(
            "PER 재무 데이터에 중복된 분기가 있습니다."
        )

    quarter_ends = []

    for quarter_label in frame["분기"]:
        matched = re.fullmatch(
            r"(\d{4})\.([1-4])Q",
            str(quarter_label),
        )

        year = int(matched.group(1))
        quarter = int(matched.group(2))

        quarter_ends.append(
            pd.Period(
                f"{year}Q{quarter}",
                freq="Q-DEC",
            ).end_time.normalize()
        )

    frame["_분기종료일"] = pd.to_datetime(
        quarter_ends
    )

    for column in ("상장주식수", "4Q_income"):
        frame[column] = frame[column].map(
            to_finite_number
        )

    return (
        frame
        .sort_values("_분기종료일")
        .reset_index(drop=True)
    )


def build_per_status(financial, reference, close_price):
    """
    PER 상태를 계산합니다.

    Args:
        financial: read_per_financial의 결과
        reference: 기준일(Timestamp)
        close_price: 기준일 종가
    """
    status = empty_per_status()

    if financial is None or financial.empty:
        status["PER 설명"] = "PER 재무 데이터가 없습니다."
        return status

    completed = financial.loc[
        financial["_분기종료일"] <= reference
    ]

    if completed.empty:
        status["PER 설명"] = (
            "기준일까지 종료된 분기 데이터가 없습니다."
        )
        return status

    row = completed.iloc[-1]

    shares = to_finite_number(row["상장주식수"])
    income = to_finite_number(row["4Q_income"])

    status.update({
        "PER 분기": str(row["분기"]),
        "PER 상장주식수": shares,
        "PER 4Q_income": income,
    })

    if shares is None or shares <= 0:
        status["PER 설명"] = (
            f'{row["분기"]}: 상장주식수가 없거나 0 이하입니다.'
        )
        return status

    if not float(shares).is_integer():
        status["PER 설명"] = (
            f'{row["분기"]}: 상장주식수는 주 단위 정수여야 합니다.'
        )
        return status

    if income is None or income <= 0:
        status["PER 설명"] = (
            f'{row["분기"]}: 4Q_income이 없거나 0 이하입니다.'
        )
        return status

    per_value = (
        close_price
        * shares
        / (income * 100_000_000)
    )

    if not math.isfinite(per_value):
        status["PER 설명"] = "PER 계산 결과가 유효하지 않습니다."
        return status

    status["PER"] = per_value
    status["PER 설명"] = (
        f'{row["분기"]} 엑셀 값 기준 · 4Q_income 단위: 억원'
    )

    return status
