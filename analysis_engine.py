from __future__ import annotations

from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd


# ============================================================
# 기본 설정
# ============================================================

RECENT_SIGNAL_DAYS = 7

MACD_SETTINGS = {
    "Daily": (12, 26, 9),
    "Weekly": (5, 13, 5),
    "Monthly": (3, 7, 3),
}

PRICE_COLUMNS = [
    "시가",
    "고가",
    "저가",
    "종가",
    "거래량",
]

FINANCIAL_COLUMNS = [
    "분기",
    "목표가",
    "매출액",
    "영업이익",
    "순이익",
]

QUARTER_PATTERN = re.compile(r"^\d{4}\.[1-4]Q$")

TRADE_COLUMNS = [
    "매수일",
    "매도일",
    "매수가",
    "매도가",
    "수익률(%)",
    "수익(원)",
    "보유기간(일)",
]


# ============================================================
# 공통 유틸리티
# ============================================================

def to_numeric(series: pd.Series) -> pd.Series:
    """
    쉼표가 있는 숫자 문자열을 포함하여 숫자형으로 변환합니다.
    무한대는 NaN으로 처리합니다.
    """
    cleaned = (
        series.astype("string")
        .str.strip()
        .str.replace(",", "", regex=False)
    )

    result = pd.to_numeric(
        cleaned,
        errors="coerce",
    )

    return result.astype(float).replace(
        [np.inf, -np.inf],
        np.nan,
    )


def prepare_price_frame(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    가격 데이터 정리.

    - 날짜 인덱스 검증
    - 한국 시간 기준 날짜 정규화
    - 중복 날짜는 마지막 행 유지
    - 가격 필수 컬럼 숫자 변환
    - 필수값 누락 및 종가 <= 0 행 제외

    투자자 순매수 등 추가 컬럼은 유지합니다.
    """
    if data is None or data.empty:
        raise ValueError("가격 데이터가 없습니다.")

    frame = data.copy()

    if frame.columns.duplicated().any():
        raise ValueError(
            "가격 데이터에 중복된 컬럼이 있습니다."
        )

    missing = [
        column
        for column in PRICE_COLUMNS
        if column not in frame.columns
    ]

    if missing:
        raise ValueError(
            "가격 데이터 컬럼 누락: "
            + ", ".join(missing)
        )

    if not isinstance(frame.index, pd.DatetimeIndex):
        if "날짜" in frame.columns:
            dates = pd.to_datetime(
                frame["날짜"],
                errors="coerce",
            )
        else:
            if pd.api.types.is_numeric_dtype(
                frame.index.dtype
            ):
                raise ValueError(
                    "숫자형 인덱스는 날짜로 사용할 수 없습니다."
                )

            dates = pd.to_datetime(
                frame.index,
                errors="coerce",
            )

        frame.index = pd.DatetimeIndex(dates)

    valid_dates = ~frame.index.isna()
    frame = frame.loc[valid_dates].copy()

    if frame.empty:
        raise ValueError(
            "유효한 날짜 데이터가 없습니다."
        )

    if frame.index.tz is not None:
        frame.index = (
            frame.index
            .tz_convert("Asia/Seoul")
            .tz_localize(None)
        )

    frame.index = frame.index.normalize()

    frame = frame.loc[
        ~frame.index.duplicated(keep="last")
    ].sort_index()

    for column in PRICE_COLUMNS:
        frame[column] = to_numeric(frame[column])

    frame = frame.dropna(
        subset=PRICE_COLUMNS
    )

    frame = frame.loc[
        frame["종가"].gt(0)
    ].copy()

    if frame.empty:
        raise ValueError(
            "분석할 수 있는 유효한 가격 데이터가 없습니다."
        )

    frame.index.name = "날짜"

    return frame


# ============================================================
# 재무 엑셀
# ============================================================

def read_financial_excel(
    excel_file: str | Path,
    stock_name: str,
) -> pd.DataFrame:
    """
    재무 엑셀 읽기.

    필수 컬럼:
        분기, 목표가, 매출액, 영업이익, 순이익

    추가 컬럼:
        상장주식수, 4Q_income 등이 있으면 유지합니다.

    재무 금액은 엑셀 입력 단위를 그대로 사용합니다.
    """
    path = Path(excel_file)

    if not path.is_file():
        raise FileNotFoundError(
            f"재무 파일이 없습니다: {path.name}"
        )

    with pd.ExcelFile(path) as workbook:
        if stock_name not in workbook.sheet_names:
            raise ValueError(
                f"엑셀에 '{stock_name}' 시트가 없습니다."
            )

        # pandas가 중복 헤더를 자동 변경하기 전에 검사합니다.
        header = pd.read_excel(
            workbook,
            sheet_name=stock_name,
            header=None,
            nrows=1,
        )

        if header.empty:
            raise ValueError(
                "재무 엑셀에 헤더가 없습니다."
            )

        header_values = [
            str(value).strip()
            for value in header.iloc[0]
            if pd.notna(value)
        ]

        if len(header_values) != len(set(header_values)):
            raise ValueError(
                "재무 엑셀에 중복된 컬럼이 있습니다."
            )

        financial = pd.read_excel(
            workbook,
            sheet_name=stock_name,
        )

    financial.columns = [
        str(column).strip()
        for column in financial.columns
    ]

    financial = financial.dropna(
        how="all"
    ).copy()

    missing = [
        column
        for column in FINANCIAL_COLUMNS
        if column not in financial.columns
    ]

    if missing:
        raise ValueError(
            "엑셀에 필요한 컬럼이 없습니다: "
            + ", ".join(missing)
        )

    financial["분기"] = (
        financial["분기"]
        .astype("string")
        .str.strip()
    )

    valid_quarters = (
        financial["분기"]
        .str.fullmatch(QUARTER_PATTERN)
        .fillna(False)
    )

    if not valid_quarters.all():
        raise ValueError(
            "분기는 '2026.1Q' 형식으로 입력하세요."
        )

    if financial["분기"].duplicated().any():
        raise ValueError(
            "재무 엑셀에 중복된 분기가 있습니다."
        )

    numeric_columns = [
        "목표가",
        "매출액",
        "영업이익",
        "순이익",
        "상장주식수",
        "4Q_income",
    ]

    for column in numeric_columns:
        if column not in financial.columns:
            continue

        original = (
            financial[column]
            .astype("string")
            .str.strip()
        )

        nonempty = (
            original.notna()
            & original.ne("")
        ).fillna(False)

        numeric = to_numeric(
            financial[column]
        )

        invalid = nonempty & numeric.isna()

        if invalid.any():
            raise ValueError(
                f"'{column}'에는 숫자 또는 빈칸만 입력하세요."
            )

        financial[column] = numeric

    # 목표가: 음수는 0, 100원 단위 반올림.
    target = financial["목표가"].clip(
        lower=0
    )

    financial["목표가"] = (
        np.floor(target / 100.0 + 0.5) * 100.0
    )

    quarter_strings = (
        financial["분기"]
        .str.replace(".", "", regex=False)
        .str.replace(
            r"^(\d{4})([1-4])Q$",
            r"\1Q\2",
            regex=True,
        )
    )

    financial["_분기"] = pd.PeriodIndex(
        quarter_strings.tolist(),
        freq="Q-DEC",
    )

    return financial.sort_values(
        "_분기"
    ).reset_index(drop=True)


def validate_excel(
    excel_file: str | Path,
    stock_name: str,
) -> bool:
    """
    유효하지 않으면 예외를 발생시킵니다.
    """
    read_financial_excel(
        excel_file,
        stock_name,
    )

    return True


class FinancialDataManager:
    @staticmethod
    def load_financial_data(
        excel_file: str | Path,
        stock_name: str,
    ) -> pd.DataFrame:
        return read_financial_excel(
            excel_file,
            stock_name,
        )

    @staticmethod
    def add_financial_data(
        df: pd.DataFrame,
        financial: pd.DataFrame | None,
    ) -> pd.DataFrame:
        """
        해당 날짜가 속한 분기의 재무값을 매핑합니다.

        주의:
            공시일 기준 매핑이 아닙니다.
            아래 재무값은 MACD 신호·백테스트 조건에 사용하지 않습니다.
        """
        frame = df.copy()

        frame["분기"] = [
            f"{date.year}.{date.quarter}Q"
            for date in frame.index
        ]

        mapped_columns = [
            "목표가",
            "매출액",
            "영업이익",
            "순이익",
        ]

        if financial is None:
            for column in mapped_columns:
                frame[column] = np.nan
        else:
            lookup = financial.set_index("분기")

            optional_columns = [
                column
                for column in (
                    "상장주식수",
                    "4Q_income",
                )
                if column in lookup.columns
            ]

            for column in (
                mapped_columns + optional_columns
            ):
                frame[column] = (
                    frame["분기"]
                    .map(lookup[column])
                )

        # 양수: 종가가 목표가보다 높음.
        # 음수: 종가가 목표가보다 낮음.
        valid_target = frame["목표가"].where(
            frame["목표가"].gt(0)
        )

        frame["괴리율(%)"] = (
            (frame["종가"] / valid_target - 1.0)
            * 100.0
        )

        return frame


# ============================================================
# MACD 계산
# ============================================================

def calculate_native_macd(
    prices: pd.Series,
    fast: int,
    slow: int,
    signal: int,
) -> pd.DataFrame:
    """
    입력된 봉의 단위 그대로 EMA와 MACD를 계산합니다.

    초기값:
        첫 EMA = 첫 종가
        첫 Signal = 첫 MACD

    별도의 워밍업 구간을 NaN으로 만들지는 않습니다.
    """
    if not (0 < fast < slow and signal > 0):
        raise ValueError(
            "MACD 기간은 0 < fast < slow, signal > 0이어야 합니다."
        )

    ema_fast = prices.ewm(
        span=fast,
        adjust=False,
        min_periods=1,
    ).mean()

    ema_slow = prices.ewm(
        span=slow,
        adjust=False,
        min_periods=1,
    ).mean()

    macd = ema_fast - ema_slow

    signal_line = macd.ewm(
        span=signal,
        adjust=False,
        min_periods=1,
    ).mean()

    return pd.DataFrame(
        {
            "EMA_Fast": ema_fast,
            "EMA_Slow": ema_slow,
            "MACD": macd,
            "Signal": signal_line,
            "Oscillator": macd - signal_line,
        }
    )


def calculate_macd_by_timeframe(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    일간·주간·월간 MACD 계산.

    일간:
        일별 종가 사용.

    주간:
        이전 주까지의 주봉 확정 상태에
        당일 종가를 이번 주의 임시 종가로 넣어 계산.

    월간:
        이전 달까지의 월봉 확정 상태에
        당일 종가를 이번 달의 임시 종가로 넣어 계산.

    같은 주·월의 전일 임시 EMA를 다시 누적하지 않습니다.
    """
    if df is None or df.empty:
        raise ValueError(
            "MACD를 계산할 데이터가 없습니다."
        )

    if not isinstance(
        df.index,
        pd.DatetimeIndex,
    ):
        raise ValueError(
            "MACD 계산에는 날짜 인덱스가 필요합니다."
        )

    if (
        df.index.hasnans
        or df.index.has_duplicates
        or not df.index.is_monotonic_increasing
    ):
        raise ValueError(
            "MACD 날짜는 결측·중복 없이 오름차순이어야 합니다."
        )

    if df.index.tz is not None:
        raise ValueError(
            "prepare_price_frame()으로 시간대를 먼저 정리하세요."
        )

    if "종가" not in df.columns:
        raise ValueError(
            "MACD 계산에 필요한 '종가' 컬럼이 없습니다."
        )

    frame = df.copy()
    close = to_numeric(frame["종가"])

    if (
        close.isna().any()
        or close.le(0).any()
    ):
        raise ValueError(
            "종가에 결측값 또는 0 이하 값이 있습니다."
        )

    # --------------------------------------------------------
    # 일간
    # --------------------------------------------------------
    fast, slow, signal = MACD_SETTINGS["Daily"]

    daily = calculate_native_macd(
        close,
        fast,
        slow,
        signal,
    )

    for column in (
        "EMA_Fast",
        "EMA_Slow",
        "MACD",
        "Signal",
        "Oscillator",
    ):
        frame[f"Daily_{column}"] = daily[column]

    frame["Daily_Prev_Oscillator"] = (
        daily["Oscillator"].shift(1)
    )

    frame["Daily_Oscillator_Change"] = (
        daily["Oscillator"].diff()
    )

    frame["Daily_Prev_Oscillator_Change"] = (
        daily["Oscillator"].diff().shift(1)
    )

    # --------------------------------------------------------
    # 주간 / 월간
    # --------------------------------------------------------
    def add_period_macd(
        prefix: str,
        frequency: str,
    ) -> None:
        fast, slow, signal = MACD_SETTINGS[prefix]

        period_keys = (
            frame.index.to_period(frequency)
        )

        # 각 기간의 마지막 가용 종가.
        period_close = close.groupby(
            period_keys,
            sort=True,
        ).last()

        native = calculate_native_macd(
            period_close,
            fast,
            slow,
            signal,
        )

        # 이전 기간의 증감:
        # 예) 전주 Oscillator - 전전주 Oscillator
        native["Oscillator_Change"] = (
            native["Oscillator"].diff()
        )

        # 현재 기간의 확정값은 사용하지 않습니다.
        # 반드시 이전 기간 값만 현재 기간에 매핑합니다.
        previous = native.shift(1)

        def map_previous(
            column: str,
        ) -> pd.Series:
            return pd.Series(
                period_keys.map(
                    previous[column]
                ),
                index=frame.index,
                dtype=float,
            )

        prev_fast = map_previous("EMA_Fast")
        prev_slow = map_previous("EMA_Slow")
        prev_signal = map_previous("Signal")

        prev_oscillator = map_previous(
            "Oscillator"
        )

        prev_change = map_previous(
            "Oscillator_Change"
        )

        alpha_fast = 2.0 / (fast + 1.0)
        alpha_slow = 2.0 / (slow + 1.0)
        alpha_signal = 2.0 / (signal + 1.0)

        # 현재 기간을 한 번만 반영하는 임시 EMA.
        current_fast = (
            (1.0 - alpha_fast) * prev_fast
            + alpha_fast * close
        ).where(
            prev_fast.notna(),
            close,
        )

        current_slow = (
            (1.0 - alpha_slow) * prev_slow
            + alpha_slow * close
        ).where(
            prev_slow.notna(),
            close,
        )

        current_macd = (
            current_fast - current_slow
        )

        # Signal도 주·월 단위로 한 번만 갱신.
        current_signal = (
            (1.0 - alpha_signal) * prev_signal
            + alpha_signal * current_macd
        ).where(
            prev_signal.notna(),
            current_macd,
        )

        current_oscillator = (
            current_macd - current_signal
        )

        frame[f"{prefix}_EMA_Fast"] = (
            current_fast
        )

        frame[f"{prefix}_EMA_Slow"] = (
            current_slow
        )

        frame[f"{prefix}_MACD"] = (
            current_macd
        )

        frame[f"{prefix}_Signal"] = (
            current_signal
        )

        frame[f"{prefix}_Oscillator"] = (
            current_oscillator
        )

        frame[f"{prefix}_Prev_Oscillator"] = (
            prev_oscillator
        )

        frame[f"{prefix}_Oscillator_Change"] = (
            current_oscillator
            - prev_oscillator
        )

        frame[
            f"{prefix}_Prev_Oscillator_Change"
        ] = prev_change

    add_period_macd(
        prefix="Weekly",
        frequency="W-FRI",
    )

    add_period_macd(
        prefix="Monthly",
        frequency="M",
    )

    return frame


class TechnicalIndicator:
    @staticmethod
    def calculate_ema(
        series: pd.Series,
        span: int,
    ) -> pd.Series:
        return series.ewm(
            span=span,
            adjust=False,
            min_periods=1,
        ).mean()

    @staticmethod
    def calculate_macd(
        prices: pd.Series,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
    ) -> pd.DataFrame:
        return calculate_native_macd(
            prices,
            fast,
            slow,
            signal,
        )

    @staticmethod
    def add_macd(
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        return calculate_macd_by_timeframe(df)


# ============================================================
# 신호 생성
# ============================================================

def direction_from_change(
    change: pd.Series,
) -> pd.Series:
    """
    증가: 1
    감소: -1
    동일 또는 비교 불가: 0
    """
    result = pd.Series(
        0,
        index=change.index,
        dtype="int8",
    )

    result.loc[change.gt(0)] = 1
    result.loc[change.lt(0)] = -1

    return result


def first_true(
    condition: pd.Series,
) -> pd.Series:
    """
    연속된 True 구간의 첫날만 True.

    True, True, False, True
      -> True, False, False, True
    """
    condition = (
        condition.fillna(False).astype(bool)
    )

    return (
        condition
        & ~condition.shift(
            1,
            fill_value=False,
        )
    )


class SignalGenerator:
    @staticmethod
    def generate_signals(
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        frame = df.copy()

        required = [
            "Daily_Oscillator_Change",
            "Weekly_Oscillator_Change",
            "Monthly_Oscillator_Change",
            "Weekly_Prev_Oscillator_Change",
        ]

        missing = [
            column
            for column in required
            if column not in frame.columns
        ]

        if missing:
            raise ValueError(
                "신호 계산 컬럼 누락: "
                + ", ".join(missing)
            )

        # 각 시간 단위의 이전 봉 대비 방향.
        for prefix in (
            "Daily",
            "Weekly",
            "Monthly",
        ):
            frame[f"{prefix}_Trend"] = (
                direction_from_change(
                    frame[
                        f"{prefix}_Oscillator_Change"
                    ]
                )
            )

        weekly_change = (
            frame["Weekly_Oscillator_Change"]
        )

        monthly_change = (
            frame["Monthly_Oscillator_Change"]
        )

        previous_week_change = frame[
            "Weekly_Prev_Oscillator_Change"
        ]

        # ----------------------------------------------------
        # 주간 단독 신호
        #
        # 전주의 증감 방향과 이번 주의 증감 방향 비교.
        # 엄격한 음수 -> 양수 / 양수 -> 음수 조건.
        # ----------------------------------------------------
        weekly_buy_condition = (
            previous_week_change.lt(0)
            & weekly_change.gt(0)
        )

        weekly_sell_condition = (
            previous_week_change.gt(0)
            & weekly_change.lt(0)
        )

        frame["Weekly_Buy"] = first_true(
            weekly_buy_condition
        )

        frame["Weekly_Sell"] = first_true(
            weekly_sell_condition
        )

        # ----------------------------------------------------
        # 주간·월간 결합 조건
        #
        # 매수:
        #   이번 주 > 전주 AND 이번 달 > 전월
        #
        # 매도:
        #   이번 주 < 전주 AND 이번 달 < 전월
        # ----------------------------------------------------
        comparable = (
            weekly_change.notna()
            & monthly_change.notna()
        )

        buy_condition = (
            comparable
            & weekly_change.gt(0)
            & monthly_change.gt(0)
        )

        sell_condition = (
            comparable
            & weekly_change.lt(0)
            & monthly_change.lt(0)
        )

        # 조건과 이벤트를 분리해 보관.
        frame[
            "Weekly_Monthly_Buy_Condition"
        ] = buy_condition

        frame[
            "Weekly_Monthly_Sell_Condition"
        ] = sell_condition

        # 실제 표시·백테스트용 이벤트:
        # 같은 조건이 연속되면 최초 1회만 발생.
        frame["Weekly_Monthly_Buy"] = (
            first_true(buy_condition)
        )

        frame["Weekly_Monthly_Sell"] = (
            first_true(sell_condition)
        )

        return frame


# ============================================================
# 신호 조회
# ============================================================

def classify_signal(
    row: pd.Series,
    buy_col: str,
    sell_col: str,
) -> str:
    buy = row.get(buy_col, False)
    sell = row.get(sell_col, False)

    buy = False if pd.isna(buy) else bool(buy)
    sell = False if pd.isna(sell) else bool(sell)

    if buy and sell:
        raise ValueError(
            "같은 날짜에 매수·매도 신호가 동시에 있습니다."
        )

    if buy:
        return "매수"

    if sell:
        return "매도"

    return "신호 없음"


def get_recent_signal(
    df: pd.DataFrame,
    buy_col: str,
    sell_col: str,
    days: int = RECENT_SIGNAL_DAYS,
) -> dict[str, Any]:
    """
    마지막 가격 날짜 포함 최근 days일(달력일)의 마지막 신호.
    """
    if (
        isinstance(days, bool)
        or not isinstance(days, (int, np.integer))
        or days < 1
    ):
        raise ValueError(
            "신호 조회 일수는 1 이상이어야 합니다."
        )

    missing = [
        column
        for column in (buy_col, sell_col)
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            "신호 컬럼 누락: "
            + ", ".join(missing)
        )

    if df.empty:
        return {
            "신호": "신호 없음",
            "발생일": None,
        }

    reference_date = df.index[-1].normalize()

    start_date = (
        reference_date
        - pd.Timedelta(days=days - 1)
    )

    recent = df.loc[
        (df.index >= start_date)
        & (df.index <= reference_date)
    ]

    mask = (
        recent[buy_col].fillna(False).astype(bool)
        | recent[sell_col].fillna(False).astype(bool)
    )

    recent = recent.loc[mask]

    if recent.empty:
        return {
            "신호": "신호 없음",
            "발생일": None,
        }

    last_row = recent.iloc[-1]

    return {
        "신호": classify_signal(
            last_row,
            buy_col,
            sell_col,
        ),
        "발생일": recent.index[-1].strftime(
            "%Y-%m-%d"
        ),
    }


def recent_signals(
    df: pd.DataFrame,
    buy_col: str,
    sell_col: str,
    limit: int = 3,
) -> pd.DataFrame:
    """
    전체 분석 기간의 마지막 limit개 신호.
    반환 순서는 날짜 오름차순입니다.
    """
    if (
        isinstance(limit, bool)
        or not isinstance(limit, (int, np.integer))
        or limit < 1
    ):
        raise ValueError(
            "신호 표시 개수는 1 이상이어야 합니다."
        )

    missing = [
        column
        for column in (
            buy_col,
            sell_col,
            "종가",
        )
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            "신호 조회 컬럼 누락: "
            + ", ".join(missing)
        )

    mask = (
        df[buy_col].fillna(False).astype(bool)
        | df[sell_col].fillna(False).astype(bool)
    )

    selected = df.loc[mask].tail(limit)

    columns = [
        "날짜",
        "신호",
        "종가",
        "목표가",
        "괴리율(%)",
        "Weekly_Trend",
        "Monthly_Trend",
    ]

    records = []

    for date, row in selected.iterrows():
        records.append(
            {
                "날짜": date.strftime("%Y-%m-%d"),
                "신호": classify_signal(
                    row,
                    buy_col,
                    sell_col,
                ),
                "종가": row["종가"],
                "목표가": row.get(
                    "목표가",
                    np.nan,
                ),
                "괴리율(%)": row.get(
                    "괴리율(%)",
                    np.nan,
                ),
                "Weekly_Trend": row.get(
                    "Weekly_Trend",
                    0,
                ),
                "Monthly_Trend": row.get(
                    "Monthly_Trend",
                    0,
                ),
            }
        )

    return pd.DataFrame(
        records,
        columns=columns,
    )


# ============================================================
# 백테스트
# ============================================================

class Backtester:
    @staticmethod
    def run_backtest(
        df: pd.DataFrame,
        buy_col: str,
        sell_col: str,
    ) -> pd.DataFrame:
        """
        단일 포지션 백테스트.

        - 신호일 종가 매매
        - 미보유일 때만 매수
        - 보유 중일 때만 매도
        - 추가 매수 없음
        - 미청산 거래 제외
        - 수수료·세금·슬리피지 제외

        수익(원)은 1주 거래 기준입니다.
        """
        missing = [
            column
            for column in (
                "종가",
                buy_col,
                sell_col,
            )
            if column not in df.columns
        ]

        if missing:
            raise ValueError(
                "백테스트 컬럼 누락: "
                + ", ".join(missing)
            )

        holding = False
        entry_date = None
        entry_price = None

        records = []

        for date, row in df.iterrows():
            signal = classify_signal(
                row,
                buy_col,
                sell_col,
            )

            price = float(row["종가"])

            if signal == "매수" and not holding:
                holding = True
                entry_date = date
                entry_price = price
                continue

            if signal == "매도" and holding:
                profit = price - entry_price

                return_pct = (
                    price / entry_price - 1.0
                ) * 100.0

                records.append(
                    {
                        "매수일": entry_date.strftime(
                            "%Y-%m-%d"
                        ),
                        "매도일": date.strftime(
                            "%Y-%m-%d"
                        ),
                        "매수가": entry_price,
                        "매도가": price,
                        "수익률(%)": return_pct,
                        "수익(원)": profit,
                        "보유기간(일)": (
                            date - entry_date
                        ).days,
                    }
                )

                holding = False
                entry_date = None
                entry_price = None

        return pd.DataFrame(
            records,
            columns=TRADE_COLUMNS,
        )

    @staticmethod
    def calculate_metrics(
        trades: pd.DataFrame,
    ) -> dict[str, Any]:
        """
        총수익률:
            거래별 수익률의 단순 합계.

        손익비:
            순수익률 합계 / 손실률 합계 절댓값.

        일반적으로 사용하는 평균이익/평균손실 또는
        Profit Factor와는 정의가 다릅니다.
        """
        if trades is None or trades.empty:
            return {
                "거래횟수": 0,
                "승리횟수": 0,
                "패배횟수": 0,
                "승률(%)": 0.0,
                "총수익률(%)": 0.0,
                "평균수익률(%)": 0.0,
                "최대수익률(%)": np.nan,
                "최대손실률(%)": np.nan,
                "손익비": np.nan,
                "평균보유기간(일)": 0.0,
            }

        returns = pd.to_numeric(
            trades["수익률(%)"],
            errors="coerce",
        ).dropna()

        if returns.empty:
            raise ValueError(
                "백테스트 수익률 데이터가 없습니다."
            )

        count = len(returns)
        wins = int(returns.gt(0).sum())
        losses = int(returns.lt(0).sum())

        net_return = float(returns.sum())

        gross_loss = float(
            -returns.loc[returns.lt(0)].sum()
        )

        if gross_loss > 0:
            profit_loss_ratio = (
                net_return / gross_loss
            )
        else:
            profit_loss_ratio = np.nan

        holding_days = pd.to_numeric(
            trades["보유기간(일)"],
            errors="coerce",
        )

        return {
            "거래횟수": count,
            "승리횟수": wins,
            "패배횟수": losses,
            "승률(%)": wins / count * 100.0,
            "총수익률(%)": net_return,
            "평균수익률(%)": float(
                returns.mean()
            ),
            "최대수익률(%)": float(
                returns.max()
            ),
            "최대손실률(%)": float(
                returns.min()
            ),
            "손익비": profit_loss_ratio,
            "평균보유기간(일)": float(
                holding_days.mean()
            ),
        }


# ============================================================
# 전체 분석
# ============================================================

def analyze_frame(
    df: pd.DataFrame,
    stock_name: str,
    excel_file: str | Path | None = None,
    recent_signal_days: int = RECENT_SIGNAL_DAYS,
) -> dict[str, Any]:
    """
    가격 데이터 -> 재무 결합 -> MACD -> 신호 -> 백테스트.

    반환 주요 키
    ------------
    df / data:
        분석 데이터프레임

    latest / latest_status:
        최근 가격 및 신호 상태

    metrics / metrics_df:
        전략별 성과표

    trades:
        {
            "Weekly": 주간 거래기록,
            "Weekly+Monthly": 결합 거래기록,
        }

    weekly_trades / combined_trades:
        전략별 거래기록 직접 접근

    messages:
        안내 메시지 목록
    """
    if (
        isinstance(recent_signal_days, bool)
        or not isinstance(
            recent_signal_days,
            (int, np.integer),
        )
        or recent_signal_days < 1
    ):
        raise ValueError(
            "신호 조회 일수는 1 이상이어야 합니다."
        )

    messages = []

    # 1. 가격 데이터 정리
    frame = prepare_price_frame(df)

    # 2. 재무 데이터 읽기
    financial = None

    if excel_file is not None:
        path = Path(excel_file)

        if path.is_file():
            financial = read_financial_excel(
                path,
                stock_name,
            )
        else:
            messages.append(
                "재무 파일이 없어 목표가와 재무값을 "
                "빈 값으로 처리했습니다."
            )
    else:
        messages.append(
            "재무 파일이 지정되지 않아 목표가와 재무값을 "
            "빈 값으로 처리했습니다."
        )

    frame = (
        FinancialDataManager.add_financial_data(
            frame,
            financial,
        )
    )

    # 3. 반드시 전체 조회 데이터로 MACD 계산
    #    최근 18개월 자르기는 charts.py에서 수행합니다.
    frame = calculate_macd_by_timeframe(
        frame
    )

    # 4. 신호 생성
    frame = SignalGenerator.generate_signals(
        frame
    )

    # 5. 백테스트
    weekly_trades = Backtester.run_backtest(
        frame,
        "Weekly_Buy",
        "Weekly_Sell",
    )

    combined_trades = Backtester.run_backtest(
        frame,
        "Weekly_Monthly_Buy",
        "Weekly_Monthly_Sell",
    )

    weekly_metrics = (
        Backtester.calculate_metrics(
            weekly_trades
        )
    )

    combined_metrics = (
        Backtester.calculate_metrics(
            combined_trades
        )
    )

    metrics = pd.DataFrame(
        [
            {
                "전략": "Weekly",
                **weekly_metrics,
            },
            {
                "전략": "Weekly+Monthly",
                **combined_metrics,
            },
        ]
    )

    # 6. 최근 신호
    weekly_recent = get_recent_signal(
        frame,
        "Weekly_Buy",
        "Weekly_Sell",
        days=recent_signal_days,
    )

    combined_recent = get_recent_signal(
        frame,
        "Weekly_Monthly_Buy",
        "Weekly_Monthly_Sell",
        days=recent_signal_days,
    )

    reference_date = frame.index[-1]

    start_date = (
        reference_date
        - pd.Timedelta(
            days=recent_signal_days - 1
        )
    )

    last_row = frame.iloc[-1]

    latest_status = {
        "종목명": stock_name,
        "기준일": reference_date.strftime(
            "%Y-%m-%d"
        ),
        "종가": float(last_row["종가"]),
        "목표가": last_row["목표가"],
        "괴리율(%)": last_row["괴리율(%)"],

        "Weekly 신호": weekly_recent["신호"],
        "Weekly 신호일": weekly_recent["발생일"],

        "Weekly+Monthly 신호": (
            combined_recent["신호"]
        ),
        "Weekly+Monthly 신호일": (
            combined_recent["발생일"]
        ),

        "Weekly 당일 신호": classify_signal(
            last_row,
            "Weekly_Buy",
            "Weekly_Sell",
        ),
        "Weekly+Monthly 당일 신호": classify_signal(
            last_row,
            "Weekly_Monthly_Buy",
            "Weekly_Monthly_Sell",
        ),

        "신호 조회 시작일": start_date.strftime(
            "%Y-%m-%d"
        ),
        "신호 조회 종료일": reference_date.strftime(
            "%Y-%m-%d"
        ),
        "신호 조회 일수": int(
            recent_signal_days
        ),

        "Daily_MACD": float(
            last_row["Daily_MACD"]
        ),
        "Weekly_MACD": float(
            last_row["Weekly_MACD"]
        ),
        "Monthly_MACD": float(
            last_row["Monthly_MACD"]
        ),

        "Daily_Oscillator": float(
            last_row["Daily_Oscillator"]
        ),
        "Weekly_Oscillator": float(
            last_row["Weekly_Oscillator"]
        ),
        "Monthly_Oscillator": float(
            last_row["Monthly_Oscillator"]
        ),
    }

    messages.append(
        "주간·월간 MACD는 각 날짜까지의 진행 중인 봉을 "
        "포함합니다. 주·월이 끝나기 전 값과 조건은 "
        "이후 거래일에 달라질 수 있습니다."
    )

    messages.append(
        "백테스트는 신호 계산에 사용한 당일 종가로 "
        "즉시 체결된다고 가정합니다. 수수료·세금·"
        "슬리피지와 미청산 손익은 포함하지 않습니다."
    )

    return {
        "stock_name": stock_name,

        "df": frame,
        "data": frame,

        "latest": latest_status,
        "latest_status": latest_status,

        "metrics": metrics,
        "metrics_df": metrics,

        "trades": {
            "Weekly": weekly_trades,
            "Weekly+Monthly": combined_trades,
        },

        "weekly_trades": weekly_trades,
        "combined_trades": combined_trades,

        "weekly_metrics": weekly_metrics,
        "combined_metrics": combined_metrics,

        "financial": financial,
        "messages": messages,
    }


# ============================================================
# 선택적 검증 함수
# ============================================================

def verify_macd_calculation(
    raw_df: pd.DataFrame,
    sample_count: int = 5,
) -> bool:
    """
    MACD 검증용 함수.

    검증 1:
        각 주·월 마지막 데이터 날짜의 MACD가
        직접 계산한 주봉·월봉 MACD와 일치하는지 검사.

    검증 2:
        전체 데이터로 계산한 과거 값이,
        그 날짜까지만 잘라 계산한 값과 일치하는지 검사.
        미래 데이터 참조 여부를 확인합니다.

    검증 실패 시 AssertionError가 발생합니다.

    실제 앱 실행 시 자동 호출하지 않습니다.
    """
    if (
        isinstance(sample_count, bool)
        or not isinstance(
            sample_count,
            (int, np.integer),
        )
        or sample_count < 1
    ):
        raise ValueError(
            "검증 표본 개수는 1 이상이어야 합니다."
        )

    clean = prepare_price_frame(raw_df)

    calculated = calculate_macd_by_timeframe(
        clean
    )

    # --------------------------------------------------------
    # 1. 주·월별 마지막 날짜의 값 검증
    # --------------------------------------------------------
    for prefix, frequency in (
        ("Weekly", "W-FRI"),
        ("Monthly", "M"),
    ):
        keys = clean.index.to_period(
            frequency
        )

        period_close = clean["종가"].groupby(
            keys,
            sort=True,
        ).last()

        fast, slow, signal = (
            MACD_SETTINGS[prefix]
        )

        expected = calculate_native_macd(
            period_close,
            fast,
            slow,
            signal,
        )

        for column in (
            "MACD",
            "Signal",
            "Oscillator",
        ):
            actual = calculated[
                f"{prefix}_{column}"
            ].groupby(
                keys,
                sort=True,
            ).last()

            np.testing.assert_allclose(
                actual.to_numpy(dtype=float),
                expected[column].to_numpy(
                    dtype=float
                ),
                rtol=1e-9,
                atol=1e-8,
                err_msg=(
                    f"{prefix}_{column}: "
                    "기간별 계산 불일치"
                ),
            )

    # --------------------------------------------------------
    # 2. 미래 데이터 참조 여부 검증
    # --------------------------------------------------------
    positions = np.unique(
        np.linspace(
            0,
            len(clean) - 1,
            num=min(
                sample_count,
                len(clean),
            ),
            dtype=int,
        )
    )

    check_columns = [
        f"{prefix}_{column}"
        for prefix in (
            "Daily",
            "Weekly",
            "Monthly",
        )
        for column in (
            "MACD",
            "Signal",
            "Oscillator",
            "Prev_Oscillator",
            "Oscillator_Change",
        )
    ]

    for position in positions:
        truncated = (
            calculate_macd_by_timeframe(
                clean.iloc[:position + 1]
            )
        )

        expected_row = truncated.iloc[-1]
        actual_row = calculated.iloc[position]

        np.testing.assert_allclose(
            actual_row[
                check_columns
            ].to_numpy(dtype=float),
            expected_row[
                check_columns
            ].to_numpy(dtype=float),
            rtol=1e-9,
            atol=1e-8,
            equal_nan=True,
            err_msg=(
                f"{clean.index[position]}: "
                "미래 데이터 참조 검증 실패"
            ),
        )

    return True
