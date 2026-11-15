# analysis_engine.py
"""
不进行界面输出或自动文件保存，只返回分析结果。

재무 엑셀 필수 컬럼:
    분기
    목표가
    매출액
    영업이익
    순이익

재무 금액:
    엑셀 입력값 그대로 사용합니다.
    단위 환산은 하지 않습니다.

최근 신호:
    가격 데이터 기준일 포함 최근 7일의 마지막 신호
    거래일이 아닌 달력일 기준

주간·월간 결합 신호:
    매수:
        이번 달 Monthly Oscillator > 전월 Oscillator
        그리고
        이번 주 Weekly Oscillator > 전주 Oscillator

    매도:
        이번 달 Monthly Oscillator < 전월 Oscillator
        그리고
        이번 주 Weekly Oscillator < 전주 Oscillator

    동일하거나 비교값이 없으면 해당 방향 조건 미충족.

    Oscillator가 양수인지 음수인지는 조건에 포함하지 않습니다.
    추세 전환 여부도 결합 신호 조건에 포함하지 않습니다.

    조건이 유지되는 동안 매일 같은 방향 조건 신호가 발생합니다.
    백테스트는 단일 포지션만 유지하므로 반복 매수하지 않습니다.

주간·월간 MACD:
    각 거래일 당시까지의 종가로 진행 중인 주·월을 계산합니다.
    나중에 확정된 주·월 값을 과거 날짜로 역배정하지 않습니다.

    각 날짜의 진행 중인 주·월 EMA 계산은
    직전 기간까지 확정된 EMA 상태에서 계산합니다.
    하루가 지날 때마다 주·월 EMA를 반복 누적하지 않습니다.

주의:
    진행 중인 주·월의 현재 조건은 날짜가 지나면서 변할 수 있습니다.
    하지만 동일한 과거 가격을 유지한다면,
    나중 데이터 추가만으로 과거 날짜의 기술적 신호를
    다시 쓰도록 계산하지 않습니다.

    재무값은 분기별로 매핑합니다.
    공시 당시의 이용 가능 시점을 반영한 데이터는 아닙니다.
    재무값은 신호나 백테스트 매매 조건에 사용하지 않습니다.

    백테스트는 신호 발생일 종가 체결을 가정합니다.
    이는 실제 체결 가능성을 보장하지 않습니다.
    수수료·세금·슬리피지·미청산 평가손익은 제외합니다.

공매도·대차거래 조회 기능은 포함하지 않습니다.
"""

import numpy as np
import pandas as pd


# =========================================================
# 기본 설정
# =========================================================

RECENT_SIGNAL_DAYS = 7

FINANCIAL_VALUE_COLUMNS = (
    "목표가",
    "매출액",
    "영업이익",
    "순이익",
)

FINANCIAL_REQUIRED_COLUMNS = {
    "분기",
    *FINANCIAL_VALUE_COLUMNS,
}


# =========================================================
# 공통 숫자·날짜 처리
# =========================================================

def to_numeric(series):
    """쉼표 포함 숫자 문자열을 숫자형으로 변환합니다."""
    if not pd.api.types.is_numeric_dtype(series):
        series = (
            series.astype("string")
            .str.replace(",", "", regex=False)
            .str.strip()
        )

    return (
        pd.to_numeric(series, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
    )


def normalize_timestamp(value):
    """한국 날짜 기준 시간대 없는 Timestamp."""
    if value is None:
        raise ValueError("유효한 날짜가 필요합니다.")

    timestamp = pd.Timestamp(value)

    if pd.isna(timestamp):
        raise ValueError("유효한 날짜가 필요합니다.")

    if timestamp.tzinfo is not None:
        timestamp = (
            timestamp
            .tz_convert("Asia/Seoul")
            .tz_localize(None)
        )

    return timestamp.normalize()


def validate_positive_integer(value, message):
    if (
        not isinstance(value, (int, np.integer))
        or isinstance(value, (bool, np.bool_))
        or value < 1
    ):
        raise ValueError(message)

    return int(value)


def previous_period_values(series, frequency):
    """
    각 날짜에 대응하는 직전 달력 기간의 마지막 값을 반환합니다.

    예:
        frequency="W-FRI":
            이번 주 각 날짜 → 전주의 마지막 값

        frequency="M":
            이번 달 각 날짜 → 전월의 마지막 값

    같은 기간의 미래 값을 현재 날짜에 사용하지 않습니다.
    직전 기간 자체가 없으면 NaN입니다.
    """
    if not isinstance(series.index, pd.DatetimeIndex):
        raise ValueError(
            "지표 인덱스는 날짜 형식이어야 합니다."
        )

    values = to_numeric(series).astype(float)
    periods = values.index.to_period(frequency)

    # NaN을 건너뛰지 않고 실제 마지막 행 값을 사용합니다.
    period_last = values.groupby(
        periods,
        sort=True,
    ).agg(
        lambda group: group.iloc[-1]
    )

    # 전 기간 값을 다음 기간의 키로 이동시킵니다.
    # 기간이 누락되면 더 오래된 값을 임의 대체하지 않습니다.
    previous_lookup = period_last.copy()
    previous_lookup.index = previous_lookup.index + 1

    mapped = periods.map(previous_lookup)

    return pd.Series(
        np.asarray(mapped, dtype=float),
        index=values.index,
        dtype=float,
    )


# =========================================================
# 재무 엑셀 읽기·검증
# =========================================================

def read_financial_excel(excel_file, stock_name):
    """
    종목별 재무 시트를 읽고 검증합니다.

    목표가:
        음수는 0으로 보정합니다.
        100원 단위 반올림은 매핑 시 적용합니다.

    나머지 재무값:
        입력값과 결측값을 그대로 유지합니다.
    """
    with pd.ExcelFile(excel_file) as book:
        if stock_name not in book.sheet_names:
            raise ValueError(
                f"엑셀에 '{stock_name}' 시트가 없습니다. "
                f"현재 시트: {', '.join(book.sheet_names)}"
            )

        financial = pd.read_excel(
            book,
            sheet_name=stock_name,
        )

    financial.columns = [
        str(column).strip()
        for column in financial.columns
    ]

    if financial.columns.duplicated().any():
        raise ValueError(
            "재무 엑셀에 중복된 컬럼이 있습니다."
        )

    missing = (
        FINANCIAL_REQUIRED_COLUMNS
        - set(financial.columns)
    )

    if missing:
        raise ValueError(
            "엑셀에 필요한 컬럼이 없습니다: "
            + ", ".join(sorted(missing))
        )

    financial = (
        financial
        .dropna(how="all")
        .copy()
    )

    if financial.empty:
        raise ValueError(
            "재무 엑셀에 데이터가 없습니다."
        )

    financial["분기"] = (
        financial["분기"]
        .astype("string")
        .str.strip()
    )

    valid_quarters = (
        financial["분기"]
        .str.fullmatch(
            r"\d{4}\.[1-4]Q",
            na=False,
        )
    )

    if not valid_quarters.all():
        raise ValueError(
            "분기는 '2026.1Q' 형식으로 입력하세요."
        )

    if financial["분기"].duplicated().any():
        raise ValueError(
            "엑셀에 중복된 분기가 있습니다."
        )

    for column in FINANCIAL_VALUE_COLUMNS:
        original = financial[column]

        cleaned = (
            original.astype("string")
            .str.replace(",", "", regex=False)
            .str.strip()
        )

        numeric = pd.to_numeric(
            cleaned,
            errors="coerce",
        )

        supplied = (
            original.notna()
            & cleaned.ne("").fillna(False)
        )

        invalid = supplied & (
            numeric.isna()
            | numeric.isin([np.inf, -np.inf])
        )

        if invalid.any():
            raise ValueError(
                f"'{column}'에는 숫자 또는 빈칸만 입력하세요."
            )

        financial[column] = (
            numeric
            .replace([np.inf, -np.inf], np.nan)
            .astype(float)
        )

    financial["목표가"] = (
        financial["목표가"]
        .clip(lower=0)
    )

    return (
        financial
        .sort_values("분기")
        .reset_index(drop=True)
    )


def validate_excel(excel_file, stock_name):
    """기존 호출 호환용 재무 엑셀 검증 함수."""
    read_financial_excel(
        excel_file,
        stock_name,
    )


# =========================================================
# 재무 데이터
# =========================================================

class FinancialDataManager:

    @staticmethod
    def load_financial_data(
        excel_file,
        sheet_name,
    ):
        return read_financial_excel(
            excel_file,
            sheet_name,
        )

    @staticmethod
    def load_target_price(
        excel_file,
        sheet_name,
        stock_name="",
    ):
        """
        기존 호출 호환성을 위해 stock_name 인자를 유지합니다.
        """
        financial = (
            FinancialDataManager.load_financial_data(
                excel_file,
                sheet_name,
            )
        )

        target_prices = {}

        for quarter, price in zip(
            financial["분기"],
            financial["목표가"],
        ):
            target_prices[str(quarter)] = (
                int(round(float(price) / 100) * 100)
                if pd.notna(price)
                else np.nan
            )

        return target_prices

    @staticmethod
    def get_quarter(date):
        quarter = (date.month - 1) // 3 + 1
        return f"{date.year}.{quarter}Q"

    @staticmethod
    def calculate_divergence_rate(
        current_price,
        target_price,
    ):
        """
        괴리율(%) = (현재가 - 목표가) / 목표가 × 100
        """
        if (
            pd.isna(current_price)
            or pd.isna(target_price)
            or target_price == 0
        ):
            return np.nan

        return (
            (current_price - target_price)
            / target_price
            * 100
        )

    @staticmethod
    def add_quarter_and_target_price(
        df,
        excel_file,
        sheet_name,
    ):
        """
        분기별 재무값을 같은 분기의 일봉에 매핑합니다.

        공시 당시 이용 가능했던 시점 기준이 아니며,
        매매 신호 및 백테스트 조건에는 사용하지 않습니다.
        """
        result = df.copy()

        financial = (
            FinancialDataManager.load_financial_data(
                excel_file,
                sheet_name,
            )
        )

        result["분기"] = result.index.map(
            FinancialDataManager.get_quarter
        )

        quarterly = financial.set_index("분기")

        rounded_targets = quarterly["목표가"].map(
            lambda value: (
                int(round(float(value) / 100) * 100)
                if pd.notna(value)
                else np.nan
            )
        )

        result["목표가"] = (
            result["분기"]
            .map(rounded_targets)
        )

        for column in (
            "매출액",
            "영업이익",
            "순이익",
        ):
            result[column] = (
                result["분기"]
                .map(quarterly[column])
            )

        valid_target = result["목표가"].where(
            result["목표가"].ne(0)
        )

        result["괴리율(%)"] = (
            (result["종가"] - valid_target)
            / valid_target
            * 100
        )

        return result


# =========================================================
# 기술적 지표
# =========================================================

class TechnicalIndicator:

    TIMEFRAME_PARAMS = {
        "daily": {
            "fast": 12,
            "slow": 26,
            "signal": 9,
        },
        "weekly": {
            "fast": 5,
            "slow": 13,
            "signal": 5,
        },
        "monthly": {
            "fast": 3,
            "slow": 7,
            "signal": 3,
        },
    }

    @staticmethod
    def ema(series, span=12):
        return series.ewm(
            span=span,
            adjust=False,
        ).mean()

    @staticmethod
    def macd(
        df,
        price_col="종가",
        timeframe="daily",
        fast=None,
        slow=None,
        signal=None,
    ):
        result = df.copy()

        defaults = (
            TechnicalIndicator.TIMEFRAME_PARAMS.get(
                str(timeframe).lower(),
                TechnicalIndicator.TIMEFRAME_PARAMS[
                    "daily"
                ],
            )
        )

        fast = (
            defaults["fast"]
            if fast is None
            else fast
        )

        slow = (
            defaults["slow"]
            if slow is None
            else slow
        )

        signal = (
            defaults["signal"]
            if signal is None
            else signal
        )

        prices = to_numeric(
            result[price_col]
        ).astype(float)

        ema_fast = TechnicalIndicator.ema(
            prices,
            fast,
        )

        ema_slow = TechnicalIndicator.ema(
            prices,
            slow,
        )

        result["MACD"] = ema_fast - ema_slow

        result["Signal"] = TechnicalIndicator.ema(
            result["MACD"],
            signal,
        )

        result["Oscillator"] = (
            result["MACD"] - result["Signal"]
        )

        return result

    @staticmethod
    def add_daily_macd(df):
        result = TechnicalIndicator.macd(
            df,
            timeframe="daily",
        )

        return result.rename(
            columns={
                "MACD": "Daily_MACD",
                "Signal": "Daily_Signal",
                "Oscillator": "Daily_Oscillator",
            }
        )

    @staticmethod
    def _add_period_macd_asof(
        df,
        prefix,
        frequency,
        fast,
        slow,
        signal,
    ):
        """
        각 날짜 당시까지의 진행 중인 기간 MACD를 계산합니다.

        예: 주간
            전주까지 확정된 EMA 상태를 보관합니다.
            이번 주 월요일은 월요일 종가로 가계산합니다.
            화요일은 같은 전주 EMA 상태에서 화요일 종가로
            다시 계산합니다.

            월요일 가계산 EMA에 화요일을 재누적하지 않습니다.

        EMA는 실제 관측된 기간의 종가를 기준으로 계산합니다.
        가격이 전혀 없는 기간의 종가를 임의 생성하지 않습니다.
        """
        result = df.copy()

        if not isinstance(
            result.index,
            pd.DatetimeIndex,
        ):
            raise ValueError(
                "MACD 데이터 인덱스는 날짜 형식이어야 합니다."
            )

        if (
            not result.index.is_monotonic_increasing
            or result.index.has_duplicates
        ):
            raise ValueError(
                "MACD 데이터 날짜는 중복 없이 "
                "오름차순이어야 합니다."
            )

        prices = (
            to_numeric(result["종가"])
            .to_numpy(dtype=float)
        )

        if not np.isfinite(prices).all():
            raise ValueError(
                "MACD 계산에 유효하지 않은 종가가 있습니다."
            )

        periods = result.index.to_period(
            frequency
        )

        alpha_fast = 2.0 / (fast + 1.0)
        alpha_slow = 2.0 / (slow + 1.0)
        alpha_signal = 2.0 / (signal + 1.0)

        count = len(result)

        macd_values = np.full(
            count,
            np.nan,
            dtype=float,
        )

        signal_values = np.full(
            count,
            np.nan,
            dtype=float,
        )

        oscillator_values = np.full(
            count,
            np.nan,
            dtype=float,
        )

        active_period = None

        # 직전 관측 기간까지 확정된 EMA 상태
        base_fast = None
        base_slow = None
        base_signal = None

        # 현재 기간 마지막 가계산 상태
        current_fast = None
        current_slow = None
        current_signal = None

        for position, (period, price) in enumerate(
            zip(periods, prices)
        ):
            if (
                active_period is None
                or period != active_period
            ):
                # 새 기간이 시작되면 직전 기간의 마지막 상태 확정
                if active_period is not None:
                    base_fast = current_fast
                    base_slow = current_slow
                    base_signal = current_signal

                active_period = period

            if base_fast is None:
                # 첫 관측 기간은 adjust=False EMA 초기화와 동일
                current_fast = float(price)
                current_slow = float(price)

                current_macd = 0.0
                current_signal = 0.0

            else:
                current_fast = (
                    alpha_fast * price
                    + (1.0 - alpha_fast) * base_fast
                )

                current_slow = (
                    alpha_slow * price
                    + (1.0 - alpha_slow) * base_slow
                )

                current_macd = (
                    current_fast - current_slow
                )

                current_signal = (
                    alpha_signal * current_macd
                    + (1.0 - alpha_signal) * base_signal
                )

            macd_values[position] = current_macd

            signal_values[position] = (
                current_signal
            )

            oscillator_values[position] = (
                current_macd - current_signal
            )

        result[f"{prefix}_MACD"] = (
            macd_values
        )

        result[f"{prefix}_Signal"] = (
            signal_values
        )

        result[f"{prefix}_Oscillator"] = (
            oscillator_values
        )

        previous = previous_period_values(
            result[f"{prefix}_Oscillator"],
            frequency,
        )

        result[
            f"{prefix}_Previous_Oscillator"
        ] = previous

        result[
            f"{prefix}_Oscillator_Change"
        ] = (
            result[f"{prefix}_Oscillator"]
            - previous
        )

        return result

    @staticmethod
    def add_weekly_macd(df):
        """
        각 거래일 당시의 이번 주 MACD를 계산합니다.

        전주 비교:
            W-FRI 기준 직전 주의 마지막 관측값
        """
        params = (
            TechnicalIndicator.TIMEFRAME_PARAMS[
                "weekly"
            ]
        )

        return TechnicalIndicator._add_period_macd_asof(
            df,
            prefix="Weekly",
            frequency="W-FRI",
            fast=params["fast"],
            slow=params["slow"],
            signal=params["signal"],
        )

    @staticmethod
    def add_monthly_macd(df):
        """
        각 거래일 당시의 이번 달 MACD를 계산합니다.

        전월 비교:
            직전 달의 마지막 관측값
        """
        params = (
            TechnicalIndicator.TIMEFRAME_PARAMS[
                "monthly"
            ]
        )

        return TechnicalIndicator._add_period_macd_asof(
            df,
            prefix="Monthly",
            frequency="M",
            fast=params["fast"],
            slow=params["slow"],
            signal=params["signal"],
        )


# =========================================================
# 매매 신호
# =========================================================

class SignalGenerator:

    @staticmethod
    def trend_from_change(change):
        """
         1: 증가
        -1: 감소
         0: 동일 또는 비교값 없음
        """
        numeric = to_numeric(change)

        trend = pd.Series(
            0,
            index=change.index,
            dtype=int,
        )

        trend.loc[numeric > 0] = 1
        trend.loc[numeric < 0] = -1

        return trend

    @staticmethod
    def calculate_trend(osc_series):
        """일간 등 인접 행 간 증감으로 추세를 계산합니다."""
        return SignalGenerator.trend_from_change(
            to_numeric(osc_series).diff()
        )

    @staticmethod
    def maintain_weekly_trend(df):
        """
        기존 함수명 호환용.

        과거처럼 주 전체를 한 방향으로 덮어쓰지 않습니다.
        각 날짜의 이번 주 값과 전주 값을 비교합니다.
        """
        result = df.copy()

        previous = previous_period_values(
            result["Weekly_Oscillator"],
            "W-FRI",
        )

        change = (
            to_numeric(
                result["Weekly_Oscillator"]
            )
            - previous
        )

        result["Weekly_Previous_Oscillator"] = (
            previous
        )

        result["Weekly_Oscillator_Change"] = (
            change
        )

        result["Weekly_Trend"] = (
            SignalGenerator.trend_from_change(
                change
            )
        )

        return result

    @staticmethod
    def maintain_monthly_trend(df):
        """
        기존 함수명 호환용.

        과거처럼 월 전체를 한 방향으로 덮어쓰지 않습니다.
        각 날짜의 이번 달 값과 전월 값을 비교합니다.
        """
        result = df.copy()

        previous = previous_period_values(
            result["Monthly_Oscillator"],
            "M",
        )

        change = (
            to_numeric(
                result["Monthly_Oscillator"]
            )
            - previous
        )

        result["Monthly_Previous_Oscillator"] = (
            previous
        )

        result["Monthly_Oscillator_Change"] = (
            change
        )

        result["Monthly_Trend"] = (
            SignalGenerator.trend_from_change(
                change
            )
        )

        return result

    @staticmethod
    def identify_signal_changes(trend_series):
        """
        단독 신호의 엄격한 추세 전환 규칙 유지.

        매수: -1 → +1
        매도: +1 → -1

        0을 경유하는 전환은 인정하지 않습니다.
        """
        previous = trend_series.shift(1)

        buy = (
            previous.eq(-1)
            & trend_series.eq(1)
        )

        sell = (
            previous.eq(1)
            & trend_series.eq(-1)
        )

        return buy, sell

    @staticmethod
    def identify_weekly_monthly_signals(df):
        """
        요청한 주간·월간 결합 조건.

        매수:
            이번 달 Oscillator > 전월 Oscillator
            AND
            이번 주 Oscillator > 전주 Oscillator

        매도:
            이번 달 Oscillator < 전월 Oscillator
            AND
            이번 주 Oscillator < 전주 Oscillator

        추세 전환 여부와 Oscillator의 부호는 검사하지 않습니다.
        조건을 만족하는 날짜마다 True를 반환합니다.
        """
        weekly_previous = previous_period_values(
            df["Weekly_Oscillator"],
            "W-FRI",
        )

        monthly_previous = previous_period_values(
            df["Monthly_Oscillator"],
            "M",
        )

        weekly_change = (
            to_numeric(df["Weekly_Oscillator"])
            - weekly_previous
        )

        monthly_change = (
            to_numeric(df["Monthly_Oscillator"])
            - monthly_previous
        )

        buy = (
            monthly_change.gt(0)
            & weekly_change.gt(0)
        )

        sell = (
            monthly_change.lt(0)
            & weekly_change.lt(0)
        )

        return buy, sell

    @staticmethod
    def add_signals(df):
        result = df.copy()

        required = {
            "Daily_Oscillator",
            "Weekly_Oscillator",
            "Monthly_Oscillator",
        }

        missing = required - set(result.columns)

        if missing:
            raise ValueError(
                "신호 계산용 지표 컬럼 누락: "
                + ", ".join(sorted(missing))
            )

        # 일간은 전일 대비 증감
        result["Daily_Trend"] = (
            SignalGenerator.calculate_trend(
                result["Daily_Oscillator"]
            )
        )

        # 주간은 전주 대비, 월간은 전월 대비 증감
        result = (
            SignalGenerator.maintain_weekly_trend(
                result
            )
        )

        result = (
            SignalGenerator.maintain_monthly_trend(
                result
            )
        )

        # 단독 신호는 기존 엄격한 방향 전환 규칙 유지
        for prefix in (
            "Daily",
            "Weekly",
            "Monthly",
        ):
            buy, sell = (
                SignalGenerator.identify_signal_changes(
                    result[f"{prefix}_Trend"]
                )
            )

            result[f"{prefix}_Buy"] = (
                buy.astype(int)
            )

            result[f"{prefix}_Sell"] = (
                sell.astype(int)
            )

        # 결합 신호는 전환이 아닌 동시 증감 조건
        combined_buy, combined_sell = (
            SignalGenerator.identify_weekly_monthly_signals(
                result
            )
        )

        result["Weekly_Monthly_Buy"] = (
            combined_buy.astype(int)
        )

        result["Weekly_Monthly_Sell"] = (
            combined_sell.astype(int)
        )

        result["Weekly_Monthly_Condition"] = (
            np.select(
                [
                    combined_buy,
                    combined_sell,
                ],
                [
                    1,
                    -1,
                ],
                default=0,
            )
            .astype(int)
        )

        return result


# =========================================================
# 백테스트
# =========================================================

class Backtester:

    @staticmethod
    def backtest(
        df,
        signal_type="Weekly",
        monthly_filter=None,
        price_col="종가",
    ):
        """
        - 신호 발생일 종가 매수·매도 가정
        - 한 번에 한 포지션
        - 반복 매수 조건은 추가 매수하지 않음
        - 미청산 포지션 제외
        - 수수료·세금·슬리피지 제외
        """
        buy_col = f"{signal_type}_Buy"
        sell_col = f"{signal_type}_Sell"

        required = {
            buy_col,
            sell_col,
            price_col,
        }

        if monthly_filter is not None:
            required.add("Monthly_Trend")

        missing = required - set(df.columns)

        if missing:
            raise ValueError(
                "백테스트 컬럼 누락: "
                + ", ".join(sorted(missing))
            )

        trade_columns = [
            "전략",
            "매수일",
            "매수가",
            "매도일",
            "매도가",
            "수익률(%)",
            "수익(원)",
            "보유기간(일)",
            "청산사유",
        ]

        trades = []

        in_position = False
        entry_date = None
        entry_price = None

        def make_trade(
            exit_date,
            exit_price,
            reason,
        ):
            profit_rate = (
                (exit_price - entry_price)
                / entry_price
                * 100
            )

            return {
                "전략": (
                    f"{signal_type} "
                    f"(Monthly={monthly_filter})"
                ),
                "매수일": entry_date,
                "매수가": round(entry_price, 0),
                "매도일": exit_date,
                "매도가": round(exit_price, 0),
                "수익률(%)": round(profit_rate, 2),
                "수익(원)": round(
                    exit_price - entry_price,
                    0,
                ),
                "보유기간(일)": (
                    exit_date - entry_date
                ).days,
                "청산사유": reason,
            }

        for position in range(len(df)):
            current_date = df.index[position]

            current_price = float(
                df[price_col].iloc[position]
            )

            if (
                not np.isfinite(current_price)
                or current_price <= 0
            ):
                continue

            if monthly_filter is not None:
                current_monthly = (
                    df["Monthly_Trend"].iloc[position]
                )

                if current_monthly != monthly_filter:
                    if in_position:
                        trades.append(
                            make_trade(
                                current_date,
                                current_price,
                                "Monthly_Trend 변화",
                            )
                        )

                        in_position = False

                    continue

            if (
                df[buy_col].iloc[position] == 1
                and not in_position
            ):
                entry_date = current_date
                entry_price = current_price
                in_position = True

            elif (
                df[sell_col].iloc[position] == 1
                and in_position
            ):
                trades.append(
                    make_trade(
                        current_date,
                        current_price,
                        "신호",
                    )
                )

                in_position = False

        return pd.DataFrame(
            trades,
            columns=trade_columns,
        )

    @staticmethod
    def calculate_metrics(
        trades_df,
        strategy_name,
    ):
        """
        기존 성과 계산 방식 유지.

        총수익률:
            거래별 수익률 단순 합계

        손익비:
            abs(순수익 합계 / 손실 거래의 손익 합계)

        일반적인 Profit Factor와 다른 기존 계산식입니다.
        """
        if trades_df.empty:
            return {
                "전략": strategy_name,
                "총거래수": 0,
                "승리": 0,
                "패배": 0,
                "중립": 0,
                "승률(%)": 0,
                "평균수익(%)": 0,
                "평균손실(%)": 0,
                "총수익률(%)": 0,
                "누적수익(원)": 0,
                "최대수익(%)": 0,
                "최대손실(%)": 0,
                "손익비": 0,
                "평균보유기간(일)": 0,
            }

        win = trades_df.loc[
            trades_df["수익률(%)"] > 0
        ]

        loss = trades_df.loc[
            trades_df["수익률(%)"] < 0
        ]

        total_profit = (
            trades_df["수익(원)"].sum()
        )

        total_loss = (
            loss["수익(원)"].sum()
        )

        profit_factor = (
            abs(total_profit / total_loss)
            if total_loss != 0
            else 0
        )

        count = len(trades_df)

        return {
            "전략": strategy_name,
            "총거래수": count,
            "승리": len(win),
            "패배": len(loss),
            "중립": (
                count - len(win) - len(loss)
            ),
            "승률(%)": round(
                len(win) / count * 100,
                1,
            ),
            "평균수익(%)": (
                round(
                    win["수익률(%)"].mean(),
                    1,
                )
                if not win.empty
                else 0
            ),
            "평균손실(%)": (
                round(
                    loss["수익률(%)"].mean(),
                    1,
                )
                if not loss.empty
                else 0
            ),
            "총수익률(%)": round(
                trades_df["수익률(%)"].sum(),
                1,
            ),
            "누적수익(원)": round(
                total_profit,
                0,
            ),
            "최대수익(%)": round(
                trades_df["수익률(%)"].max(),
                1,
            ),
            "최대손실(%)": round(
                trades_df["수익률(%)"].min(),
                1,
            ),
            "손익비": round(
                profit_factor,
                1,
            ),
            "평균보유기간(일)": round(
                trades_df["보유기간(일)"].mean(),
                1,
            ),
        }


# =========================================================
# 신호 조회
# =========================================================

def classify_signal(
    row,
    buy_col="Weekly_Buy",
    sell_col="Weekly_Sell",
):
    if row.get(buy_col, 0) == 1:
        return "매수"

    if row.get(sell_col, 0) == 1:
        return "매도"

    return "신호 없음"


def get_recent_signal(
    df,
    buy_col="Weekly_Buy",
    sell_col="Weekly_Sell",
    days=RECENT_SIGNAL_DAYS,
    reference_date=None,
):
    """기준일 포함 최근 days일의 마지막 조건 신호."""
    empty_result = {
        "신호": "신호 없음",
        "발생일": None,
    }

    days = validate_positive_integer(
        days,
        "신호 조회 일수는 1 이상의 정수여야 합니다.",
    )

    if df is None or df.empty:
        return empty_result

    missing = (
        {buy_col, sell_col}
        - set(df.columns)
    )

    if missing:
        raise ValueError(
            "신호 컬럼 누락: "
            + ", ".join(sorted(missing))
        )

    if not isinstance(
        df.index,
        pd.DatetimeIndex,
    ):
        raise ValueError(
            "신호 데이터 인덱스는 날짜 형식이어야 합니다."
        )

    frame = df.copy()

    if frame.index.tz is not None:
        frame.index = (
            frame.index
            .tz_convert("Asia/Seoul")
            .tz_localize(None)
        )

    frame = (
        frame.loc[
            ~frame.index.isna()
        ]
        .sort_index()
    )

    if frame.empty:
        return empty_result

    if reference_date is None:
        reference_date = frame.index.max()

    reference = normalize_timestamp(
        reference_date
    )

    start_date = (
        reference
        - pd.Timedelta(days=days - 1)
    )

    dates = frame.index.normalize()

    date_mask = (
        (dates >= start_date)
        & (dates <= reference)
    )

    signal_mask = (
        frame[buy_col].eq(1)
        | frame[sell_col].eq(1)
    )

    recent = frame.loc[
        date_mask & signal_mask
    ]

    if recent.empty:
        return empty_result

    return {
        "신호": classify_signal(
            recent.iloc[-1],
            buy_col,
            sell_col,
        ),
        "발생일": (
            recent.index[-1]
            .strftime("%Y-%m-%d")
        ),
    }


def recent_signals(
    df,
    combined=False,
    limit=10,
):
    """
    전체 기간 중 최근 limit개 조건 신호 행을 반환합니다.

    날짜 오름차순으로 반환하므로 기존 app.py와 호환됩니다.

    결합 조건이 여러 날 유지되면
    같은 방향 신호가 여러 행 표시될 수 있습니다.
    실제 체결 기록과는 다른 표입니다.
    """
    prefix = (
        "Weekly_Monthly"
        if combined
        else "Weekly"
    )

    buy_col = f"{prefix}_Buy"
    sell_col = f"{prefix}_Sell"

    output_columns = [
        "날짜",
        "종가",
        "목표가",
        "괴리율(%)",
        "신호",
        "Daily_Trend",
        "Weekly_Trend",
        "Monthly_Trend",
    ]

    if (
        df is None
        or df.empty
        or limit < 1
    ):
        return pd.DataFrame(
            columns=output_columns
        )

    missing = (
        {buy_col, sell_col}
        - set(df.columns)
    )

    if missing:
        raise ValueError(
            "신호 컬럼 누락: "
            + ", ".join(sorted(missing))
        )

    mask = (
        df[buy_col].eq(1)
        | df[sell_col].eq(1)
    )

    result = (
        df.loc[mask]
        .sort_index()
        .tail(limit)
        .copy()
    )

    if result.empty:
        return pd.DataFrame(
            columns=output_columns
        )

    result["날짜"] = (
        result.index.strftime("%Y-%m-%d")
    )

    result["신호"] = result.apply(
        lambda row: classify_signal(
            row,
            buy_col,
            sell_col,
        ),
        axis=1,
    )

    columns = [
        column
        for column in output_columns
        if column in result.columns
    ]

    return (
        result[columns]
        .reset_index(drop=True)
    )


# =========================================================
# 가격 데이터 검증
# =========================================================

def prepare_price_frame(frame):
    if frame is None or frame.empty:
        raise ValueError(
            "가격 데이터가 없습니다."
        )

    df = frame.copy()

    if df.columns.duplicated().any():
        raise ValueError(
            "가격 데이터에 중복된 컬럼이 있습니다."
        )

    required = {
        "시가",
        "고가",
        "저가",
        "종가",
        "거래량",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            "가격 데이터 컬럼 누락: "
            + ", ".join(sorted(missing))
        )

    if not isinstance(
        df.index,
        pd.DatetimeIndex,
    ):
        if pd.api.types.is_numeric_dtype(
            df.index.dtype
        ):
            raise ValueError(
                "가격 데이터 인덱스는 날짜 형식이어야 합니다."
            )

        df.index = pd.to_datetime(
            df.index,
            errors="coerce",
        )

    df = df.loc[
        ~df.index.isna()
    ].copy()

    if df.index.tz is not None:
        df.index = (
            df.index
            .tz_convert("Asia/Seoul")
            .tz_localize(None)
        )

    df.index = df.index.normalize()
    df.index.name = "날짜"

    df = df.sort_index()

    df = df.loc[
        ~df.index.duplicated(keep="last")
    ].copy()

    for column in required:
        df[column] = to_numeric(
            df[column]
        )

    df = df.dropna(
        subset=list(required)
    )

    df = df.loc[
        df["종가"] > 0
    ].copy()

    if df.empty:
        raise ValueError(
            "유효한 가격 데이터가 없습니다."
        )

    df["날짜"] = (
        df.index.strftime("%Y-%m-%d")
    )

    return df


# =========================================================
# 분석 실행
# =========================================================

def analyze_frame(
    frame,
    stock_name,
    excel_file=None,
    recent_signal_days=RECENT_SIGNAL_DAYS,
):
    """
    가격 데이터와 재무 엑셀을 분석합니다.

    반환:
        data
        metrics
        trades
        latest
        messages
    """
    recent_signal_days = validate_positive_integer(
        recent_signal_days,
        "최근 신호 조회 일수는 "
        "1 이상의 정수여야 합니다.",
    )

    df = prepare_price_frame(frame)
    messages = []

    # -----------------------------------------------------
    # 1. 재무 정보
    # -----------------------------------------------------

    if (
        excel_file is not None
        and str(excel_file) != ""
    ):
        df = (
            FinancialDataManager.add_quarter_and_target_price(
                df,
                excel_file,
                stock_name,
            )
        )

    else:
        df["분기"] = df.index.map(
            FinancialDataManager.get_quarter
        )

        for column in (
            "목표가",
            "매출액",
            "영업이익",
            "순이익",
            "괴리율(%)",
        ):
            df[column] = np.nan

        messages.append(
            "재무 엑셀이 없어 목표가·괴리율·"
            "매출액·영업이익·순이익은 표시하지 않습니다."
        )

    # -----------------------------------------------------
    # 2. 기술적 지표
    # -----------------------------------------------------

    df = TechnicalIndicator.add_daily_macd(
        df
    )

    df = TechnicalIndicator.add_weekly_macd(
        df
    )

    df = TechnicalIndicator.add_monthly_macd(
        df
    )

    # -----------------------------------------------------
    # 3. 매매 신호
    # -----------------------------------------------------

    df = SignalGenerator.add_signals(
        df
    )

    # -----------------------------------------------------
    # 4. 백테스트
    # -----------------------------------------------------

    strategies = [
        (
            "Weekly",
            "Weekly 단독",
            "Weekly",
        ),
        (
            "Weekly+Monthly",
            "Weekly+Monthly 결합",
            "Weekly_Monthly",
        ),
    ]

    trades = {}
    metrics = []

    for key, label, signal_type in strategies:
        trade_frame = Backtester.backtest(
            df,
            signal_type=signal_type,
            monthly_filter=None,
            price_col="종가",
        )

        trades[key] = trade_frame

        metrics.append(
            Backtester.calculate_metrics(
                trade_frame,
                label,
            )
        )

    # -----------------------------------------------------
    # 5. 최근 신호
    # -----------------------------------------------------

    latest_row = df.iloc[-1]
    reference_date = df.index[-1]

    signal_start = (
        reference_date
        - pd.Timedelta(
            days=recent_signal_days - 1
        )
    )

    weekly_recent = get_recent_signal(
        df,
        buy_col="Weekly_Buy",
        sell_col="Weekly_Sell",
        days=recent_signal_days,
        reference_date=reference_date,
    )

    combined_recent = get_recent_signal(
        df,
        buy_col="Weekly_Monthly_Buy",
        sell_col="Weekly_Monthly_Sell",
        days=recent_signal_days,
        reference_date=reference_date,
    )

    latest_status = {
        "종목명": stock_name,
        "기준일": (
            reference_date.strftime("%Y-%m-%d")
        ),
        "종가": float(
            latest_row["종가"]
        ),
        "Weekly 신호": (
            weekly_recent["신호"]
        ),
        "Weekly 신호일": (
            weekly_recent["발생일"]
        ),
        "Weekly+Monthly 신호": (
            combined_recent["신호"]
        ),
        "Weekly+Monthly 신호일": (
            combined_recent["발생일"]
        ),
        "신호 조회 시작일": (
            signal_start.strftime("%Y-%m-%d")
        ),
        "신호 조회 종료일": (
            reference_date.strftime("%Y-%m-%d")
        ),
        "신호 조회 일수": (
            recent_signal_days
        ),
        "당일 Weekly 신호": classify_signal(
            latest_row,
            "Weekly_Buy",
            "Weekly_Sell",
        ),
        "당일 Weekly+Monthly 신호": classify_signal(
            latest_row,
            "Weekly_Monthly_Buy",
            "Weekly_Monthly_Sell",
        ),
    }

    return {
        "data": df,
        "metrics": pd.DataFrame(metrics),
        "trades": trades,
        "latest": latest_status,
        "messages": messages,
    }
