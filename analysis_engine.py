# analysis_engine.py
"""
화면 출력이나 자동 파일 저장 없이 분석 결과를 반환합니다.

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
    가격 데이터 기준일 포함 최근 5일의 마지막 신호
    거래일이 아닌 달력일 기준

주의:
    기존 주간·월간 MACD 계산 방식을 유지합니다.
    같은 주·월의 마지막 가용 종가를 해당 기간 전체에
    배정하므로 데이터가 추가되면 과거 값과 신호가
    바뀔 수 있습니다.

    기존 백테스트는 이 신호를 그대로 사용하므로
    미래 정보 참조 가능성이 있는 기존 방식입니다.
    이번 수정은 이를 교정하는 변경이 아닙니다.

공매도·대차거래 조회 기능은 포함하지 않습니다.
"""

import numpy as np
import pandas as pd


RECENT_SIGNAL_DAYS = 5

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


# =========================================================
# 재무 엑셀 읽기·검증
# =========================================================

def read_financial_excel(excel_file, stock_name):
    """
    종목별 재무 시트를 읽고 검증합니다.

    반환값:
        분기 오름차순으로 정렬된 DataFrame

    목표가:
        음수는 0으로 보정합니다.
        100원 단위 반올림은 목표가 매핑 시 적용합니다.

    매출액·영업이익·순이익:
        입력값을 그대로 유지합니다.
        음수와 결측값도 그대로 유지합니다.
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

    # 완전히 빈 행만 제거합니다.
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
        .str.fullmatch(r"\d{4}\.[1-4]Q")
        .fillna(False)
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
            & cleaned.ne("")
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

    # YYYY.NQ는 문자열 정렬로도 분기 순서가 유지됩니다.
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
    """재무 데이터 및 분기 정보 관리."""

    @staticmethod
    def load_financial_data(excel_file, sheet_name):
        """분기별 재무 데이터를 반환합니다."""
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
        분기별 목표가 사전을 반환합니다.

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
            if pd.notna(price):
                target_prices[str(quarter)] = int(
                    round(float(price) / 100) * 100
                )
            else:
                target_prices[str(quarter)] = np.nan

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
        기존 함수명을 유지하면서 다음 컬럼을 추가합니다.

            분기
            목표가
            매출액
            영업이익
            순이익
            괴리율(%)

        주의:
            재무값은 같은 분기의 일봉 행에 매핑합니다.
            공시 당시 이용 가능했던 시점 기준 데이터는 아닙니다.
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

        quarterly = financial.set_index(
            "분기"
        )

        rounded_targets = (
            quarterly["목표가"]
            .map(
                lambda value: (
                    int(round(float(value) / 100) * 100)
                    if pd.notna(value)
                    else np.nan
                )
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

        timeframe_params = {
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

        defaults = timeframe_params.get(
            str(timeframe).lower(),
            timeframe_params["daily"],
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

        ema_fast = TechnicalIndicator.ema(
            result[price_col],
            fast,
        )

        ema_slow = TechnicalIndicator.ema(
            result[price_col],
            slow,
        )

        result["MACD"] = (
            ema_fast - ema_slow
        )

        result["Signal"] = (
            TechnicalIndicator.ema(
                result["MACD"],
                signal,
            )
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
    def add_weekly_macd(df):
        """
        기존 주간 계산 방식 유지.

        같은 주의 모든 날짜에 해당 주의 마지막 가용 종가로
        계산한 주간 값을 배정합니다.
        """
        weekly = (
            df[["종가"]]
            .resample("W-FRI")
            .last()
        )

        weekly = TechnicalIndicator.macd(
            weekly,
            timeframe="weekly",
        )

        weekly = weekly.rename(
            columns={
                "MACD": "Weekly_MACD",
                "Signal": "Weekly_Signal",
                "Oscillator": "Weekly_Oscillator",
            }
        )

        columns = [
            "Weekly_MACD",
            "Weekly_Signal",
            "Weekly_Oscillator",
        ]

        weekly = weekly[columns].copy()

        weekly.index = (
            weekly.index.to_period("W-FRI")
        )

        result = df.copy()

        periods = result.index.to_period(
            "W-FRI"
        )

        for column in columns:
            result[column] = periods.map(
                weekly[column]
            )

        return result

    @staticmethod
    def add_monthly_macd(df):
        """
        기존 월간 계산 방식 유지.

        같은 월의 모든 날짜에 해당 월의 마지막 가용 종가로
        계산한 월간 값을 배정합니다.
        """
        monthly = (
            df[["종가"]]
            .resample("ME")
            .last()
        )

        monthly = TechnicalIndicator.macd(
            monthly,
            timeframe="monthly",
        )

        monthly = monthly.rename(
            columns={
                "MACD": "Monthly_MACD",
                "Signal": "Monthly_Signal",
                "Oscillator": "Monthly_Oscillator",
            }
        )

        columns = [
            "Monthly_MACD",
            "Monthly_Signal",
            "Monthly_Oscillator",
        ]

        monthly = monthly[columns].copy()

        monthly.index = (
            monthly.index.to_period("M")
        )

        result = df.copy()

        periods = result.index.to_period(
            "M"
        )

        for column in columns:
            result[column] = periods.map(
                monthly[column]
            )

        return result


# =========================================================
# 매매 신호
# =========================================================

class SignalGenerator:

    @staticmethod
    def calculate_trend(osc_series):
        """
         1: Oscillator 증가
        -1: Oscillator 감소
         0: 변화 없음
        """
        trend = pd.Series(
            0,
            index=osc_series.index,
            dtype=int,
        )

        diff = osc_series.diff()

        trend.loc[diff > 0] = 1
        trend.loc[diff < 0] = -1

        return trend

    @staticmethod
    def maintain_weekly_trend(df):
        """기존 방식대로 주간 추세를 같은 주 전체에 배정."""
        result = df.copy()

        result["_year_week"] = (
            result.index.to_period("W-FRI")
        )

        weekly_trend = (
            result.groupby("_year_week")["Weekly_Trend"]
            .apply(
                lambda values: (
                    values.loc[values != 0].iloc[0]
                    if (values != 0).any()
                    else np.nan
                )
            )
            .ffill()
            .fillna(0)
            .astype(int)
        )

        result["Weekly_Trend"] = (
            result["_year_week"]
            .map(weekly_trend)
            .astype(int)
        )

        return result.drop(
            columns=["_year_week"]
        )

    @staticmethod
    def maintain_monthly_trend(df):
        """기존 방식대로 월간 추세를 같은 월 전체에 배정."""
        result = df.copy()

        result["_year_month"] = (
            result.index.to_period("M")
        )

        monthly_trend = (
            result.groupby("_year_month")["Monthly_Trend"]
            .apply(
                lambda values: (
                    values.loc[values != 0].iloc[0]
                    if (values != 0).any()
                    else np.nan
                )
            )
            .ffill()
            .fillna(0)
            .astype(int)
        )

        result["Monthly_Trend"] = (
            result["_year_month"]
            .map(monthly_trend)
            .astype(int)
        )

        return result.drop(
            columns=["_year_month"]
        )

    @staticmethod
    def identify_signal_changes(trend_series):
        """
        엄격한 추세 전환.

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
        previous_monthly = (
            df["Monthly_Trend"].shift(1)
        )

        monthly_turn_up = (
            previous_monthly.le(0)
            & df["Monthly_Trend"].eq(1)
        )

        monthly_turn_down = (
            previous_monthly.ge(0)
            & df["Monthly_Trend"].eq(-1)
        )

        buy = (
            df["Weekly_Trend"].eq(1)
            & monthly_turn_up
        )

        sell = (
            df["Weekly_Trend"].eq(-1)
            & monthly_turn_down
        )

        return buy, sell

    @staticmethod
    def add_signals(df):
        result = df.copy()

        for prefix in (
            "Daily",
            "Weekly",
            "Monthly",
        ):
            result[f"{prefix}_Trend"] = (
                SignalGenerator.calculate_trend(
                    result[f"{prefix}_Oscillator"]
                )
            )

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
        기존 백테스트 계산 방식 유지.

        - 신호 발생일 종가 매수·매도
        - 한 번에 한 포지션
        - 미청산 포지션 제외
        - 수수료·세금·슬리피지 제외
        """
        trades = []

        buy_col = f"{signal_type}_Buy"
        sell_col = f"{signal_type}_Sell"

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

        for index in range(len(df)):
            current_date = df.index[index]
            current_price = (
                df[price_col].iloc[index]
            )

            if monthly_filter is not None:
                current_monthly = (
                    df["Monthly_Trend"].iloc[index]
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
                df[buy_col].iloc[index] == 1
                and not in_position
            ):
                entry_date = current_date
                entry_price = current_price
                in_position = True

            elif (
                df[sell_col].iloc[index] == 1
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

        return pd.DataFrame(trades)

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
            "중립": count - len(win) - len(loss),
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
    """
    기준일 포함 최근 days일의 마지막 신호.

    days는 달력일 기준입니다.
    """
    empty_result = {
        "신호": "신호 없음",
        "발생일": None,
    }

    if (
        not isinstance(days, (int, np.integer))
        or isinstance(days, (bool, np.bool_))
        or days < 1
    ):
        raise ValueError(
            "신호 조회 일수는 1 이상의 정수여야 합니다."
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

    frame = frame.loc[
        ~frame.index.isna()
    ].sort_index()

    if frame.empty:
        return empty_result

    if reference_date is None:
        reference_date = (
            frame.index.max()
        )

    reference = normalize_timestamp(
        reference_date
    )

    start_date = (
        reference
        - pd.Timedelta(days=int(days) - 1)
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

    last_row = recent.iloc[-1]

    return {
        "신호": classify_signal(
            last_row,
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
    전체 분석 기간 중 최근 limit개 신호를 반환합니다.

    최근 5일로 제한하는 함수가 아닙니다.
    기존 app.py와 호환되도록 날짜 오름차순으로 반환합니다.
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

    df.index = (
        df.index.normalize()
    )

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

    기존 app.py 호출 방식과 호환됩니다.
    """
    if (
        not isinstance(
            recent_signal_days,
            (int, np.integer),
        )
        or isinstance(
            recent_signal_days,
            (bool, np.bool_),
        )
        or recent_signal_days < 1
    ):
        raise ValueError(
            "최근 신호 조회 일수는 "
            "1 이상의 정수여야 합니다."
        )

    recent_signal_days = int(
        recent_signal_days
    )

    df = prepare_price_frame(
        frame
    )

    messages = []

    # -----------------------------------------------------
    # 1. 재무 정보
    # -----------------------------------------------------

    if (
        excel_file is not None
        and str(excel_file) != ""
    ):
        # 내부에서 엑셀 검증 및 재무 데이터 읽기를 수행합니다.
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
