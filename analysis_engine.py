"""화면 출력이나 자동 파일 저장 없이 분석 결과를 반환합니다."""

import numpy as np
import pandas as pd


# =========================================================
# 재무 데이터
# =========================================================

class FinancialDataManager:
    """재무 데이터 및 분기 정보 관리"""

    @staticmethod
    def load_target_price(excel_file, sheet_name, stock_name=""):
        """
        엑셀에서 분기별 목표가를 읽습니다.

        - 음수 목표가는 0으로 보정
        - 빈 값은 NaN 유지
        - 100원 단위 반올림
        """
        df = pd.read_excel(
            excel_file,
            sheet_name=sheet_name,
        )

        required = {"분기", "목표가"}
        missing = required - set(df.columns)

        if missing:
            raise ValueError(
                "목표가 데이터 컬럼 누락: "
                + ", ".join(sorted(missing))
            )

        df = df.copy()
        df["분기"] = df["분기"].astype(str).str.strip()

        df["목표가"] = (
            pd.to_numeric(
                df["목표가"],
                errors="coerce",
            )
            .clip(lower=0)
        )

        target_prices = {}

        for quarter, price in zip(
            df["분기"],
            df["목표가"],
        ):
            if pd.notna(price):
                target_prices[quarter] = int(
                    round(price / 100) * 100
                )
            else:
                target_prices[quarter] = np.nan

        return target_prices

    @staticmethod
    def get_quarter(date):
        """날짜로부터 YYYY.NQ 형식의 분기를 계산합니다."""
        quarter = (date.month - 1) // 3 + 1
        return f"{date.year}.{quarter}Q"

    @staticmethod
    def calculate_divergence_rate(current_price, target_price):
        """
        목표가 대비 괴리율을 계산합니다.

        괴리율(%) = (현재가 - 목표가) / 목표가 × 100

        양수: 현재가가 목표가보다 높음
        음수: 현재가가 목표가보다 낮음
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
        """분기, 목표가, 괴리율 컬럼을 추가합니다."""
        df = df.copy()

        target_prices = FinancialDataManager.load_target_price(
            excel_file,
            sheet_name,
        )

        df["분기"] = df.index.map(
            FinancialDataManager.get_quarter
        )

        df["목표가"] = df["분기"].map(target_prices)

        # 목표가가 0인 경우 괴리율은 NaN으로 처리합니다.
        valid_target = df["목표가"].where(
            df["목표가"].ne(0)
        )

        df["괴리율(%)"] = (
            (df["종가"] - valid_target)
            / valid_target
            * 100
        )

        return df


# =========================================================
# 기술적 지표
# =========================================================

class TechnicalIndicator:
    """기술적 지표 계산 클래스"""

    @staticmethod
    def ema(series, span=12):
        """지수이동평균을 계산합니다."""
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
        """시간 단위별 MACD를 계산합니다."""
        df = df.copy()

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

        timeframe_str = str(timeframe).lower()

        defaults = timeframe_params.get(
            timeframe_str,
            timeframe_params["daily"],
        )

        fast = defaults["fast"] if fast is None else fast
        slow = defaults["slow"] if slow is None else slow
        signal = (
            defaults["signal"]
            if signal is None
            else signal
        )

        ema_fast = TechnicalIndicator.ema(
            df[price_col],
            fast,
        )

        ema_slow = TechnicalIndicator.ema(
            df[price_col],
            slow,
        )

        df["MACD"] = ema_fast - ema_slow

        df["Signal"] = TechnicalIndicator.ema(
            df["MACD"],
            signal,
        )

        df["Oscillator"] = (
            df["MACD"] - df["Signal"]
        )

        return df

    @staticmethod
    def add_daily_macd(df):
        """일간 MACD를 추가합니다."""
        df = TechnicalIndicator.macd(
            df,
            timeframe="daily",
        )

        return df.rename(
            columns={
                "MACD": "Daily_MACD",
                "Signal": "Daily_Signal",
                "Oscillator": "Daily_Oscillator",
            }
        )

    @staticmethod
    def add_weekly_macd(df):
        """
        주간 MACD를 추가합니다.

        기존 계산 방식을 유지합니다.
        같은 주의 날짜에는 해당 주의 마지막 가용 종가로
        계산한 동일한 주간 값을 배정합니다.

        주의:
        주중 데이터가 추가되면 같은 주의 과거 값도
        변경될 수 있습니다.
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
        weekly.index = weekly.index.to_period("W-FRI")

        result = df.copy()

        # map을 사용해 원래 일봉 인덱스를 유지합니다.
        periods = result.index.to_period("W-FRI")

        for column in columns:
            result[column] = periods.map(weekly[column])

        return result

    @staticmethod
    def add_monthly_macd(df):
        """
        월간 MACD를 추가합니다.

        기존 계산 방식을 유지합니다.
        같은 월의 날짜에는 해당 월의 마지막 가용 종가로
        계산한 동일한 월간 값을 배정합니다.

        주의:
        월중 데이터가 추가되면 같은 월의 과거 값도
        변경될 수 있습니다.
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
        monthly.index = monthly.index.to_period("M")

        result = df.copy()
        periods = result.index.to_period("M")

        for column in columns:
            result[column] = periods.map(monthly[column])

        return result


# =========================================================
# 매매 신호
# =========================================================

class SignalGenerator:
    """매매 신호 생성 클래스"""

    @staticmethod
    def calculate_trend(osc_series):
        """
        Oscillator 변화 방향을 계산합니다.

         1: 증가
        -1: 감소
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
        """주간 추세를 해당 주 전체에 동일하게 배정합니다."""
        df = df.copy()
        df["year_week"] = df.index.to_period("W-FRI")

        weekly_trend = (
            df.groupby("year_week")["Weekly_Trend"]
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

        df["Weekly_Trend"] = (
            df["year_week"]
            .map(weekly_trend)
            .astype(int)
        )

        return df.drop(columns=["year_week"])

    @staticmethod
    def maintain_monthly_trend(df):
        """월간 추세를 해당 월 전체에 동일하게 배정합니다."""
        df = df.copy()
        df["year_month"] = df.index.to_period("M")

        monthly_trend = (
            df.groupby("year_month")["Monthly_Trend"]
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

        df["Monthly_Trend"] = (
            df["year_month"]
            .map(monthly_trend)
            .astype(int)
        )

        return df.drop(columns=["year_month"])

    @staticmethod
    def identify_signal_changes(trend_series):
        """
        엄격한 추세 전환을 감지합니다.

        매수: -1 → +1
        매도: +1 → -1

        0을 거치는 전환은 신호로 인정하지 않습니다.
        """
        previous = trend_series.shift(1)

        buy = (
            (previous == -1)
            & (trend_series == 1)
        )

        sell = (
            (previous == 1)
            & (trend_series == -1)
        )

        return buy, sell

    @staticmethod
    def identify_weekly_monthly_signals(df):
        """
        주간 추세와 월간 추세 전환을 결합합니다.

        매수:
        - 주간 추세가 +1
        - 월간 추세가 0 이하에서 +1로 전환

        매도:
        - 주간 추세가 -1
        - 월간 추세가 0 이상에서 -1로 전환
        """
        previous_monthly = df["Monthly_Trend"].shift(1)

        monthly_turn_up = (
            (previous_monthly <= 0)
            & (df["Monthly_Trend"] == 1)
        )

        monthly_turn_down = (
            (previous_monthly >= 0)
            & (df["Monthly_Trend"] == -1)
        )

        buy = (
            (df["Weekly_Trend"] == 1)
            & monthly_turn_up
        )

        sell = (
            (df["Weekly_Trend"] == -1)
            & monthly_turn_down
        )

        return buy, sell

    @staticmethod
    def add_signals(df):
        """일간·주간·월간·결합 신호를 추가합니다."""
        df = df.copy()

        for prefix in ("Daily", "Weekly", "Monthly"):
            df[f"{prefix}_Trend"] = (
                SignalGenerator.calculate_trend(
                    df[f"{prefix}_Oscillator"]
                )
            )

        df = SignalGenerator.maintain_weekly_trend(df)
        df = SignalGenerator.maintain_monthly_trend(df)

        for prefix in ("Daily", "Weekly", "Monthly"):
            buy, sell = (
                SignalGenerator.identify_signal_changes(
                    df[f"{prefix}_Trend"]
                )
            )

            df[f"{prefix}_Buy"] = buy.astype(int)
            df[f"{prefix}_Sell"] = sell.astype(int)

        combined_buy, combined_sell = (
            SignalGenerator.identify_weekly_monthly_signals(df)
        )

        df["Weekly_Monthly_Buy"] = (
            combined_buy.astype(int)
        )

        df["Weekly_Monthly_Sell"] = (
            combined_sell.astype(int)
        )

        return df


# =========================================================
# 백테스트
# =========================================================

class Backtester:
    """기존 신호 기반 백테스트"""

    @staticmethod
    def backtest(
        df,
        signal_type="Weekly",
        monthly_filter=None,
        price_col="종가",
    ):
        """
        신호 발생일 종가로 매수·매도합니다.

        기존 방식 유지:
        - 한 번에 한 포지션만 보유
        - 미청산 포지션은 거래 내역에서 제외
        - 수수료·세금·슬리피지 제외
        """
        df = df.copy()
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
            current_price = df[price_col].iloc[index]

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
    def calculate_metrics(trades_df, strategy_name):
        """
        성과 지표를 계산합니다.

        기존 계산 방식 유지:
        - 총수익률은 거래별 수익률의 단순 합계
        - 손익비는 기존 코드의 순수익 / 총손실 절댓값 방식
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

        total_profit = trades_df["수익(원)"].sum()
        total_loss = loss["수익(원)"].sum()

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
                round(win["수익률(%)"].mean(), 1)
                if not win.empty
                else 0
            ),
            "평균손실(%)": (
                round(loss["수익률(%)"].mean(), 1)
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
            "손익비": round(profit_factor, 1),
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
    """단일 행의 매수·매도 신호를 반환합니다."""
    if row.get(buy_col, 0) == 1:
        return "매수"

    if row.get(sell_col, 0) == 1:
        return "매도"

    return "신호 없음"


def get_recent_signal(
    df,
    buy_col="Weekly_Buy",
    sell_col="Weekly_Sell",
    days=7,
    reference_date=None,
):
    """
    기준일 포함 최근 days일 안의 마지막 신호를 반환합니다.

    days는 거래일 수가 아닌 달력일 수입니다.

    reference_date가 없으면 데이터의 마지막 날짜를 사용합니다.
    """
    empty_result = {
        "신호": "신호 없음",
        "발생일": None,
    }

    if days < 1:
        raise ValueError("신호 조회 일수는 1 이상이어야 합니다.")

    if df.empty:
        return empty_result

    missing = {buy_col, sell_col} - set(df.columns)

    if missing:
        raise ValueError(
            "신호 컬럼 누락: "
            + ", ".join(sorted(missing))
        )

    if reference_date is None:
        reference_date = df.index.max()

    reference = pd.Timestamp(reference_date).normalize()
    start_date = reference - pd.Timedelta(days=days - 1)

    # 기준일의 모든 시간대를 포함하도록 날짜 단위로 비교합니다.
    dates = df.index.normalize()

    date_mask = (
        (dates >= start_date)
        & (dates <= reference)
    )

    signal_mask = (
        df[buy_col].eq(1)
        | df[sell_col].eq(1)
    )

    recent = (
        df.loc[date_mask & signal_mask]
        .sort_index()
    )

    if recent.empty:
        return empty_result

    last_row = recent.iloc[-1]

    return {
        "신호": classify_signal(
            last_row,
            buy_col,
            sell_col,
        ),
        "발生일": None,
        "발생일": recent.index[-1].strftime("%Y-%m-%d"),
    }


def recent_signals(df, combined=False, limit=10):
    """
    최근 매매 신호 표를 반환합니다.

    기존 app.py와의 호환성을 위해 날짜 오름차순으로 반환합니다.
    최근 7일로 제한하지 않고 전체 분석 기간에서 limit개를 조회합니다.
    """
    prefix = (
        "Weekly_Monthly"
        if combined
        else "Weekly"
    )

    buy_col = f"{prefix}_Buy"
    sell_col = f"{prefix}_Sell"

    empty_columns = [
        "날짜",
        "종가",
        "목표가",
        "괴리율(%)",
        "신호",
        "Daily_Trend",
        "Weekly_Trend",
        "Monthly_Trend",
    ]

    if limit < 1 or df.empty:
        return pd.DataFrame(columns=empty_columns)

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
        return pd.DataFrame(columns=empty_columns)

    result["날짜"] = result.index.strftime("%Y-%m-%d")

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
        for column in empty_columns
        if column in result.columns
    ]

    return result[columns].reset_index(drop=True)


# =========================================================
# 입력 검증
# =========================================================

def validate_excel(excel_file, stock_name):
    """재무 엑셀의 시트와 필수 컬럼을 검증합니다."""
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

    required = {
        "분기",
        "목표가",
        "영업이익",
    }

    missing = required - set(financial.columns)

    if missing:
        raise ValueError(
            "엑셀에 필요한 컬럼이 없습니다: "
            + ", ".join(sorted(missing))
        )

    if financial.empty:
        raise ValueError("재무 엑셀에 데이터가 없습니다.")

    quarters = (
        financial["분기"]
        .astype(str)
        .str.strip()
    )

    valid_quarters = quarters.str.fullmatch(
        r"\d{4}\.[1-4]Q"
    )

    if not valid_quarters.all():
        raise ValueError(
            "분기는 '2026.1Q' 형식으로 입력하세요."
        )

    if quarters.duplicated().any():
        raise ValueError(
            "엑셀에 중복된 분기가 있습니다."
        )

    for column in ("목표가", "영업이익"):
        numeric = pd.to_numeric(
            financial[column],
            errors="coerce",
        )

        invalid = (
            financial[column].notna()
            & numeric.isna()
        )

        if invalid.any():
            raise ValueError(
                f"'{column}'에는 숫자 또는 빈칸만 입력하세요."
            )


def prepare_price_frame(frame):
    """가격 데이터의 인덱스와 숫자형 컬럼을 정리합니다."""
    if frame is None or frame.empty:
        raise ValueError("가격 데이터가 없습니다.")

    df = frame.copy()

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

    # 숫자형 인덱스를 날짜로 오인하지 않도록 차단합니다.
    if not isinstance(df.index, pd.DatetimeIndex):
        if pd.api.types.is_numeric_dtype(df.index.dtype):
            raise ValueError(
                "가격 데이터 인덱스는 날짜 형식이어야 합니다."
            )

        df.index = pd.to_datetime(
            df.index,
            errors="coerce",
        )

    df = df.loc[~df.index.isna()].copy()

    # 시간대가 있으면 한국 시간으로 맞춘 뒤 제거합니다.
    if df.index.tz is not None:
        df.index = (
            df.index
            .tz_convert("Asia/Seoul")
            .tz_localize(None)
        )

    df.index = df.index.normalize()
    df = df.sort_index()

    if df.index.has_duplicates:
        df = df.loc[
            ~df.index.duplicated(keep="last")
        ].copy()

    for column in required:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    df[list(required)] = (
        df[list(required)]
        .replace([np.inf, -np.inf], np.nan)
    )

    df = df.dropna(subset=list(required))
    df = df.loc[df["종가"] > 0].copy()

    if df.empty:
        raise ValueError(
            "유효한 가격 데이터가 없습니다."
        )

    df["날짜"] = df.index.strftime("%Y-%m-%d")

    return df


# =========================================================
# 분석 실행
# =========================================================

def analyze_frame(
    frame,
    stock_name,
    excel_file=None,
    recent_signal_days=7,
):
    """
    가격 데이터와 재무 엑셀을 분석합니다.

    latest의 주간·결합 신호:
        마지막 거래일 하루의 신호가 아니라,
        기준일 포함 최근 recent_signal_days일의 마지막 신호.

    기존 app.py 호출과 호환됩니다.
    """
    if recent_signal_days < 1:
        raise ValueError(
            "최근 신호 조회 일수는 1 이상이어야 합니다."
        )

    df = prepare_price_frame(frame)
    messages = []

    # -----------------------------------------------------
    # 재무 정보
    # -----------------------------------------------------

    if excel_file is not None and str(excel_file) != "":
        validate_excel(
            excel_file,
            stock_name,
        )

        df = FinancialDataManager.add_quarter_and_target_price(
            df,
            excel_file,
            stock_name,
        )

    else:
        df["분기"] = df.index.map(
            FinancialDataManager.get_quarter
        )

        df["목표가"] = np.nan
        df["괴리율(%)"] = np.nan

        messages.append(
            "재무 엑셀이 없어 목표가·괴리율·영업이익은 "
            "표시하지 않습니다."
        )

    # -----------------------------------------------------
    # 지표와 신호
    # -----------------------------------------------------

    df = TechnicalIndicator.add_daily_macd(df)
    df = TechnicalIndicator.add_weekly_macd(df)
    df = TechnicalIndicator.add_monthly_macd(df)

    df = SignalGenerator.add_signals(df)

    # -----------------------------------------------------
    # 백테스트
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
    # 최근 7일 신호
    # -----------------------------------------------------

    latest_row = df.iloc[-1]
    reference_date = df.index[-1]

    signal_start = (
        reference_date
        - pd.Timedelta(days=recent_signal_days - 1)
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
        "기준일": reference_date.strftime("%Y-%m-%d"),
        "종가": float(latest_row["종가"]),

        # 기존 app.py가 읽는 키입니다.
        # 최근 조회 기간 안에 발생한 마지막 신호를 반환합니다.
        "Weekly 신호": weekly_recent["신호"],
        "Weekly+Monthly 신호": combined_recent["신호"],

        # 신호 발생일
        "Weekly 신호일": weekly_recent["발생일"],
        "Weekly+Monthly 신호일": combined_recent["발생일"],

        # 조회 기간
        "신호 조회 시작일": signal_start.strftime("%Y-%m-%d"),
        "신호 조회 종료일": reference_date.strftime("%Y-%m-%d"),
        "신호 조회 일수": recent_signal_days,

        # 마지막 거래일 하루의 신호도 별도로 보존합니다.
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
