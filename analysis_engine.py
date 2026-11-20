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
    def enforce_signal_pairing(buy, sell):
        """
        매수·매도 신호가 번갈아 짝을 이루도록 정리합니다.

        - 직전 신호와 같은 종류의 신호는 무시합니다.
          (동일한 매수·매도 신호가 계속 발생하는 것을 제거)
        - 매수 뒤에는 매도, 매도 뒤에는 매수만 올 수 있습니다.
        - 첫 신호는 매수·매도 어느 쪽이든 허용합니다.
          (첫 신호가 매도여도 백테스트는 포지션이 없어 무시합니다.)
        - 같은 날 매수와 매도가 겹치면 방향을 판단할 수 없으므로
          두 신호 모두 제거합니다.
        """
        paired_buy = buy.copy()
        paired_sell = sell.copy()

        #  0: 신호 없음,  1: 직전 신호가 매수,  -1: 직전 신호가 매도
        last_signal = 0

        for timestamp in buy.index:
            day_buy = bool(buy.loc[timestamp])
            day_sell = bool(sell.loc[timestamp])

            if day_buy and day_sell:
                paired_buy.loc[timestamp] = False
                paired_sell.loc[timestamp] = False
                continue

            if day_buy:
                if last_signal == 1:
                    # 동일한 매수 신호는 무시합니다.
                    paired_buy.loc[timestamp] = False
                else:
                    last_signal = 1
            elif day_sell:
                if last_signal == -1:
                    # 동일한 매도 신호는 무시합니다.
                    paired_sell.loc[timestamp] = False
                else:
                    last_signal = -1

        return paired_buy, paired_sell

    @staticmethod
    def expand_turn_window(turn):
        """
        월간 전환 조건을 전환일부터 그 달의 마지막 날까지
        앞쪽 방향으로만 확장합니다.

        다음 달이 되면 창을 초기화하며, 전환이 없던 달은
        조건 구간이 아닙니다.
        """
        window = pd.Series(False, index=turn.index)

        active_month = None
        active = False

        for timestamp in turn.index:
            month = timestamp.to_period("M")

            if month != active_month:
                active_month = month
                active = False

            if turn.loc[timestamp]:
                active = True

            window.loc[timestamp] = active

        return window

    @staticmethod
    def identify_weekly_monthly_signals(df):
        """
        주간 추세와 월간 추세 전환을 결합합니다.

        매수:
        - 월간 추세가 감소(-1)하다가 증가(+1)로 전환한 달에서
        - 주간 추세가 전주보다 증가(+1)할 때

        매도:
        - 월간 추세가 증가(+1)하다가 감소(-1)로 전환한 달에서
        - 주간 추세가 전주보다 감소(-1)할 때

        월간 전환은 전환일부터 그 달의 마지막 날까지 조건 구간으로
        유지하므로, 같은 구간 안에서 주간 추세가 증가 조건을
        여러 번 충족해도 첫 신호만 인정합니다
        (동일 신호 반복 무시).

        또한 매수와 매도가 번갈아 짝을 이루도록 정리합니다.
        """
        previous_monthly = df["Monthly_Trend"].shift(1)

        turn_up = (
            (previous_monthly == -1)
            & (df["Monthly_Trend"] == 1)
        )

        turn_down = (
            (previous_monthly == 1)
            & (df["Monthly_Trend"] == -1)
        )

        buy_window = SignalGenerator.expand_turn_window(turn_up)
        sell_window = SignalGenerator.expand_turn_window(turn_down)

        buy = (
            buy_window
            & (df["Weekly_Trend"] == 1)
        )

        sell = (
            sell_window
            & (df["Weekly_Trend"] == -1)
        )

        return SignalGenerator.enforce_signal_pairing(
            buy,
            sell,
        )

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

def trend_flag(row, column):
    """
    추세 값(+1 상승 / -1 하락 / 0 없음)을 JSON 친화적 정수로 돌려줍니다.
    값이 없거나 계산 불가면 None입니다(화면에서 회색 원).
    """
    if column not in row.index:
        return None

    value = row.get(column)

    if value is None or pd.isna(value):
        return None

    try:
        number = float(value)
    except (TypeError, ValueError):
        return None

    if not np.isfinite(number):
        return None

    return int(np.sign(number))


NOMAD_SURGE_RATIO = 2.0    # 거래량 폭등: 전일 대비 2배 이상
NOMAD_DRY_RATIO = 0.25     # 거래량 급감: 폭등일 거래량의 25% 이하
NOMAD_RECENT_DAYS = 3      # 폭등일은 마지막 거래일 기준 최근 3거래일 이내
NOMAD_DRY_DAYS = (1, 2)    # 폭등일 이후 1~2일 연속 음봉
NOMAD_CHART_MONTHS = 18    # charts.DISPLAY_MONTHS와 같아야 합니다(차트 표시 구간)
NOMAD_RECENT_WINDOW = 7    # 전체 결과 요약에서 '최근 신호'로 볼 기간(달력일)


def nomad_patterns(
    df,
    surge_ratio=NOMAD_SURGE_RATIO,
    dry_ratio=NOMAD_DRY_RATIO,
    recent_days=None,
    dry_days=NOMAD_DRY_DAYS,
):
    """
    유목민 매수 패턴을 전 구간에서 찾습니다. (차트 별표·성과 집계용)

    규칙
        ① 폭등일 g: 거래량[g] >= 거래량[g-1] x 2   (전일 대비 2배 이상)
        ② 폭등일 g: 종가[g] > 종가[g-1]            (상승)
        ③ 이후 1~2거래일 연속 음봉: 종가 < 시가
        ④ 그 음봉일들의 거래량 <= 거래량[g] x 0.25
        ⑤ recent_days가 주어지면 데이터 끝에서 그만큼 이내의 폭등만 인정

    신호는 마지막 음봉일(패턴이 완성된 날)에 표시합니다.
    같은 폭등일에 1일 완성과 2일 완성이 겹치면 더 늦은 쪽(긴 음봉)만 남겨
    한 번의 패턴이 두 번 집계되지 않게 합니다.
    """
    frame = pd.DataFrame(index=df.index)
    columns = (
        "유목민신호",
        "유목민신호일수",
        "유목민 거래량배수",
        "유목민 급감비율",
        "유목민 폭등상승률",
        "유목민 음봉등락률",
        "유목민 폭등일",
        "유목민 음봉일",
        "유목민 음봉일목록",
    )

    for column in columns:
        frame[column] = 0 if column in ("유목민신호", "유목민신호일수") else None

    if df is None or df.empty:
        return frame

    if not {"시가", "종가", "거래량"}.issubset(df.columns):
        return frame

    volume = pd.to_numeric(df["거래량"], errors="coerce").to_numpy(dtype=float)
    close = pd.to_numeric(df["종가"], errors="coerce").to_numpy(dtype=float)
    open_price = pd.to_numeric(df["시가"], errors="coerce").to_numpy(dtype=float)
    dates = pd.to_datetime(df.index)

    count = len(df)
    last = count - 1
    candidates = {}

    for index in range(count):
        for days in sorted(dry_days, reverse=True):
            spike = index - days
            base = spike - 1

            if base < 0:
                continue

            if recent_days is not None and last - spike > recent_days:
                continue

            values = (volume[base], volume[spike], close[base], close[spike])

            if not all(np.isfinite(values)) or values[0] <= 0 or values[1] <= 0:
                continue

            if volume[spike] < volume[base] * surge_ratio:
                continue

            if close[spike] <= close[base]:
                continue

            dry_slice = range(spike + 1, index + 1)

            bearish = all(
                np.isfinite(open_price[j])
                and np.isfinite(close[j])
                and close[j] < open_price[j]
                for j in dry_slice
            )

            if not bearish:
                continue

            quiet = all(
                np.isfinite(volume[j]) and volume[j] <= volume[spike] * dry_ratio
                for j in dry_slice
            )

            if not quiet:
                continue

            if spike not in candidates or index >= candidates[spike][0]:
                candidates[spike] = (index, days)

            break

    for spike, (index, days) in sorted(candidates.items()):
        dry_slice = range(spike + 1, index + 1)
        day_change = 0.0

        if index >= 1 and np.isfinite(close[index - 1]) and close[index - 1]:
            day_change = (close[index] / close[index - 1] - 1) * 100

        frame.iloc[index, frame.columns.get_loc("유목민신호")] = 1
        frame.iloc[index, frame.columns.get_loc("유목민신호일수")] = days
        frame.iloc[index, frame.columns.get_loc("유목민 거래량배수")] = round(
            volume[spike] / volume[spike - 1], 2
        )
        frame.iloc[index, frame.columns.get_loc("유목민 급감비율")] = round(
            max(volume[j] for j in dry_slice) / volume[spike] * 100, 1
        )
        frame.iloc[index, frame.columns.get_loc("유목민 폭등상승률")] = round(
            (close[spike] / close[spike - 1] - 1) * 100, 2
        )
        frame.iloc[index, frame.columns.get_loc("유목민 음봉등락률")] = round(
            day_change, 2
        )
        frame.iloc[index, frame.columns.get_loc("유목민 폭등일")] = dates[
            spike
        ].strftime("%Y-%m-%d")
        frame.iloc[index, frame.columns.get_loc("유목민 음봉일")] = dates[
            index
        ].strftime("%Y-%m-%d")
        frame.iloc[index, frame.columns.get_loc("유목민 음봉일목록")] = ", ".join(
            dates[j].strftime("%Y-%m-%d") for j in dry_slice
        )

    return frame


def recent_nomad_signal(df, days=NOMAD_RECENT_WINDOW):
    """
    최근 days일(달력일) 안에 발생한 마지막 유목민 매수 신호를 돌려줍니다.

    전체 결과 요약의 '유목민 신호' 열에서 사용합니다.
    신호가 없거나 데이터가 모자라면 신호 없음입니다.
    """
    result = {
        "신호": "신호 없음",
        "발생일": None,
        "음봉일수": None,
        "경과일": None,
        "거래량배수": None,
        "급감비율": None,
    }

    if df is None or df.empty or not {"시가", "종가", "거래량"}.issubset(df.columns):
        return result

    patterns = nomad_patterns(df)

    if patterns.empty:
        return result

    dates = pd.to_datetime(df.index)
    reference = dates[-1]
    cutoff = reference - pd.Timedelta(days=days - 1)
    flags = patterns["유목민신호"].to_numpy(dtype=int) == 1
    hits = [index for index in np.flatnonzero(flags) if dates[index] >= cutoff]

    if not hits:
        return result

    index = hits[-1]

    result.update({
        "신호": "매수",
        "발생일": dates[index].strftime("%Y-%m-%d"),
        "음봉일수": int(patterns["유목민신호일수"].iat[index]),
        "경과일": int((reference - dates[index]).days),
        "거래량배수": float(patterns["유목민 거래량배수"].iat[index]),
        "급감비율": float(patterns["유목민 급감비율"].iat[index]),
    })

    return result


def pattern_index(patterns):
    """nomad_patterns 결과에서 신호가 1인 위치(bool 배열)를 돌려줍니다."""
    return patterns["유목민신호"].to_numpy(dtype=int) == 1


NOMAD_FORWARD_DAYS = (5, 10, 20)


def nomad_signal_stats(
    df,
    horizons=NOMAD_FORWARD_DAYS,
    surge_ratio=NOMAD_SURGE_RATIO,
    dry_ratio=NOMAD_DRY_RATIO,
    dry_days=NOMAD_DRY_DAYS,
    chart_months=NOMAD_CHART_MONTHS,
):
    """
    과거 유목민 매수 신호가 난 뒤 5·10·20거래일 수익률을 집계합니다.

    - 진입: 신호 다음 거래일 시가 (신호는 종가 확정 후 알 수 있으므로)
    - 청산: 신호일 + N거래일 종가
    - 기준선: 같은 진입 규칙으로 전 구간 아무 날에나 진입했을 때의
      상승 비율. 이보다 낮으면 신호에 우위가 없다는 뜻입니다.
    """
    empty = {
        "신호수": 0,
        "차트신호수": 0,
        "차트개월": chart_months,
        "차트시작일": None,
        "기간": None,
        "보유기간": list(horizons),
        "요약": [],
        "신호목록": [],
    }

    if df is None or df.empty or not {"시가", "종가", "거래량"}.issubset(df.columns):
        return empty

    patterns = nomad_patterns(
        df,
        surge_ratio=surge_ratio,
        dry_ratio=dry_ratio,
        dry_days=dry_days,
    )

    close = pd.to_numeric(df["종가"], errors="coerce").to_numpy(dtype=float)
    open_price = pd.to_numeric(df["시가"], errors="coerce").to_numpy(dtype=float)
    dates = pd.to_datetime(df.index)
    count = len(df)

    empty["기간"] = [
        dates[0].strftime("%Y-%m-%d"),
        dates[-1].strftime("%Y-%m-%d"),
    ]

    # 차트는 최근 chart_months개월만 그립니다. 별표와 숫자를 맞추기 위해
    # 차트 표시 구간 안의 신호 수를 따로 셉니다.
    chart_start = dates[-1] - pd.DateOffset(months=chart_months)
    empty["차트시작일"] = chart_start.strftime("%Y-%m-%d")

    signal_index = np.flatnonzero(pattern_index(patterns))
    records = []

    for index in signal_index:
        entry_index = index + 1

        if entry_index < count and np.isfinite(open_price[entry_index]):
            entry = float(open_price[entry_index])
        elif np.isfinite(close[index]):
            entry = float(close[index])
            entry_index = index
        else:
            continue

        if entry <= 0:
            continue

        record = {
            "폭등일": patterns["유목민 폭등일"].iat[index],
            "음봉일": patterns["유목민 음봉일"].iat[index],
            "음봉일수": int(patterns["유목민신호일수"].iat[index]),
            "거래량배수": float(patterns["유목민 거래량배수"].iat[index]),
            "급감비율": float(patterns["유목민 급감비율"].iat[index]),
            "폭등상승률": float(patterns["유목민 폭등상승률"].iat[index]),
            "진입일": dates[entry_index].strftime("%Y-%m-%d"),
            "진입가": round(entry, 0),
            "차트표시": bool(dates[index] >= chart_start),
        }

        for horizon in horizons:
            exit_index = index + horizon
            value = None

            if exit_index < count and np.isfinite(close[exit_index]):
                value = (close[exit_index] / entry - 1.0) * 100.0

            record[f"수익률{horizon}"] = None if value is None else round(value, 2)

        records.append(record)

    summary = []

    for horizon in horizons:
        values = [
            record[f"수익률{horizon}"]
            for record in records
            if record[f"수익률{horizon}"] is not None
        ]

        baseline_values = []

        for index in range(count - horizon - 1):
            entry = open_price[index + 1]
            exit_price = close[index + horizon]

            if np.isfinite(entry) and np.isfinite(exit_price) and entry > 0:
                baseline_values.append(exit_price / entry - 1.0)

        baseline = (
            sum(1 for value in baseline_values if value > 0) / len(baseline_values) * 100
            if baseline_values else None
        )

        if values:
            series = pd.Series(values)
            win_rate = float((series > 0).mean() * 100)
            average = float(series.mean())
        else:
            win_rate = average = None

        summary.append({
            "보유일수": horizon,
            "표본": len(values),
            "승률": None if win_rate is None else round(win_rate, 1),
            "평균": None if average is None else round(average, 2),
            "중앙값": None if not values else round(float(pd.Series(values).median()), 2),
            "최선": None if not values else round(float(max(values)), 2),
            "최악": None if not values else round(float(min(values)), 2),
            "기준선": None if baseline is None else round(float(baseline), 1),
            "초과": (
                None
                if win_rate is None or baseline is None
                else round(win_rate - float(baseline), 1)
            ),
        })

    return {
        "신호수": len(records),
        "차트신호수": sum(1 for record in records if record["차트표시"]),
        "차트개월": chart_months,
        "차트시작일": empty["차트시작일"],
        "기간": empty["기간"],
        "보유기간": list(horizons),
        "요약": summary,
        "신호목록": records,
    }


def add_nomad_signal(df, **kwargs):
    """유목민 매수 신호(0/1)와 음봉일수를 컬럼으로 추가합니다."""
    frame = df.copy()
    patterns = nomad_patterns(frame, **kwargs)

    frame["유목민신호"] = patterns["유목민신호"].astype(int)
    frame["유목민신호일수"] = patterns["유목민신호일수"].astype(int)

    return frame


def detect_nomad_buy(
    df,
    surge_ratio=NOMAD_SURGE_RATIO,
    dry_ratio=NOMAD_DRY_RATIO,
    recent_days=None,
    dry_days=NOMAD_DRY_DAYS,
):
    """
    마지막 거래일 기준 유목민 매수 신호를 판정합니다.

    최근 3거래일 이내에 '전일 대비 거래량 2배 이상 + 상승'이 나오고,
    그 뒤 1~2일 연속 음봉이면서 거래량이 폭등일의 25% 이하로 줄면
    매수 신호로 봅니다.
    """
    status = {
        "유목민 매수": False,
        "유목민 폭등일": None,
        "유목민 음봉일": None,
        "유목민 음봉일수": None,
        "유목민 음봉일목록": None,
        "유목민 거래량배수": None,
        "유목민 급감비율": None,
        "유목민 폭등상승률": None,
        "유목민 음봉등락률": None,
        "유목민 판정": "",
    }

    if df is None or len(df) < 3 or not {"시가", "종가", "거래량"}.issubset(df.columns):
        status["유목민 판정"] = "판정에 필요한 데이터가 부족합니다."
        return status

    patterns = nomad_patterns(
        df,
        surge_ratio=surge_ratio,
        dry_ratio=dry_ratio,
        recent_days=recent_days,
        dry_days=dry_days,
    )

    last_row = patterns.iloc[-1]

    if int(last_row["유목민신호"]) == 1:
        status.update({
            "유목민 매수": True,
            "유목민 폭등일": last_row["유목민 폭등일"],
            "유목민 음봉일": last_row["유목민 음봉일"],
            "유목민 음봉일수": int(last_row["유목민신호일수"]),
            "유목민 음봉일목록": last_row["유목민 음봉일목록"],
            "유목민 거래량배수": float(last_row["유목민 거래량배수"]),
            "유목민 급감비율": float(last_row["유목민 급감비율"]),
            "유목민 폭등상승률": float(last_row["유목민 폭등상승률"]),
            "유목민 음봉등락률": float(last_row["유목민 음봉등락률"]),
            "유목민 판정": "조건 충족",
        })
        return status

    # 불충족 사유: 가장 조건에 가까운 후보(사유가 적은 쪽)를 설명합니다.
    volume = pd.to_numeric(df["거래량"], errors="coerce").to_numpy(dtype=float)
    close = pd.to_numeric(df["종가"], errors="coerce").to_numpy(dtype=float)
    open_price = pd.to_numeric(df["시가"], errors="coerce").to_numpy(dtype=float)
    last = len(df) - 1

    def diagnose(days):
        spike = last - days
        base = spike - 1

        if base < 0:
            return ["비교할 과거 데이터 부족"]

        if not np.isfinite(volume[base]) or volume[base] <= 0:
            return ["거래량 결측"]

        reasons = []

        if volume[spike] < volume[base] * surge_ratio:
            reasons.append(f"거래량 폭등 미달({volume[spike] / volume[base]:.2f}배)")

        if close[spike] <= close[base]:
            reasons.append("폭등일 상승 아님")

        dry_slice = range(spike + 1, last + 1)
        bearish = all(
            np.isfinite(open_price[j]) and close[j] < open_price[j]
            for j in dry_slice
        )

        if not bearish:
            reasons.append("음봉 아님")

        worst = max(volume[j] for j in dry_slice) / volume[spike] * 100

        if worst > dry_ratio * 100:
            reasons.append(f"거래량 급감 미달({worst:.1f}%)")

        return reasons

    best = min((diagnose(days) for days in dry_days), key=len)

    status["유목민 판정"] = " / ".join(best) if best else "조건 충족"

    return status


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
    df = add_nomad_signal(df)

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

    # 전일 종가와 전일대비 등락률
    previous_close = None
    previous_close_date = None
    change_pct = None

    if len(df) >= 2:
        previous_close = float(df["종가"].iloc[-2])
        previous_close_date = df.index[-2].strftime("%Y-%m-%d")

        if previous_close:
            change_pct = (
                float(latest_row["종가"]) / previous_close - 1.0
            ) * 100.0

    latest_status = {
        "종목명": stock_name,
        "기준일": reference_date.strftime("%Y-%m-%d"),
        "종가": float(latest_row["종가"]),
        "전일 종가": previous_close,
        "전일 종가일": previous_close_date,
        "전일대비(%)": (
            round(change_pct, 2)
            if change_pct is not None
            else None
        ),

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

        # 추세 신호등용 값: 마지막 거래일 기준
        # (+1 상승 / -1 하락 / 0 보합 / None 계산 불가)
        "Daily_Trend": trend_flag(latest_row, "Daily_Trend"),
        "Weekly_Trend": trend_flag(latest_row, "Weekly_Trend"),
        "Monthly_Trend": trend_flag(latest_row, "Monthly_Trend"),
    }

    # 유목민 매수(거래량 폭등 → 급감 음봉) 판정
    latest_status.update(detect_nomad_buy(df))

    # 최근 7일 이내 유목민 신호 (전체 결과 요약용)
    recent_nomad = recent_nomad_signal(df)

    latest_status["유목민 신호"] = recent_nomad["신호"]
    latest_status["유목민 신호일"] = recent_nomad["발생일"]
    latest_status["유목민 신호 음봉일수"] = recent_nomad["음봉일수"]
    latest_status["유목민 신호 경과일"] = recent_nomad["경과일"]

    # 과거 신호의 5·10·20거래일 성과
    nomad_stats = nomad_signal_stats(df)

    return {
        "data": df,
        "metrics": pd.DataFrame(metrics),
        "trades": trades,
        "latest": latest_status,
        "nomad_stats": nomad_stats,
        "messages": messages,
    }
