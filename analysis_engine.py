"""화면 출력이나 자동 파일 저장 없이 분석 결과를 반환합니다."""
from datetime import datetime, timedelta
import os

import numpy as np
import pandas as pd

class FinancialDataManager:
    """재무 데이터 및 분기 정보 관리"""
    
    @staticmethod
    def load_target_price(excel_file, sheet_name, stock_name):
        """
        엑셀 파일에서 목표가 데이터 읽기
        NaN 값 처리 및 백의 자리 반올림
        """
        try:
            df = pd.read_excel(excel_file, sheet_name=sheet_name)
            
            # 음수 목표가는 0으로 보정합니다.
            # 빈 값은 NaN으로 유지합니다.
            if "목표가" in df.columns:
                df["목표가"] = (
                    pd.to_numeric(df["목표가"], errors="coerce")
                    .clip(lower=0)
                )

            # 분기(분기)를 인덱스로 설정
            if '분기' in df.columns:
                df.set_index('분기', inplace=True)
                
                # 분기별 목표가 딕셔너리 생성
                # NaN 값을 제외하고, 백의 자리에서 반올림
                target_prices = {}
                for quarter, price in df['목표가'].items():
                    if pd.notna(price):  # NaN 값 제외
                        # 백의 자리에서 반올림
                        rounded_price = round(price / 100) * 100
                        target_prices[quarter] = int(rounded_price)
                    else:
                        target_prices[quarter] = np.nan
                
                return target_prices
            else:
                print(f"❌ 엑셀 파일에 '분기' 컬럼이 없습니다.")
                return None
        except FileNotFoundError:
            print(f"❌ 엑셀 파일을 찾을 수 없습니다: {excel_file}")
            return None
        except Exception as e:
            print(f"❌ 엑셀 파일 읽기 오류: {e}")
            return None
    
    @staticmethod
    def get_quarter(date):
        """
        날짜로부터 분기(YYYY.QN) 계산
        """
        year = date.year
        month = date.month
        quarter = (month - 1) // 3 + 1
        return f"{year}.{quarter}Q"
    
    @staticmethod
    def calculate_divergence_rate(current_price, target_price):
        """
        목표가 대비 괴리율 계산
        괴리율(%) = (현재가 - 목표가) / 목표가 * 100
        양수: 목표가보다 높음 (고평가), 음수: 목표가보다 낮음 (저평가)
        """
        if pd.isna(target_price) or target_price == 0:
            return np.nan
        return ((current_price - target_price) / target_price) * 100
    
    @staticmethod
    def add_quarter_and_target_price(df, excel_file, sheet_name):
        """
        DataFrame에 분기, 목표가, 괴리율 컬럼 추가
        """
        # 목표가 데이터 로드
        target_prices = FinancialDataManager.load_target_price(excel_file, sheet_name, '')
        
        if target_prices is None:
            print("⚠️  목표가 데이터를 로드할 수 없어 분기 컬럼만 추가합니다.")
            df['분기'] = df.index.map(FinancialDataManager.get_quarter)
            df['목표가'] = np.nan
            df['괴리율(%)'] = np.nan
            return df
        else:
            # 분기 컬럼 추가
            df['분기'] = df.index.map(FinancialDataManager.get_quarter)
            
            # 목표가 컬럼 추가 (백의 자리 반올림)
            df['목표가'] = df['분기'].map(target_prices).fillna(np.nan)
        
            # 괴리율 계산
            df['괴리율(%)'] = df.apply(
                lambda row: FinancialDataManager.calculate_divergence_rate(row['종가'], row['목표가']),
                axis=1
            )
        
        return df

class TechnicalIndicator:
    """기술적 지표 계산 클래스"""
    
    @staticmethod
    def ema(series, span=12):
        """지수이동평균(EMA) 계산"""
        return series.ewm(span=span, adjust=False).mean()
    
    @staticmethod
    def macd(df, price_col='종가', timeframe='daily', fast=None, slow=None, signal=None):
        """MACD 계산 - 시간봉별 최적화"""
        df = df.copy()
        
        # 시간봉별 기본 파라미터 설정
        timeframe_params = {
            'daily': {'fast': 12, 'slow': 26, 'signal': 9},
            'weekly': {'fast': 5, 'slow': 13, 'signal': 5},
            'monthly': {'fast': 3, 'slow': 7, 'signal': 3}
        }
        
        # timeframe을 문자열로 변환하여 안전하게 처리
        timeframe_str = str(timeframe).lower()
        default_params = timeframe_params.get(timeframe_str, timeframe_params['daily'])
        
        # 사용자 정의 파라미터 또는 기본값 사용
        if fast is None:
            fast = default_params['fast']
        if slow is None:
            slow = default_params['slow']
        if signal is None:
            signal = default_params['signal']
        
        ema_fast = TechnicalIndicator.ema(df[price_col], fast)
        ema_slow = TechnicalIndicator.ema(df[price_col], slow)
        
        df['MACD'] = ema_fast - ema_slow
        df['Signal'] = TechnicalIndicator.ema(df['MACD'], signal)
        df['Oscillator'] = df['MACD'] - df['Signal']
        
        return df
    
    @staticmethod
    def add_daily_macd(df):
        """일일 MACD 추가"""
        df = TechnicalIndicator.macd(df, timeframe='daily')
        df.rename(columns={'MACD': 'Daily_MACD', 'Signal': 'Daily_Signal', 
                          'Oscillator': 'Daily_Oscillator'}, inplace=True)
        return df
    
    @staticmethod
    def add_weekly_macd(df):
        """주간 MACD 추가 (현재 진행 중인 주까지 즉시 반영)"""
        df_temp = df[['종가']].copy()

        # 주간 데이터 생성
        df_weekly = df_temp.resample('W-FRI').last()
        df_weekly = TechnicalIndicator.macd(df_weekly, timeframe='weekly')
        df_weekly.rename(columns={'MACD': 'Weekly_MACD', 'Signal': 'Weekly_Signal',
                                 'Oscillator': 'Weekly_Oscillator'}, inplace=True)
        df_weekly = df_weekly[['Weekly_MACD', 'Weekly_Signal', 'Weekly_Oscillator']]

        # 핵심 수정:
        # 금요일 라벨을 단순 ffill 하지 말고, 같은 주(period)에 직접 매핑
        # -> 월~목에도 이번 주의 최신 주간 값이 반영됨
        df_weekly['year_week'] = df_weekly.index.to_period('W-FRI')

        df = df.copy()
        df['year_week'] = df.index.to_period('W-FRI')
        df = df.join(
            df_weekly.set_index('year_week')[['Weekly_MACD', 'Weekly_Signal', 'Weekly_Oscillator']],
            on='year_week'
        )
        df.drop(columns=['year_week'], inplace=True)

        return df
    
    @staticmethod
    def add_monthly_macd(df):
        """월간 MACD 추가 (현재 진행 중인 월까지 즉시 반영)"""
        df_temp = df[['종가']].copy()

        # 월간 데이터 생성
        df_monthly = df_temp.resample('ME').last()
        df_monthly = TechnicalIndicator.macd(df_monthly, timeframe='monthly')
        df_monthly.rename(columns={'MACD': 'Monthly_MACD', 'Signal': 'Monthly_Signal',
                                  'Oscillator': 'Monthly_Oscillator'}, inplace=True)
        df_monthly = df_monthly[['Monthly_MACD', 'Monthly_Signal', 'Monthly_Oscillator']]

        # 핵심 수정:
        # 월말 라벨을 단순 ffill 하지 말고, 같은 월(period)에 직접 매핑
        # -> 월중에도 이번 달의 최신 월간 값이 반영됨
        df_monthly['year_month'] = df_monthly.index.to_period('M')

        df = df.copy()
        df['year_month'] = df.index.to_period('M')
        df = df.join(
            df_monthly.set_index('year_month')[['Monthly_MACD', 'Monthly_Signal', 'Monthly_Oscillator']],
            on='year_month'
        )
        df.drop(columns=['year_month'], inplace=True)

        return df

class SignalGenerator:
    """매매 신호 생성 클래스"""
    
    @staticmethod
    def calculate_trend(osc_series):
        """Oscillator 추세 계산 (+1: 증가, -1: 감소, 0: 변화 없음)"""
        trend = pd.Series(0, index=osc_series.index)
        diff = osc_series.diff()
        trend[diff > 0] = 1
        trend[diff < 0] = -1
        return trend
    
    @staticmethod
    def maintain_weekly_trend(df):
        """Weekly_Trend를 해당 주 전체에 걸쳐 동일 값으로 유지"""
        df = df.copy()
        df['year_week'] = df.index.to_period('W-FRI')

        weekly_trend = (
            df.groupby('year_week')['Weekly_Trend']
              .apply(lambda s: s[s != 0].iloc[0] if (s != 0).any() else np.nan)
              .ffill()
              .fillna(0)
              .astype(int)
        )

        df['Weekly_Trend'] = df['year_week'].map(weekly_trend).astype(int)
        df.drop(columns=['year_week'], inplace=True)
        return df
    
    @staticmethod
    def maintain_monthly_trend(df):
        """Monthly_Trend를 해당 월 전체에 걸쳐 동일 값으로 유지"""
        df = df.copy()
        df['year_month'] = df.index.to_period('M')

        monthly_trend = (
            df.groupby('year_month')['Monthly_Trend']
              .apply(lambda s: s[s != 0].iloc[0] if (s != 0).any() else np.nan)
              .ffill()
              .fillna(0)
              .astype(int)
        )

        df['Monthly_Trend'] = df['year_month'].map(monthly_trend).astype(int)
        df.drop(columns=['year_month'], inplace=True)
        return df
    
    @staticmethod
    def identify_weekly_monthly_signals(df):
        """Weekly 추세 + Monthly 추세 전환 결합 신호

        - 매수(1): Weekly_Trend == +1 이면서 Monthly_Trend 가 양수로 전환된 날
                   (전일 Monthly_Trend <= 0  ->  당일 Monthly_Trend == +1)
        - 매도(1): Weekly_Trend == -1 이면서 Monthly_Trend 가 음수로 전환된 날
                   (전일 Monthly_Trend >= 0  ->  당일 Monthly_Trend == -1)

        Monthly_Trend 가 이미 같은 부호로 유지 중이면(전환일이 아니면) 신호 아님.
        """
        prev_monthly = df['Monthly_Trend'].shift(1)

        monthly_turn_up = (prev_monthly <= 0) & (df['Monthly_Trend'] == 1)
        monthly_turn_down = (prev_monthly >= 0) & (df['Monthly_Trend'] == -1)

        buy = (df['Weekly_Trend'] == 1) & monthly_turn_up
        sell = (df['Weekly_Trend'] == -1) & monthly_turn_down
        return buy, sell

    @staticmethod
    def identify_signal_changes(trend_series):
        """신호 전환점 감지 (엄격한 전환만 인정)

        - 매수(1): 추세가 -1 -> +1 로 전환된 날
        - 매도(1): 추세가 +1 -> -1 로 전환된 날
        - 0을 거치는 경우(0->1, 0->-1, 1->0, -1->0)는 신호로 보지 않음
        """
        prev_trend = trend_series.shift(1)
        sell = (prev_trend == 1) & (trend_series == -1)
        buy = (prev_trend == -1) & (trend_series == 1)
        return buy, sell

    @staticmethod
    def add_signals(df):
        """모든 신호 추가"""
        df['Daily_Trend'] = SignalGenerator.calculate_trend(df['Daily_Oscillator'])
        df['Weekly_Trend'] = SignalGenerator.calculate_trend(df['Weekly_Oscillator'])
        df['Monthly_Trend'] = SignalGenerator.calculate_trend(df['Monthly_Oscillator'])
        
        df = SignalGenerator.maintain_weekly_trend(df)
        df = SignalGenerator.maintain_monthly_trend(df)
        
        # Daily
        daily_buy, daily_sell = SignalGenerator.identify_signal_changes(df['Daily_Trend'])
        df['Daily_Buy'] = daily_buy.astype(int)
        df['Daily_Sell'] = daily_sell.astype(int)
        
        # Weekly
        weekly_buy, weekly_sell = SignalGenerator.identify_signal_changes(df['Weekly_Trend'])
        df['Weekly_Buy'] = weekly_buy.astype(int)
        df['Weekly_Sell'] = weekly_sell.astype(int)
        
        # Monthly
        monthly_buy, monthly_sell = SignalGenerator.identify_signal_changes(df['Monthly_Trend'])
        df['Monthly_Buy'] = monthly_buy.astype(int)
        df['Monthly_Sell'] = monthly_sell.astype(int)

        # Weekly + Monthly 결합 신호 (추가)
        wm_buy, wm_sell = SignalGenerator.identify_weekly_monthly_signals(df)
        df['Weekly_Monthly_Buy'] = wm_buy.astype(int)
        df['Weekly_Monthly_Sell'] = wm_sell.astype(int)

        return df

class Backtester:
    """백테스팅 클래스"""

    @staticmethod
    def backtest(df, signal_type='Weekly', monthly_filter=None, price_col='종가'):
        """기존 시그널 백테스트 수행"""
        df = df.copy()
        trades = []

        buy_col = f'{signal_type}_Buy'
        sell_col = f'{signal_type}_Sell'

        in_position = False
        entry_date, entry_price = None, None

        for idx in range(len(df)):
            current_date = df.index[idx]
            current_price = df[price_col].iloc[idx]

            if monthly_filter is not None:
                if df['Monthly_Trend'].iloc[idx] != monthly_filter:
                    if in_position:
                        profit_rate = (current_price - entry_price) / entry_price * 100
                        profit = current_price - entry_price
                        holding_days = (current_date - entry_date).days

                        trades.append({
                            '전략': f'{signal_type} (Monthly={monthly_filter})',
                            '매수일': entry_date, '매수가': round(entry_price, 0),
                            '매도일': current_date, '매도가': round(current_price, 0),
                            '수익률(%)': round(profit_rate, 2), '수익(원)': round(profit, 0),
                            '보유기간(일)': holding_days, '청산사유': 'Monthly_Trend 변화'
                        })
                        in_position = False
                    continue

            if df[buy_col].iloc[idx] == 1 and not in_position:
                entry_date, entry_price = current_date, current_price
                in_position = True

            elif df[sell_col].iloc[idx] == 1 and in_position:
                profit_rate = (current_price - entry_price) / entry_price * 100
                profit = current_price - entry_price
                holding_days = (current_date - entry_date).days

                trades.append({
                    '전략': f'{signal_type} (Monthly={monthly_filter})',
                    '매수일': entry_date, '매수가': round(entry_price, 0),
                    '매도일': current_date, '매도가': round(current_price, 0),
                    '수익률(%)': round(profit_rate, 2), '수익(원)': round(profit, 0),
                    '보유기간(일)': holding_days, '청산사유': '신호'
                })
                in_position = False

        return pd.DataFrame(trades) if trades else pd.DataFrame()

    # ── 등급(강력 매수 / 적극 매수 / 매수 / 적극 매도 / 매도) 기반 백테스트는 삭제됨 ──
    #    신호별 성과는 AnalysisPipeline.run() 의
    #    Backtester.backtest(...) + Backtester.calculate_metrics(...) 로 확인한다.

    @staticmethod
    def calculate_metrics(trades_df, strategy_name):
        """성과 지표 계산"""
        if len(trades_df) == 0:
            return {'전략': strategy_name, '총거래수': 0, '승리': 0, '패배': 0, '중립': 0,
                   '승률(%)': 0, '평균수익(%)': 0, '평균손실(%)': 0, '총수익률(%)': 0,
                   '누적수익(원)': 0, '최대수익(%)': 0, '최대손실(%)': 0, '손익비': 0,
                   '평균보유기간(일)': 0}

        win = trades_df[trades_df['수익률(%)'] > 0]
        loss = trades_df[trades_df['수익률(%)'] < 0]

        total_profit = trades_df['수익(원)'].sum()
        total_loss = loss['수익(원)'].sum()
        profit_factor = abs(total_profit / total_loss) if total_loss != 0 else 0

        return {
            '전략': strategy_name,
            '총거래수': len(trades_df),
            '승리': len(win),
            '패배': len(loss),
            '중립': len(trades_df) - len(win) - len(loss),
            '승률(%)': round(len(win) / len(trades_df) * 100, 1) if len(trades_df) > 0 else 0,
            '평균수익(%)': round(win['수익률(%)'].mean(), 1) if len(win) > 0 else 0,
            '평균손실(%)': round(loss['수익률(%)'].mean(), 1) if len(loss) > 0 else 0,
            '총수익률(%)': round(trades_df['수익률(%)'].sum(), 1),
            '누적수익(원)': round(total_profit, 0),
            '최대수익(%)': round(trades_df['수익률(%)'].max(), 1) if len(trades_df) > 0 else 0,
            '최대손실(%)': round(trades_df['수익률(%)'].min(), 1) if len(trades_df) > 0 else 0,
            '손익비': round(profit_factor, 1),
            '평균보유기간(일)': round(trades_df['보유기간(일)'].mean(), 1) if len(trades_df) > 0 else 0
        }

def classify_signal(row, buy_col="Weekly_Buy", sell_col="Weekly_Sell"):
    if row.get(buy_col, 0) == 1:
        return "매수"
    if row.get(sell_col, 0) == 1:
        return "매도"
    return "신호 없음"


def validate_excel(excel_file, stock_name):
    """원본 코드에서 사용하는 엑셀 구조를 확인합니다."""
    with pd.ExcelFile(excel_file) as book:
        if stock_name not in book.sheet_names:
            raise ValueError(
                f"엑셀에 '{stock_name}' 시트가 없습니다. "
                f"현재 시트: {', '.join(book.sheet_names)}"
            )
        financial = pd.read_excel(book, sheet_name=stock_name)

    required = {"분기", "목표가", "영업이익"}
    missing = required - set(financial.columns)
    if missing:
        raise ValueError(
            "엑셀에 필요한 컬럼이 없습니다: " + ", ".join(sorted(missing))
        )

    quarters = financial["분기"].astype(str).str.strip()
    if not quarters.str.fullmatch(r"\d{4}\.[1-4]Q").all():
        raise ValueError("분기는 '2026.1Q' 형식으로 입력하세요.")
    if quarters.duplicated().any():
        raise ValueError("엑셀에 중복된 분기가 있습니다.")

    for column in ("목표가", "영업이익"):
        numeric = pd.to_numeric(financial[column], errors="coerce")
        invalid = financial[column].notna() & numeric.isna()
        if invalid.any():
            raise ValueError(f"'{column}'에는 숫자 또는 빈칸만 입력하세요.")


def analyze_frame(frame, stock_name, excel_file=None):
    """원본 계산 로직을 이용해 분석합니다."""
    df = frame.copy()

    required = {"시가", "고가", "저가", "종가", "거래량"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError("가격 데이터 컬럼 누락: " + ", ".join(sorted(missing)))

    for column in required:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(subset=list(required))
    df = df.loc[df["종가"] > 0].copy()
    if df.empty:
        raise ValueError("유효한 가격 데이터가 없습니다.")

    df["날짜"] = df.index.strftime("%Y-%m-%d")
    messages = []

    if excel_file:
        validate_excel(excel_file, stock_name)
        df = FinancialDataManager.add_quarter_and_target_price(
            df, excel_file, stock_name
        )
    else:
        df["분기"] = df.index.map(FinancialDataManager.get_quarter)
        df["목표가"] = np.nan
        df["괴리율(%)"] = np.nan
        messages.append(
            "재무 엑셀이 없어 목표가·괴리율·영업이익은 표시하지 않습니다."
        )

    df = TechnicalIndicator.add_daily_macd(df)
    df = TechnicalIndicator.add_weekly_macd(df)
    df = TechnicalIndicator.add_monthly_macd(df)
    df = SignalGenerator.add_signals(df)

    strategies = [
        ("Weekly", "Weekly 단독", "Weekly"),
        ("Weekly+Monthly", "Weekly+Monthly 결합", "Weekly_Monthly"),
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
        metrics.append(Backtester.calculate_metrics(trade_frame, label))

    latest = df.iloc[-1]
    latest_status = {
        "종목명": stock_name,
        "기준일": df.index[-1].strftime("%Y-%m-%d"),
        "종가": float(latest["종가"]),
        "Weekly 신호": classify_signal(latest),
        "Weekly+Monthly 신호": classify_signal(
            latest, "Weekly_Monthly_Buy", "Weekly_Monthly_Sell"
        ),
    }

    return {
        "data": df,
        "metrics": pd.DataFrame(metrics),
        "trades": trades,
        "latest": latest_status,
        "messages": messages,
    }


def recent_signals(df, combined=False, limit=10):
    prefix = "Weekly_Monthly" if combined else "Weekly"
    buy_col, sell_col = f"{prefix}_Buy", f"{prefix}_Sell"

    mask = (df[buy_col] == 1) | (df[sell_col] == 1)
    result = df.loc[mask].tail(limit).copy()

    if result.empty:
        return pd.DataFrame(columns=["날짜", "종가", "신호"])

    result["신호"] = result.apply(
        lambda row: classify_signal(row, buy_col, sell_col), axis=1
    )
    columns = [
        "날짜", "종가", "목표가", "괴리율(%)", "신호",
        "Daily_Trend", "Weekly_Trend", "Monthly_Trend",
    ]
    return result[[c for c in columns if c in result.columns]].reset_index(drop=True)
