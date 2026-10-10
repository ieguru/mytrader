"""원본 DataCollector 재사용. import 시 데이터 수집을 실행하지 않습니다."""
import io
import re
import time
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests
from pykrx import stock

# 원본 클래스의 이름 조회 메서드를 위한 변수입니다.
# MVP는 웹에서 6자리 종목코드를 직접 전달합니다.
STOCK_DICT = {}

class DataCollector:
    """KRX 데이터 수집 클래스"""

    @staticmethod
    def get_stock_code(stock_name):
        """종목명 → 종목코드 변환"""
        return STOCK_DICT.get(stock_name, None)

    @staticmethod
    def _safe_to_int(value):
        """문자/숫자 값을 int로 안전 변환"""
        try:
            if value is None:
                return None
            if isinstance(value, (int, float)):
                return int(value)
            value = str(value).replace(',', '').replace('+', '').strip()
            if value == '':
                return None
            return int(float(value))
        except:
            return None

    @staticmethod
    def _call_pykrx_safely(fetch_func, *args, retries=3, base_sleep=1.2, **kwargs):
        """pykrx 호출 시 빈 응답/JSONDecode 오류를 숨기고 재시도"""
        last_exception = None

        for attempt in range(1, retries + 1):
            captured = io.StringIO()
            try:
                with redirect_stdout(captured), redirect_stderr(captured):
                    df = fetch_func(*args, **kwargs)

                if isinstance(df, pd.DataFrame) and not df.empty:
                    return df

            except Exception as e:
                last_exception = e

            if attempt < retries:
                time.sleep(base_sleep * attempt)

        if last_exception:
            print(f"⚠️ pykrx 투자자 수급 조회 실패: {last_exception}")

        return pd.DataFrame()

    @staticmethod
    def _normalize_investor_frame(df, suffix):
        """투자자 수급 DataFrame 후처리"""
        if df is None or df.empty:
            return pd.DataFrame()

        df = df.copy()
        df.index = pd.to_datetime(df.index)
        df.columns = [f'{col}_{suffix}' for col in df.columns]
        return df

    @staticmethod
    def fetch_today_live_naver(ticker):
        """
        네이버 금융에서 오늘 장중 시세 조회
        - 시가
        - 현재가(장중 종가로 사용)
        - 고가
        - 저가
        - 거래량

        주의:
        - 네이버 금융은 비공식 소스라 응답 구조가 바뀔 수 있음
        - 실패 시 None 반환
        """
        try:
            # 네이버 금융 비공식 JSON 엔드포인트로 많이 쓰이는 형태
            url = f"https://finance.naver.com/item/main.naver?code={ticker}"
            headers = {
                "User-Agent": "Mozilla/5.0",
                "Referer": "https://finance.naver.com/"
            }

            resp = requests.get(url, headers=headers, timeout=10)
            resp.raise_for_status()
            html = resp.text

            # HTML 내부에서 현재가/시가/고가/저가/거래량 파싱
            # 네이버 구조 변경 가능성 있으므로 pandas.read_html 방식 보조 사용
            tables = pd.read_html(html)

            current_price = None
            open_price = None
            high_price = None
            low_price = None
            volume = None

            # 테이블들 순회하면서 필요한 필드 탐색
            for table in tables:
                flat_values = table.astype(str).values.flatten().tolist()
                joined = " ".join(flat_values)

                # 현재가 정보 테이블 추정
                if "시가" in joined and "고가" in joined and "저가" in joined:
                    values = flat_values

                    for i, v in enumerate(values):
                        if v == "시가" and i + 1 < len(values):
                            open_price = DataCollector._safe_to_int(values[i + 1])

                        elif v == "고가" and i + 1 < len(values):
                            high_price = DataCollector._safe_to_int(values[i + 1])

                        elif v == "저가" and i + 1 < len(values):
                            low_price = DataCollector._safe_to_int(values[i + 1])

                        elif v == "거래량" and i + 1 < len(values):
                            volume = DataCollector._safe_to_int(values[i + 1])

                # 현재가는 HTML 본문에서 추가 탐색
            import re

            # 현재가 추정 패턴
            price_patterns = [
                r'현재가\s*([0-9,]+)',
                r'"nv">([0-9,]+)</p>',
                r'blind">현재가</span>\s*<strong[^>]*>([0-9,]+)</strong>'
            ]

            for pattern in price_patterns:
                match = re.search(pattern, html)
                if match:
                    current_price = DataCollector._safe_to_int(match.group(1))
                    if current_price is not None:
                        break

            # 테이블에서 현재가 못 찾으면 시가/고가/저가와 함께 추정
            if current_price is None:
                for table in tables:
                    for val in table.astype(str).values.flatten():
                        num = DataCollector._safe_to_int(val)
                        if num is not None and num > 0:
                            # 첫 번째 큰 숫자를 현재가 후보로 보는 단순 fallback
                            current_price = num
                            break
                    if current_price is not None:
                        break

            if all(v is None for v in [current_price, open_price, high_price, low_price, volume]):
                print("❌ 네이버 금융 오늘 시세 파싱 실패")
                return None

            today = pd.Timestamp(datetime.now().date())

            row = pd.Series({
                "시가": open_price,
                "고가": high_price,
                "저가": low_price,
                "종가": current_price,   # 장중에는 현재가를 종가처럼 사용
                "거래량": volume
            }, name=today)

            return row

        except Exception as e:
            print(f"❌ 네이버 금융 오늘 시세 조회 오류: {e}")
            return None

    @staticmethod
    def fetch_today_open_close(ticker):
        """오늘 시가/장중 종가(현재가)만 간단 조회"""
        row = DataCollector.fetch_today_live_naver(ticker)
        if row is None:
            return None

        return {
            "날짜": row.name.strftime('%Y-%m-%d'),
            "시가": row.get("시가"),
            "종가": row.get("종가"),   # 장중 현재가
            "고가": row.get("고가"),
            "저가": row.get("저가"),
            "거래량": row.get("거래량")
        }

    @staticmethod
    def fetch_ohlcv(ticker, start_date, end_date, include_today_live=False):
        """
        OHLCV 데이터 추출
    
        수정 이유:
        장중 현재가를 오늘 종가로 반영할 때,
        시가/고가/저가가 비어 있으면 오늘 캔들이 차트에 안 보일 수 있으므로
        반드시 OHLC를 완성해서 오늘 일봉이 그려지도록 보정함.
        """
        try:
            df = stock.get_market_ohlcv(start_date, end_date, ticker)
            if df is None or df.empty:
                print("❌ OHLCV 데이터 없음")
                return None
    
            df.index = pd.to_datetime(df.index)
    
            # 숫자형 강제 변환
            for col in ["시가", "고가", "저가", "종가", "거래량"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
    
            if "등락률" not in df.columns:
                df["등락률"] = np.nan
    
            if include_today_live:
                today_live = DataCollector.fetch_today_live_naver(ticker)
    
                if today_live is not None:
                    today_idx = pd.Timestamp(datetime.now().date()).normalize()
    
                    for col in ["시가", "고가", "저가", "종가", "거래량", "등락률"]:
                        if col not in df.columns:
                            df[col] = np.nan
    
                    # 전일 종가 확보
                    prev_close = None
                    prev_rows = df.loc[df.index < today_idx]
                    if len(prev_rows) > 0 and pd.notna(prev_rows["종가"].iloc[-1]):
                        prev_close = float(prev_rows["종가"].iloc[-1])
    
                    live_close = pd.to_numeric(today_live.get("종가"), errors="coerce")
                    live_open = pd.to_numeric(today_live.get("시가"), errors="coerce")
                    live_high = pd.to_numeric(today_live.get("고가"), errors="coerce")
                    live_low = pd.to_numeric(today_live.get("저가"), errors="coerce")
                    live_volume = pd.to_numeric(today_live.get("거래량"), errors="coerce")
                    live_change = pd.to_numeric(today_live.get("등락률"), errors="coerce")
    
                    # ------------------------------------------
                    # 수정 핵심:
                    # 시가/고가/저가가 비어도 오늘 캔들이 반드시 생기도록 보정
                    # ------------------------------------------
                    if pd.isna(live_close):
                        print("⚠️ 오늘 현재가를 가져오지 못해 장중 일봉 반영을 건너뜁니다.")
                        return df.sort_index()
    
                    if pd.isna(live_open):
                        # 시가가 없으면 전일 종가를 우선 사용, 그것도 없으면 현재가 사용
                        live_open = prev_close if prev_close is not None else live_close
    
                    candidate_high = [x for x in [live_high, live_open, live_close, prev_close] if pd.notna(x)]
                    candidate_low  = [x for x in [live_low,  live_open, live_close, prev_close] if pd.notna(x)]
    
                    if pd.isna(live_high):
                        live_high = max(candidate_high) if len(candidate_high) > 0 else live_close
    
                    if pd.isna(live_low):
                        live_low = min(candidate_low) if len(candidate_low) > 0 else live_close
    
                    if pd.isna(live_volume):
                        live_volume = 0
    
                    # 오늘 행 반영
                    df.loc[today_idx, "시가"] = float(live_open)
                    df.loc[today_idx, "고가"] = float(live_high)
                    df.loc[today_idx, "저가"] = float(live_low)
                    df.loc[today_idx, "종가"] = float(live_close)   # 현재가를 종가로 반영
                    df.loc[today_idx, "거래량"] = float(live_volume)
    
                    # 등락률 반영
                    if pd.notna(live_change):
                        df.loc[today_idx, "등락률"] = float(live_change)
                    elif prev_close is not None and prev_close != 0:
                        df.loc[today_idx, "등락률"] = ((float(live_close) - prev_close) / prev_close) * 100
    
                    # 숫자형 재정리
                    for col in ["시가", "고가", "저가", "종가", "거래량", "등락률"]:
                        df[col] = pd.to_numeric(df[col], errors="coerce")
    
                    #print(
                    #    f"✓ 장중 일봉 반영 완료: {today_idx.strftime('%Y-%m-%d')} "
                    #    f"/ 시가 {int(live_open):,} / 고가 {int(live_high):,} "
                    #    f"/ 저가 {int(live_low):,} / 종가 {int(live_close):,}"
                    #)
    
            return df.sort_index()
    
        except Exception as e:
            print(f"❌ OHLCV 추출 오류: {e}")
            return None


    @staticmethod
    def fetch_investor_trading(ticker, start_date, end_date):
        """거래주체별 매매량 추출 (pykrx 빈 응답/JSON 오류 내성 버전)"""
        try:
            query_args = (start_date, end_date, ticker)

            df_net = DataCollector._call_pykrx_safely(
                stock.get_market_trading_volume_by_date, *query_args, on='순매수'
            )
            df_buy = DataCollector._call_pykrx_safely(
                stock.get_market_trading_volume_by_date, *query_args, on='매수'
            )
            df_sell = DataCollector._call_pykrx_safely(
                stock.get_market_trading_volume_by_date, *query_args, on='매도'
            )

            frames = [
                DataCollector._normalize_investor_frame(df_net, '순매수'),
                DataCollector._normalize_investor_frame(df_buy, '매수'),
                DataCollector._normalize_investor_frame(df_sell, '매도'),
            ]
            frames = [df for df in frames if not df.empty]

            if not frames:
                print(f"\n⚠️ {ticker} 투자자 매매량 데이터 크롤링 실패")
                return None

            return pd.concat(frames, axis=1).sort_index()

        except Exception as e:
            print(f"❌ 거래주체별 매매량 추출 오류: {e}")
            return None

    @staticmethod
    def merge_data(df_ohlcv, df_investor):
        """OHLCV와 거래주체별 매매량 병합"""
        if df_ohlcv is None or df_ohlcv.empty:
            return None

        if df_investor is not None:
            df = df_ohlcv.join(df_investor, how='left')
        else:
            df = df_ohlcv.copy()

        # 기본 날짜 컬럼
        df['날짜'] = df.index.strftime('%Y-%m-%d')
        df['년'] = df.index.year
        df['월'] = df.index.month
        df['일'] = df.index.day
        df['년월'] = df.index.strftime('%Y-%m')
        df['년주'] = df.index.strftime('%Y-W%U')

        return df

def fetch_data(ticker, start_date, end_date, include_investors=True):
    """과거 일봉과 선택적 투자자 데이터를 수집합니다."""
    ticker = str(ticker).strip().zfill(6)

    ohlcv = DataCollector.fetch_ohlcv(
        ticker,
        start_date,
        end_date,
        include_today_live=False,
    )
    if ohlcv is None or ohlcv.empty:
        raise ValueError(
            "가격 데이터를 가져오지 못했습니다. "
            "종목코드, 기간, 인터넷 연결 또는 pykrx/KRX 상태를 확인하세요."
        )

    messages = []
    investors = None   

    frame = DataCollector.merge_data(ohlcv, investors)
    if frame is None or frame.empty:
        raise ValueError("분석할 데이터가 없습니다.")

    frame = frame.sort_index()
    if frame.index.has_duplicates:
        frame = frame.loc[~frame.index.duplicated(keep="last")].copy()
        messages.append("중복 날짜를 제거했습니다.")

    return frame, messages
