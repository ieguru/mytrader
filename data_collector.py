# data_collector.py
"""
가격, 투자자 순매수 수량, 공매도 거래 수량을 수집합니다.

패널 5용 컬럼
    외국인합계_순매수
    기관합계_순매수
    개인_순매수

패널 6용 컬럼
    공매도수량

수량 단위는 모두 '주'입니다.

주의:
- import만으로 데이터를 수집하지 않습니다.
- 조회 실패 및 누락값을 0으로 보정하지 않습니다.
- 장중 시세를 별도로 합성하지 않습니다.
- KRX 인증이 필요한 환경에서는 이 모듈을 import하기 전에
  KRX_ID, KRX_PW 환경변수를 설정해야 합니다.
"""

import logging
import time

import numpy as np
import pandas as pd
from pykrx import stock


logger = logging.getLogger(__name__)

# 필요한 경우 외부에서 종목명 → 코드 정보를 등록할 수 있습니다.
STOCK_DICT = {}

PRICE_COLUMNS = [
    "시가",
    "고가",
    "저가",
    "종가",
    "거래량",
]

INVESTOR_COLUMNS = [
    "외국인합계",
    "기관합계",
    "개인",
]


# =========================================================
# 공통 처리
# =========================================================

def _to_numeric(series):
    """숫자와 쉼표가 포함된 숫자 문자열을 숫자형으로 변환합니다."""
    if not pd.api.types.is_numeric_dtype(series):
        series = (
            series.astype("string")
            .str.replace(",", "", regex=False)
            .str.strip()
        )

    return pd.to_numeric(
        series,
        errors="coerce",
    ).replace(
        [np.inf, -np.inf],
        np.nan,
    )


def _normalize_frame(frame):
    """날짜 인덱스, 정렬, 중복 날짜를 정리합니다."""
    if frame is None or not isinstance(frame, pd.DataFrame):
        return pd.DataFrame()

    if frame.empty:
        return pd.DataFrame()

    result = frame.copy()

    # 정수형 인덱스가 나노초 날짜로 오인되는 것을 방지합니다.
    if not isinstance(result.index, pd.DatetimeIndex):
        if pd.api.types.is_numeric_dtype(result.index.dtype):
            raise ValueError(
                "조회 결과의 날짜 인덱스가 올바르지 않습니다."
            )

    result.index = pd.to_datetime(
        result.index,
        errors="coerce",
    )

    result = result.loc[
        ~result.index.isna()
    ].copy()

    if result.empty:
        return pd.DataFrame()

    if result.index.tz is not None:
        result.index = (
            result.index
            .tz_convert("Asia/Seoul")
            .tz_localize(None)
        )

    result.index = result.index.normalize()
    result.index.name = "날짜"

    result.columns = [
        str(column).strip()
        for column in result.columns
    ]

    if result.columns.duplicated().any():
        raise ValueError(
            "조회 결과에 중복된 컬럼이 있습니다."
        )

    result = result.sort_index()

    result = result.loc[
        ~result.index.duplicated(keep="last")
    ].copy()

    return result


def _validate_request(ticker, start_date, end_date):
    """종목코드와 조회 기간을 검증하고 표준 형식으로 반환합니다."""
    ticker = str(ticker).strip()

    if (
        not ticker.isascii()
        or not ticker.isdigit()
        or len(ticker) > 6
    ):
        raise ValueError(
            "종목코드는 6자리 숫자 형식이어야 합니다."
        )

    ticker = ticker.zfill(6)

    try:
        start = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date)

        if pd.isna(start) or pd.isna(end):
            raise ValueError

        if start.tzinfo is not None:
            start = (
                start
                .tz_convert("Asia/Seoul")
                .tz_localize(None)
            )

        if end.tzinfo is not None:
            end = (
                end
                .tz_convert("Asia/Seoul")
                .tz_localize(None)
            )

        start = start.normalize()
        end = end.normalize()

    except (TypeError, ValueError, OverflowError):
        raise ValueError(
            "조회 날짜가 올바르지 않습니다. "
            "'YYYYMMDD' 또는 'YYYY-MM-DD' 형식을 사용하세요."
        ) from None

    if start > end:
        raise ValueError(
            "시작일은 종료일보다 늦을 수 없습니다."
        )

    return (
        ticker,
        start.strftime("%Y%m%d"),
        end.strftime("%Y%m%d"),
    )


def _filter_date_range(frame, start_date, end_date):
    """응답 데이터를 요청 기간 안으로 제한합니다."""
    if frame is None or frame.empty:
        return pd.DataFrame()

    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()

    return frame.loc[
        (frame.index >= start)
        & (frame.index <= end)
    ].copy()


# =========================================================
# 데이터 수집
# =========================================================

class DataCollector:
    """pykrx 기반 데이터 수집 클래스"""

    @staticmethod
    def get_stock_code(stock_name):
        """등록된 종목명에서 종목코드를 반환합니다."""
        return STOCK_DICT.get(stock_name)

    @staticmethod
    def _safe_to_int(value):
        """숫자 또는 숫자 문자열을 정수로 변환합니다."""
        try:
            if value is None or pd.isna(value):
                return None

            cleaned = (
                str(value)
                .replace(",", "")
                .replace("+", "")
                .strip()
            )

            if not cleaned:
                return None

            number = float(cleaned)

            if not np.isfinite(number):
                return None

            return int(number)

        except (TypeError, ValueError, OverflowError):
            return None

    @staticmethod
    def _call_pykrx_safely(
        fetch_func,
        *args,
        retries=3,
        base_sleep=1.2,
        **kwargs,
    ):
        """
        빈 응답과 일시적 오류를 재시도합니다.

        실패 시 빈 DataFrame을 반환합니다.
        인증정보가 포함될 수 있는 인자나 원본 오류 내용은
        로그에 출력하지 않습니다.
        """
        retries = max(1, int(retries))
        last_error_type = None

        for attempt in range(1, retries + 1):
            try:
                result = fetch_func(
                    *args,
                    **kwargs,
                )

                if (
                    isinstance(result, pd.DataFrame)
                    and not result.empty
                ):
                    return result.copy()

            except Exception as error:
                last_error_type = type(error).__name__

            if attempt < retries:
                time.sleep(
                    base_sleep * attempt
                )

        function_name = getattr(
            fetch_func,
            "__name__",
            "pykrx",
        )

        if last_error_type:
            logger.warning(
                "%s 조회 실패: %s",
                function_name,
                last_error_type,
            )
        else:
            logger.warning(
                "%s 조회 결과가 비어 있습니다.",
                function_name,
            )

        return pd.DataFrame()

    @staticmethod
    def _normalize_investor_frame(frame, suffix):
        """투자자 데이터 컬럼에 접미사를 추가합니다."""
        result = _normalize_frame(frame)

        if result.empty:
            return result

        result.columns = [
            f"{column}_{suffix}"
            for column in result.columns
        ]

        return result

    # -----------------------------------------------------
    # 가격
    # -----------------------------------------------------

    @staticmethod
    def fetch_ohlcv(
        ticker,
        start_date,
        end_date,
        include_today_live=False,
    ):
        """
        일별 OHLCV를 조회합니다.

        별도의 장중 시세를 합성하지 않습니다.
        include_today_live는 기존 호출과의 호환성을 위한 인자입니다.
        """
        if include_today_live:
            logger.warning(
                "이 모듈은 별도의 장중 시세 합성을 지원하지 않습니다. "
                "pykrx가 반환한 일별 데이터만 사용합니다."
            )

        ticker, start_date, end_date = _validate_request(
            ticker,
            start_date,
            end_date,
        )

        raw = DataCollector._call_pykrx_safely(
            stock.get_market_ohlcv,
            start_date,
            end_date,
            ticker,
        )

        frame = _normalize_frame(raw)

        if frame.empty:
            return None

        frame = _filter_date_range(
            frame,
            start_date,
            end_date,
        )

        missing = set(PRICE_COLUMNS) - set(frame.columns)

        if missing:
            raise ValueError(
                "가격 데이터 컬럼 누락: "
                + ", ".join(sorted(missing))
            )

        for column in PRICE_COLUMNS:
            frame[column] = _to_numeric(
                frame[column]
            )

        if "등락률" in frame.columns:
            frame["등락률"] = _to_numeric(
                frame["등락률"]
            )
        else:
            frame["등락률"] = np.nan

        # 가격 결측 및 유효하지 않은 종가 행은 제외합니다.
        frame = frame.dropna(
            subset=PRICE_COLUMNS
        )

        frame = frame.loc[
            frame["종가"] > 0
        ].copy()

        if frame.empty:
            return None

        return frame

    # -----------------------------------------------------
    # 투자자 순매수 수량
    # -----------------------------------------------------

    @staticmethod
    def fetch_investor_trading(
        ticker,
        start_date,
        end_date,
    ):
        """
        외국인·기관·개인의 일별 순매수 수량을 조회합니다.

        반환 컬럼:
            외국인합계_순매수
            기관합계_순매수
            개인_순매수

        단위: 주
        """
        ticker, start_date, end_date = _validate_request(
            ticker,
            start_date,
            end_date,
        )

        raw = DataCollector._call_pykrx_safely(
            stock.get_market_trading_volume_by_date,
            start_date,
            end_date,
            ticker,
            on="순매수",
        )

        frame = _normalize_frame(raw)

        if frame.empty:
            return None

        frame = _filter_date_range(
            frame,
            start_date,
            end_date,
        )

        if frame.empty:
            return None

        result = pd.DataFrame(index=frame.index)

        for investor in INVESTOR_COLUMNS:
            target = f"{investor}_순매수"

            if investor in frame.columns:
                result[target] = _to_numeric(
                    frame[investor]
                )
            else:
                # 컬럼이 누락된 경우 0이 아니라 NaN을 유지합니다.
                result[target] = np.nan

        if result.isna().all().all():
            return None

        return result

    # -----------------------------------------------------
    # 공매도 거래 수량
    # -----------------------------------------------------

    @staticmethod
    def fetch_shorting_volume(
        ticker,
        start_date,
        end_date,
    ):
        """
        일별 공매도 거래 수량을 조회합니다.

        공매도 잔고가 아닌 해당 거래일의 거래량입니다.

        반환 컬럼:
            공매도수량

        단위: 주
        """
        ticker, start_date, end_date = _validate_request(
            ticker,
            start_date,
            end_date,
        )

        fetch_function = getattr(
            stock,
            "get_shorting_volume_by_date",
            None,
        )

        if fetch_function is None:
            raise ValueError(
                "설치된 pykrx에서 공매도 거래량 조회 함수를 "
                "찾을 수 없습니다."
            )

        raw = DataCollector._call_pykrx_safely(
            fetch_function,
            start_date,
            end_date,
            ticker,
        )

        frame = _normalize_frame(raw)

        if frame.empty:
            return None

        frame = _filter_date_range(
            frame,
            start_date,
            end_date,
        )

        if frame.empty:
            return None

        source_column = None

        for candidate in (
            "공매도",
            "공매도거래량",
            "공매도 거래량",
        ):
            if candidate in frame.columns:
                source_column = candidate
                break

        if source_column is None:
            raise ValueError(
                "공매도 수량 컬럼을 찾지 못했습니다. "
                "반환 컬럼: "
                + ", ".join(map(str, frame.columns))
            )

        result = pd.DataFrame(index=frame.index)

        result["공매도수량"] = _to_numeric(
            frame[source_column]
        )

        # 음수 거래량은 정상 값으로 사용하지 않습니다.
        result.loc[
            result["공매도수량"] < 0,
            "공매도수량",
        ] = np.nan

        if result["공매도수량"].isna().all():
            return None

        return result

    # -----------------------------------------------------
    # 병합
    # -----------------------------------------------------

    @staticmethod
    def merge_data(df_ohlcv, df_investor=None):
        """가격 데이터에 투자자 순매수 수량을 병합합니다."""
        prices = _normalize_frame(df_ohlcv)

        if prices.empty:
            return None

        if df_investor is not None and not df_investor.empty:
            investors = _normalize_frame(df_investor)

            if not investors.empty:
                frame = prices.join(
                    investors,
                    how="left",
                )
            else:
                frame = prices.copy()
        else:
            frame = prices.copy()

        frame["날짜"] = frame.index.strftime("%Y-%m-%d")
        frame["년"] = frame.index.year
        frame["월"] = frame.index.month
        frame["일"] = frame.index.day
        frame["년월"] = frame.index.strftime("%Y-%m")
        frame["년주"] = frame.index.strftime("%Y-W%U")

        return frame.sort_index()


# =========================================================
# app.py / batch_daily.py에서 사용하는 함수
# =========================================================

def fetch_data(
    ticker,
    start_date,
    end_date,
    include_investors=True,
):
    """
    가격, 선택적 투자자 순매수 수량, 공매도 수량을 반환합니다.

    Returns:
        frame: pandas.DataFrame
        messages: 경고 및 안내 문자열 리스트

    투자자 체크박스:
        외국인·기관·개인 순매수 수량 조회에 적용됩니다.

    공매도:
        체크박스와 관계없이 조회합니다.
    """
    ticker, start_date, end_date = _validate_request(
        ticker,
        start_date,
        end_date,
    )

    messages = []

    # -----------------------------------------------------
    # 1. 가격 데이터
    # -----------------------------------------------------

    try:
        ohlcv = DataCollector.fetch_ohlcv(
            ticker,
            start_date,
            end_date,
            include_today_live=False,
        )

    except Exception:
        raise ValueError(
            "가격 데이터를 조회하지 못했습니다. "
            "조회 기간, pykrx 버전 및 KRX 인증 상태를 확인하세요."
        ) from None

    if ohlcv is None or ohlcv.empty:
        raise ValueError(
            "가격 데이터를 가져오지 못했습니다. "
            "종목코드, 기간, 인터넷 연결 또는 "
            "pykrx/KRX 인증 상태를 확인하세요."
        )

    # -----------------------------------------------------
    # 2. 외국인·기관·개인 순매수 수량
    # -----------------------------------------------------

    investors = None

    if include_investors:
        try:
            investors = DataCollector.fetch_investor_trading(
                ticker,
                start_date,
                end_date,
            )

            if investors is None or investors.empty:
                investors = None

                messages.append(
                    "외국인·기관·개인 순매수 수량을 "
                    "가져오지 못했습니다. "
                    "KRX 인증 및 데이터 조회 상태를 확인하세요."
                )

        except Exception:
            investors = None

            messages.append(
                "투자자 순매수 수량 조회 중 오류가 발생했습니다. "
                "가격 분석은 계속 진행합니다."
            )

    # -----------------------------------------------------
    # 3. 가격과 투자자 데이터 병합
    # -----------------------------------------------------

    frame = DataCollector.merge_data(
        ohlcv,
        investors,
    )

    if frame is None or frame.empty:
        raise ValueError("분석할 데이터가 없습니다.")

    if include_investors and investors is not None:
        for investor in INVESTOR_COLUMNS:
            column = f"{investor}_순매수"

            if column not in frame.columns:
                continue

            missing_count = int(
                frame[column].isna().sum()
            )

            if missing_count:
                messages.append(
                    f"{investor} 순매수 수량이 "
                    f"{missing_count}개 날짜에서 누락되었습니다. "
                    "누락값은 0으로 보정하지 않으므로 "
                    "누적 수량 차트가 끊길 수 있습니다."
                )

    # -----------------------------------------------------
    # 4. 공매도 거래 수량
    # -----------------------------------------------------

    try:
        shorting = DataCollector.fetch_shorting_volume(
            ticker,
            start_date,
            end_date,
        )

        if shorting is not None and not shorting.empty:
            frame = frame.join(
                shorting,
                how="left",
            )

            missing_count = int(
                frame["공매도수량"].isna().sum()
            )

            if missing_count:
                messages.append(
                    "공매도 거래 수량이 "
                    f"{missing_count}개 날짜에서 누락되었습니다. "
                    "미조회 데이터와 실제 0주를 구분하여 표시합니다."
                )

        else:
            messages.append(
                "공매도 거래 수량을 가져오지 못했습니다. "
                "KRX 인증, pykrx 버전 또는 "
                "해당 기간의 데이터 제공 여부를 확인하세요."
            )

    except ValueError as error:
        # 이 모듈에서 만든 컬럼·지원 여부 안내입니다.
        messages.append(
            f"공매도 조회 안내: {error}"
        )

    except Exception:
        messages.append(
            "공매도 거래 수량 조회 중 오류가 발생했습니다. "
            "가격 및 투자자 수급 분석은 계속 진행합니다."
        )

    # -----------------------------------------------------
    # 5. 최종 정리
    # -----------------------------------------------------

    frame = frame.sort_index()

    if frame.index.has_duplicates:
        frame = frame.loc[
            ~frame.index.duplicated(keep="last")
        ].copy()

        messages.append("중복 날짜를 제거했습니다.")

    return frame, messages
