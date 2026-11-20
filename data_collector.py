# data_collector.py
"""
가격과 투자자 순매수 수량만 수집합니다.

수집 항목:
    시가, 고가, 저가, 종가, 거래량
    외국인합계_순매수
    기관합계_순매수
    개인_순매수

공매도 및 대차거래 데이터는 조회하지 않습니다.

주의:
- import만으로 데이터를 수집하지 않습니다.
- 조회 실패 및 누락값을 0으로 보정하지 않습니다.
- 장중 시세를 별도로 합성하지 않습니다.
- 인증정보를 이 파일에 넣지 않습니다.
"""

import logging
import time

import numpy as np
import pandas as pd
from pykrx import stock


logger = logging.getLogger(__name__)

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


def _normalize_frame(frame):
    if (
        frame is None
        or not isinstance(frame, pd.DataFrame)
        or frame.empty
    ):
        return pd.DataFrame()

    result = frame.copy()

    if not isinstance(
        result.index,
        pd.DatetimeIndex,
    ):
        if pd.api.types.is_numeric_dtype(
            result.index.dtype
        ):
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

    result.index = (
        result.index.normalize()
    )

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


def _validate_request(
    ticker,
    start_date,
    end_date,
):
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

    if isinstance(
        start_date,
        (int, np.integer),
    ):
        start_date = str(start_date)

    if isinstance(
        end_date,
        (int, np.integer),
    ):
        end_date = str(end_date)

    try:
        start = pd.Timestamp(
            start_date
        )

        end = pd.Timestamp(
            end_date
        )

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

    except (
        TypeError,
        ValueError,
        OverflowError,
    ):
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


def _filter_date_range(
    frame,
    start_date,
    end_date,
):
    if frame is None or frame.empty:
        return pd.DataFrame()

    start = pd.Timestamp(
        start_date
    ).normalize()

    end = pd.Timestamp(
        end_date
    ).normalize()

    return frame.loc[
        (frame.index >= start)
        & (frame.index <= end)
    ].copy()


# =========================================================
# 데이터 수집
# =========================================================

class DataCollector:

    @staticmethod
    def get_stock_code(stock_name):
        return STOCK_DICT.get(
            stock_name
        )

    @staticmethod
    def _safe_to_int(value):
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

            number = float(
                cleaned
            )

            if not np.isfinite(number):
                return None

            return int(number)

        except (
            TypeError,
            ValueError,
            OverflowError,
        ):
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
        빈 응답과 일시적인 오류를 재시도합니다.

        인증정보가 포함될 수 있는 인자와
        원본 오류 내용은 로그에 출력하지 않습니다.
        """
        retries = max(
            1,
            int(retries),
        )

        last_error_type = None

        for attempt in range(
            1,
            retries + 1,
        ):
            try:
                result = fetch_func(
                    *args,
                    **kwargs,
                )

                if (
                    isinstance(
                        result,
                        pd.DataFrame,
                    )
                    and not result.empty
                ):
                    return result.copy()

            except Exception as error:
                last_error_type = (
                    type(error).__name__
                )

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
    def _normalize_investor_frame(
        frame,
        suffix,
    ):
        result = _normalize_frame(
            frame
        )

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
        if include_today_live:
            logger.warning(
                "별도 장중 시세 합성은 지원하지 않습니다. "
                "pykrx의 일별 데이터만 사용합니다."
            )

        ticker, start_date, end_date = (
            _validate_request(
                ticker,
                start_date,
                end_date,
            )
        )

        raw = (
            DataCollector._call_pykrx_safely(
                stock.get_market_ohlcv,
                start_date,
                end_date,
                ticker,
            )
        )

        frame = _normalize_frame(
            raw
        )

        if frame.empty:
            return None

        frame = _filter_date_range(
            frame,
            start_date,
            end_date,
        )

        if frame.empty:
            return None

        missing = (
            set(PRICE_COLUMNS)
            - set(frame.columns)
        )

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
        ticker, start_date, end_date = (
            _validate_request(
                ticker,
                start_date,
                end_date,
            )
        )

        raw = (
            DataCollector._call_pykrx_safely(
                stock.get_market_trading_volume_by_date,
                start_date,
                end_date,
                ticker,
                on="순매수",
            )
        )

        frame = _normalize_frame(
            raw
        )

        if frame.empty:
            return None

        frame = _filter_date_range(
            frame,
            start_date,
            end_date,
        )

        if frame.empty:
            return None

        result = pd.DataFrame(
            index=frame.index
        )

        for investor in INVESTOR_COLUMNS:
            target = (
                f"{investor}_순매수"
            )

            if investor in frame.columns:
                result[target] = _to_numeric(
                    frame[investor]
                )
            else:
                result[target] = np.nan

        if result.isna().all().all():
            return None

        return result

    # -----------------------------------------------------
    # 가격·투자자 병합
    # -----------------------------------------------------

    @staticmethod
    def merge_data(
        df_ohlcv,
        df_investor=None,
    ):
        prices = _normalize_frame(
            df_ohlcv
        )

        if prices.empty:
            return None

        frame = prices.copy()

        if (
            df_investor is not None
            and not df_investor.empty
        ):
            investors = _normalize_frame(
                df_investor
            )

            if not investors.empty:
                frame = frame.join(
                    investors,
                    how="left",
                )

        frame["날짜"] = (
            frame.index.strftime(
                "%Y-%m-%d"
            )
        )

        frame["년"] = (
            frame.index.year
        )

        frame["월"] = (
            frame.index.month
        )

        frame["일"] = (
            frame.index.day
        )

        frame["년월"] = (
            frame.index.strftime(
                "%Y-%m"
            )
        )

        frame["년주"] = (
            frame.index.strftime(
                "%Y-W%U"
            )
        )

        return frame.sort_index()


# =========================================================
# 앱 / 배치 실행용
# =========================================================

def investor_missing_report(frame, column):
    """
    투자자 순매수 누락을 (당일 미집계, 실제 누락)으로 나눕니다.

    마지막 거래일의 투자자 수량은 장 마감 후에 집계되므로 그 날 값이
    비어 있는 것은 정상입니다. 끝에서 이어지는 누락은 '당일 미집계'로 보고
    알리지 않으며, 중간에 실제로 빠진 날짜가 있을 때만 알립니다.

    반환값: (끝에서 이어지는 누락 수, 중간 누락 날짜 목록)
    """
    if frame is None or column not in frame.columns:
        return 0, []

    missing = frame[column].isna()
    flags = missing.to_numpy()

    if not flags.any():
        return 0, []

    trailing = 0

    for flag in flags[::-1]:
        if not flag:
            break

        trailing += 1

    dates = list(frame.index)
    missing_dates = [
        dates[position] for position, flag in enumerate(flags) if flag
    ]

    if trailing:
        missing_dates = missing_dates[:-trailing]

    def label(value):
        formatter = getattr(value, "strftime", None)

        return formatter("%Y-%m-%d") if formatter else str(value)

    return trailing, [label(value) for value in missing_dates]


def fetch_data(
    ticker,
    start_date,
    end_date,
    include_investors=True,
):
    """
    가격 및 선택적 투자자 순매수 수량만 반환합니다.

    공매도 API를 호출하지 않습니다.
    대차 API 또는 대차 CSV를 읽지 않습니다.

    Returns:
        frame: pandas.DataFrame
        messages: 안내 문자열 리스트
    """
    ticker, start_date, end_date = (
        _validate_request(
            ticker,
            start_date,
            end_date,
        )
    )

    messages = []

    # -----------------------------------------------------
    # 1. 가격
    # -----------------------------------------------------

    try:
        ohlcv = (
            DataCollector.fetch_ohlcv(
                ticker,
                start_date,
                end_date,
                include_today_live=False,
            )
        )

    except Exception:
        raise ValueError(
            "가격 데이터를 조회하지 못했습니다. "
            "조회 기간, pykrx 버전 및 "
            "KRX 인증 상태를 확인하세요."
        ) from None

    if (
        ohlcv is None
        or ohlcv.empty
    ):
        raise ValueError(
            "가격 데이터를 가져오지 못했습니다. "
            "종목코드, 기간, 인터넷 연결 또는 "
            "pykrx/KRX 인증 상태를 확인하세요."
        )

    # -----------------------------------------------------
    # 2. 투자자 순매수 수량
    # -----------------------------------------------------

    investors = None

    if include_investors:
        try:
            investors = (
                DataCollector.fetch_investor_trading(
                    ticker,
                    start_date,
                    end_date,
                )
            )

            if (
                investors is None
                or investors.empty
            ):
                investors = None

                messages.append(
                    "외국인·기관·개인 순매수 수량을 "
                    "가져오지 못했습니다. "
                    "KRX 인증과 조회 상태를 확인하세요."
                )

        except Exception:
            investors = None

            messages.append(
                "투자자 순매수 수량 조회 중 "
                "오류가 발생했습니다. "
                "가격 분석은 계속합니다."
            )

    # -----------------------------------------------------
    # 3. 병합
    # -----------------------------------------------------

    frame = (
        DataCollector.merge_data(
            ohlcv,
            investors,
        )
    )

    if frame is None or frame.empty:
        raise ValueError(
            "분석할 데이터가 없습니다."
        )

    if (
        include_investors
        and investors is not None
    ):
        for investor in INVESTOR_COLUMNS:
            # 마지막 거래일(당일)은 집계 전이라 비어 있는 것이 정상입니다.
            # 중간에 실제로 빠진 날짜가 있을 때만 알립니다.
            _, missing_dates = investor_missing_report(
                frame,
                f"{investor}_순매수",
            )

            if missing_dates:
                messages.append(
                    f"{investor} 순매수 수량이 "
                    f"{len(missing_dates)}개 날짜에서 "
                    "누락되었습니다"
                    f"({', '.join(missing_dates)}). "
                    "누락 이후 정확한 누적 수량은 "
                    "계산하지 않습니다."
                )

    frame = frame.sort_index()

    if frame.index.has_duplicates:
        frame = frame.loc[
            ~frame.index.duplicated(
                keep="last"
            )
        ].copy()

        messages.append(
            "중복 날짜를 제거했습니다."
        )

    return frame, messages
