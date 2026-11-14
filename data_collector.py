# data_collector.py
"""
가격, 투자자 순매수 수량, 공매도 거래 수량,
대차 체결·상환 수량을 수집합니다.

투자자·공매도:
    기존 pykrx 조회 사용

대차:
    프로젝트의 loan/종목코드.csv 파일 사용
    현재 버전은 대차 API 자동조회가 아닙니다.

단위:
    모든 수량은 주

주의:
- import만으로 데이터를 수집하지 않습니다.
- 조회 실패 및 누락값을 0으로 보정하지 않습니다.
- 장중 시세를 별도로 합성하지 않습니다.
- 계정정보를 이 파일에 넣지 않습니다.
"""

import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
from pykrx import stock


logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
LOAN_DIR = BASE_DIR / "loan"

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

LOAN_COLUMNS = [
    "대차체결수량",
    "대차상환수량",
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
    """날짜 인덱스, 컬럼명, 정렬을 정리합니다."""
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

    # 숫자가 날짜의 나노초로 해석되지 않도록 문자열화
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
# 데이터 수집 클래스
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
        일시적 오류와 빈 응답을 재시도합니다.

        인증정보가 포함될 수 있는 인자나
        원본 오류 내용은 출력하지 않습니다.
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
                "pykrx 일별 데이터만 사용합니다."
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
    # 공매도 거래 수량
    # -----------------------------------------------------

    @staticmethod
    def fetch_shorting_volume(
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

        fetch_function = getattr(
            stock,
            "get_shorting_volume_by_date",
            None,
        )

        if fetch_function is None:
            raise ValueError(
                "설치된 pykrx에서 공매도 거래량 "
                "조회 함수를 찾을 수 없습니다."
            )

        raw = (
            DataCollector._call_pykrx_safely(
                fetch_function,
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

        source_column = next(
            (
                candidate
                for candidate in (
                    "공매도",
                    "공매도거래량",
                    "공매도 거래량",
                )
                if candidate in frame.columns
            ),
            None,
        )

        if source_column is None:
            raise ValueError(
                "공매도 수량 컬럼을 찾지 못했습니다. "
                "반환 컬럼: "
                + ", ".join(
                    map(str, frame.columns)
                )
            )

        result = pd.DataFrame(
            index=frame.index
        )

        values = _to_numeric(
            frame[source_column]
        )

        result["공매도수량"] = (
            values.where(values >= 0)
        )

        if (
            result["공매도수량"]
            .isna()
            .all()
        ):
            return None

        return result

    # -----------------------------------------------------
    # 대차 CSV
    # -----------------------------------------------------

    @staticmethod
    def fetch_loan_volume(
        ticker,
        start_date,
        end_date,
    ):
        """
        loan/종목코드.csv를 읽습니다.

        필수 컬럼:
            날짜
            대차체결수량
            대차상환수량

        날짜 형식:
            YYYY-MM-DD 또는 YYYYMMDD

        수량 단위:
            주

        조회 실패나 빈칸을 0으로 보정하지 않습니다.
        API 자동조회는 수행하지 않습니다.
        """
        ticker, start_date, end_date = (
            _validate_request(
                ticker,
                start_date,
                end_date,
            )
        )

        path = (
            LOAN_DIR / f"{ticker}.csv"
        )

        if not path.is_file():
            return None

        frame = pd.read_csv(
            path,
            encoding="utf-8-sig",
            dtype="string",
        )

        frame.columns = [
            str(column).strip()
            for column in frame.columns
        ]

        if frame.columns.duplicated().any():
            raise ValueError(
                "대차 CSV에 중복된 컬럼이 있습니다."
            )

        required = {
            "날짜",
            "대차체결수량",
            "대차상환수량",
        }

        missing = (
            required - set(frame.columns)
        )

        if missing:
            raise ValueError(
                "대차 CSV 필수 컬럼 누락: "
                + ", ".join(sorted(missing))
            )

        if frame.empty:
            return None

        # 완전히 빈 행만 제외
        frame = frame.dropna(
            subset=list(required),
            how="all",
        ).copy()

        if frame.empty:
            return None

        date_text = (
            frame["날짜"]
            .astype("string")
            .str.strip()
        )

        dates = pd.Series(
            pd.NaT,
            index=frame.index,
            dtype="datetime64[ns]",
        )

        compact_mask = (
            date_text
            .str.fullmatch(r"\d{8}")
            .fillna(False)
        )

        dashed_mask = (
            date_text
            .str.fullmatch(
                r"\d{4}-\d{2}-\d{2}"
            )
            .fillna(False)
        )

        dates.loc[compact_mask] = (
            pd.to_datetime(
                date_text.loc[compact_mask],
                format="%Y%m%d",
                errors="coerce",
            )
        )

        dates.loc[dashed_mask] = (
            pd.to_datetime(
                date_text.loc[dashed_mask],
                format="%Y-%m-%d",
                errors="coerce",
            )
        )

        if dates.isna().any():
            raise ValueError(
                "대차 CSV 날짜는 YYYY-MM-DD 또는 "
                "YYYYMMDD 형식이어야 합니다."
            )

        if dates.duplicated().any():
            raise ValueError(
                "대차 CSV에 중복된 날짜가 있습니다. "
                "종목별·날짜별 한 행으로 정리하세요."
            )

        result = pd.DataFrame(
            index=pd.DatetimeIndex(
                dates.to_numpy()
            )
        )

        result.index.name = "날짜"

        for column in LOAN_COLUMNS:
            original = frame[column]

            converted = _to_numeric(
                original
            )

            supplied = (
                original.notna()
                & original.str.strip().ne("")
            )

            invalid = (
                supplied
                & converted.isna()
            )

            if invalid.any():
                raise ValueError(
                    f"대차 CSV의 '{column}'에는 "
                    "숫자 또는 빈칸만 입력하세요."
                )

            if (
                converted.dropna() < 0
            ).any():
                raise ValueError(
                    f"대차 CSV의 '{column}'에는 "
                    "음수를 입력할 수 없습니다."
                )

            # 주 단위 정수 수량만 허용
            valid = converted.dropna()

            if (
                valid.mod(1).ne(0)
            ).any():
                raise ValueError(
                    f"대차 CSV의 '{column}'은 "
                    "주 단위 정수여야 합니다."
                )

            # 날짜 인덱스와 원본 행 인덱스의
            # 자동 정렬을 피하기 위해 배열 대입
            result[column] = (
                converted.to_numpy(
                    dtype=float,
                    na_value=np.nan,
                )
            )

        result = (
            result
            .sort_index()
        )

        result = _filter_date_range(
            result,
            start_date,
            end_date,
        )

        if (
            result.empty
            or result.isna().all().all()
        ):
            return None

        return result

    # -----------------------------------------------------
    # 병합
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

def fetch_data(
    ticker,
    start_date,
    end_date,
    include_investors=True,
):
    """
    Returns:
        frame: 가격·수급·공매도·대차 DataFrame
        messages: 안내 및 경고 문자열 리스트

    투자자 데이터:
        include_investors에 따라 조회

    공매도·대차:
        투자자 체크와 관계없이 처리
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
    # 2. 투자자
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
            column = (
                f"{investor}_순매수"
            )

            missing_count = int(
                frame[column]
                .isna()
                .sum()
            )

            if missing_count:
                messages.append(
                    f"{investor} 순매수 수량이 "
                    f"{missing_count}개 날짜에서 "
                    "누락되었습니다. "
                    "정확한 누적합을 위해 누락 이후 "
                    "누적값은 표시하지 않습니다."
                )

    # -----------------------------------------------------
    # 3. 공매도
    # -----------------------------------------------------

    try:
        shorting = (
            DataCollector.fetch_shorting_volume(
                ticker,
                start_date,
                end_date,
            )
        )

        if (
            shorting is not None
            and not shorting.empty
        ):
            frame = frame.join(
                shorting,
                how="left",
            )

            missing_count = int(
                frame["공매도수량"]
                .isna()
                .sum()
            )

            if missing_count:
                messages.append(
                    "공매도 거래 수량이 "
                    f"{missing_count}개 날짜에서 "
                    "누락되었습니다. "
                    "누락값은 실제 0주와 구분합니다."
                )

        else:
            messages.append(
                "공매도 거래 수량을 가져오지 못했습니다. "
                "KRX 인증, pykrx 버전 및 "
                "데이터 제공 여부를 확인하세요."
            )

    except ValueError as error:
        messages.append(
            f"공매도 조회 안내: {error}"
        )

    except Exception:
        messages.append(
            "공매도 조회 중 오류가 발생했습니다. "
            "다른 분석은 계속합니다."
        )

    # -----------------------------------------------------
    # 4. 대차 CSV
    # -----------------------------------------------------

    try:
        loans = (
            DataCollector.fetch_loan_volume(
                ticker,
                start_date,
                end_date,
            )
        )

        if (
            loans is not None
            and not loans.empty
        ):
            frame = frame.join(
                loans,
                how="left",
            )

            messages.append(
                f"대차 수량은 loan/{ticker}.csv에서 "
                "읽었습니다. 원본 제공처의 수집 범위와 "
                "단위 및 이용조건을 확인하세요."
            )

            for column in LOAN_COLUMNS:
                missing_count = int(
                    frame[column]
                    .isna()
                    .sum()
                )

                if missing_count:
                    messages.append(
                        f"{column}이 "
                        f"{missing_count}개 가격 데이터 "
                        "날짜에서 누락되었습니다. "
                        "누락값은 0으로 보정하지 않습니다."
                    )

        else:
            messages.append(
                "조회 기간의 대차 체결·상환 수량이 없습니다. "
                f"loan/{ticker}.csv 파일을 확인하세요. "
                "현재 버전은 대차 API 자동조회가 아닙니다."
            )

    except (
        ValueError,
        UnicodeError,
        pd.errors.ParserError,
        pd.errors.EmptyDataError,
    ) as error:
        messages.append(
            f"대차 CSV 안내: {error}"
        )

    except Exception:
        messages.append(
            "대차 CSV를 읽지 못했습니다. "
            "다른 분석은 계속합니다."
        )

    # -----------------------------------------------------
    # 5. 최종 정리
    # -----------------------------------------------------

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
