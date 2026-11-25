"""CSV 읽기·쓰기와 형식 검증. pykrx를 import하지 않습니다."""
import os
import uuid

import numpy as np
import pandas as pd

from .config import RAW_DIR, universe_csv

PRICE_COLUMNS = ["시가", "고가", "저가", "종가", "거래량"]
INVESTOR_COLUMNS = ["외국인합계_순매수", "기관합계_순매수", "개인_순매수"]
RAW_COLUMNS = PRICE_COLUMNS + ["등락률"] + INVESTOR_COLUMNS


def write_csv(frame, path):
    """임시 파일에 쓴 뒤 교체합니다(중간에 끊겨도 기존 파일이 남습니다)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"

    try:
        frame.to_csv(temporary, index_label="날짜", date_format="%Y-%m-%d",
                     encoding="utf-8-sig")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def read_csv(path):
    frame = pd.read_csv(path, index_col="날짜", parse_dates=["날짜"], encoding="utf-8-sig")
    frame.index = pd.DatetimeIndex(frame.index).normalize()

    if frame.index.has_duplicates:
        raise ValueError(f"{path.name}: 중복 날짜가 있습니다.")
    if not frame.index.is_monotonic_increasing:
        frame = frame.sort_index()

    for column in frame.columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    return frame


def raw_path(ticker):
    return RAW_DIR / f"{ticker}.csv"


def read_raw(ticker):
    """원본 CSV를 fetch_data()와 같은 모양으로 읽습니다."""
    path = raw_path(ticker)

    if not path.is_file():
        raise FileNotFoundError(f"{path.name} 이 없습니다. 먼저 python -m rl.download 를 실행하세요.")

    frame = read_csv(path)
    missing = set(PRICE_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"{path.name}: 컬럼 누락 {sorted(missing)}")

    for column in RAW_COLUMNS:
        if column not in frame:
            frame[column] = np.nan

    # data_collector.merge_data가 붙이는 파생 컬럼을 똑같이 만듭니다.
    idx = frame.index
    frame["날짜"] = idx.strftime("%Y-%m-%d")
    frame["년"], frame["월"], frame["일"] = idx.year, idx.month, idx.day
    frame["년월"] = idx.strftime("%Y-%m")
    frame["년주"] = idx.strftime("%Y-W%U")
    frame.index.name = "날짜"

    return frame


def check_raw(frame):
    """저장 전 점검. 경고 문자열 목록을 돌려줍니다."""
    warnings = []

    if (frame["종가"] <= 0).any():
        warnings.append("종가 0 이하 행 존재")

    gaps = frame.index.to_series().diff().dt.days
    long_gaps = gaps[gaps > 10]
    if len(long_gaps):
        warnings.append(f"10일 넘는 공백 {len(long_gaps)}곳(거래정지 가능) "
                        f"예: {long_gaps.index[0].date()}")

    for column in INVESTOR_COLUMNS:
        if column in frame:
            mid = frame[column].iloc[:-1].isna().sum()   # 마지막 날 미집계는 정상
            if mid:
                warnings.append(f"{column} 중간 누락 {mid}일")

    return warnings


def read_universe(kind):
    path = universe_csv(kind)

    if not path.is_file():
        raise FileNotFoundError(f"{path.name} 이 없습니다. python -m rl.download --universe {kind}")

    table = pd.read_csv(path, dtype={"종목코드": str}, encoding="utf-8-sig")
    table["종목코드"] = table["종목코드"].str.strip().str.zfill(6)
    return list(zip(table["종목명"], table["종목코드"]))


def write_universe(kind, items, source):
    table = pd.DataFrame(items, columns=["종목명", "종목코드"]).drop_duplicates("종목코드")
    table["출처"] = source
    path = universe_csv(kind)
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False, encoding="utf-8-sig")
