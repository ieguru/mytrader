"""CSV만 읽어 지표·point-in-time 신호를 계산하고 processed/에 저장합니다. 네트워크 호출 없음."""
import hashlib
import json
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from analysis_engine import (
    NOMAD_SELL_DRY_RATIO,
    analyze_frame,
    detect_double_patterns,
    nomad_patterns,
)

from .config import MARKET_CSV, PROCESSED_DIR
from .features import SIGNAL_MAP
from .storage import INVESTOR_COLUMNS, raw_path, read_csv, read_raw, write_csv

logger = logging.getLogger(__name__)

PIPELINE_VERSION = "pit-v1"      # 이 파일의 계산 방식을 바꾸면 올리세요
PIT_COLUMNS = ("유목민신호", "유목민매도신호", "쌍바닥신호", "쌍고점신호")
DOUBLE_LOOKBACK = 260

PROCESSED_COLUMNS = [
    "시가", "고가", "저가", "종가", "거래량",
    "Daily_Oscillator", "Weekly_Oscillator", "Monthly_Oscillator",
    "Weekly_Trend", "Monthly_Trend", "RSI30", "이격도20", "이격도5",
    *INVESTOR_COLUMNS, *SIGNAL_MAP.keys(),
]

ENGINE_FILE = Path(__file__).resolve().parent.parent / "analysis_engine.py"


def patterns_at(df, t):
    """t일 장 마감 시점에 실시간으로 보였을 패턴 신호 4개(0/1)."""
    tail = df.iloc[max(0, t - 5): t + 1]
    buy = sell = 0

    if len(tail) >= 3:
        buy = int(nomad_patterns(tail)["유목민신호"].iat[-1])
        sell = int(nomad_patterns(tail, direction="sell", dry_ratio=NOMAD_SELL_DRY_RATIO)
                   ["유목민신호"].iat[-1])

    double = detect_double_patterns(df.iloc[max(0, t - DOUBLE_LOOKBACK): t + 1])
    return buy, sell, int(double["쌍바닥신호"].iat[-1]), int(double["쌍고점신호"].iat[-1])


def point_in_time_patterns(df):
    values = [patterns_at(df, t) for t in range(len(df))]
    return pd.DataFrame(values, index=df.index, columns=PIT_COLUMNS)


def fingerprint(ticker):
    """원본 CSV, analysis_engine.py, 파이프라인 버전 중 하나라도 바뀌면 값이 달라집니다."""
    h = hashlib.sha1(PIPELINE_VERSION.encode())
    h.update(ENGINE_FILE.read_bytes())
    h.update(raw_path(ticker).read_bytes())
    return h.hexdigest()


def process_stock(name, ticker, refresh=False):
    out_csv = PROCESSED_DIR / f"{ticker}.csv"
    meta = PROCESSED_DIR / f"{ticker}.json"
    fp = fingerprint(ticker)

    if not refresh and out_csv.is_file() and meta.is_file():
        if json.loads(meta.read_text(encoding="utf-8")).get("fingerprint") == fp:
            return read_csv(out_csv)

    raw = read_raw(ticker)
    # 목표가·PER은 미래 정보 문제가 있어 쓰지 않으므로 엑셀 없이 분석합니다.
    data = analyze_frame(raw, name, excel_file=None)["data"]

    pit = point_in_time_patterns(data)
    for column in PIT_COLUMNS:
        data[column] = pit[column].to_numpy()

    missing = [c for c in PROCESSED_COLUMNS if c not in data]
    if missing:
        logger.warning("%s: 엔진 결과에 없는 컬럼 %s", name, missing)

    data = data[[c for c in PROCESSED_COLUMNS if c in data]]
    write_csv(data, out_csv)
    meta.write_text(json.dumps({"name": name, "fingerprint": fp, "rows": len(data)},
                               ensure_ascii=False), encoding="utf-8")
    return read_csv(out_csv)   # 학습에는 항상 'CSV에서 읽은 값'을 씁니다


def _worker(job):
    name, ticker, refresh = job
    try:
        return ticker, name, process_stock(name, ticker, refresh), None
    except Exception as error:
        return ticker, name, None, f"{type(error).__name__}: {error}"


def load_all(universe, refresh=False, workers=1):
    """{종목코드: processed DataFrame}. 처음 한 번만 오래 걸립니다."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    jobs = [(n, t, refresh) for n, t in universe]
    data = {}

    def collect(result):
        ticker, name, frame, error = result
        if error:
            logger.warning("%s(%s) 건너뜀: %s", name, ticker, error)
        else:
            data[ticker] = frame

    if workers <= 1:
        for i, job in enumerate(jobs, 1):
            logger.info("[처리 %d/%d] %s", i, len(jobs), job[0])
            collect(_worker(job))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_worker, job) for job in jobs]
            for i, future in enumerate(as_completed(futures), 1):
                collect(future.result())
                logger.info("[처리 %d/%d]", i, len(jobs))

    return {t: data[t] for _, t in universe if t in data}   # 원래 순서 유지


def load_market(data):
    """market.csv가 있으면 지수, 없으면 등록 종목 평균으로 대신합니다."""
    if MARKET_CSV.is_file():
        close = read_csv(MARKET_CSV)
        market = pd.DataFrame({
            "kospi_ret": np.log(close["kospi_close"]).diff(),
            "kosdaq_ret": np.log(close["kosdaq_close"]).diff(),
        })
    else:
        logger.warning("market.csv 없음: 등록 종목 평균 수익률로 대신합니다.")
        rets = pd.DataFrame({t: np.log(d["종가"]).diff() for t, d in data.items()})
        market = pd.DataFrame({"kospi_ret": rets.mean(axis=1),
                               "kosdaq_ret": rets.median(axis=1)})

    market["kospi_vol20"] = market["kospi_ret"].rolling(20).std() * np.sqrt(252)
    return market
