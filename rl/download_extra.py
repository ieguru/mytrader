"""rl/download_extra.py : 추가 데이터 다운로드 (pykrx)
PER·PBR·배당수익률·시가총액·상장주식수·공매도잔고·외국인지분율 -> data/rl/extra/{코드}.csv

실행:
  python -m rl.download_extra                  # processed/ 종목 전체, 이어받기
  python -m rl.download_extra --force          # 처음부터 다시
  python -m rl.download_extra --tickers 000660 005930
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time

import pandas as pd

from rl.features_extra import EXTRA_DIR, extra_path
from rl.strategies import select_files

logging.raiseExceptions = False      # pykrx 로깅 버그로 생기는 긴 Traceback 숨김
logging.getLogger().setLevel(logging.WARNING)

SHORT_START = pd.Timestamp("2016-06-01")   # 공매도 잔고 공시 시작 무렵
OVERLAP_DAYS = 14                           # 이어받기 시 겹쳐 받는 기간
RETRY = 3


def login():
    """KRX 로그인 정보(환경변수 또는 .streamlit/secrets.toml)를 준비."""
    try:
        from rl.download import load_krx_credentials
        load_krx_credentials()
    except Exception as e:
        print(f"[경고] KRX 로그인 정보 준비 실패: {e}")


def _fns(stock):
    """(이름, 함수, 열 변환 dict, 시작 하한) 목록. pykrx 버전에 따라 없는 함수는 건너뜀."""
    def pick(*names):
        for n in names:
            if hasattr(stock, n):
                return getattr(stock, n)
        return None
    cand = [
        ("fundamental", pick("get_market_fundamental_by_date", "get_market_fundamental"),
         {"PER": "PER", "PBR": "PBR", "DIV": "DIV"}, None),
        ("cap", pick("get_market_cap_by_date", "get_market_cap"),
         {"시가총액": "시가총액", "상장주식수": "상장주식수"}, None),
        ("short", pick("get_shorting_balance_by_date"),
         {"공매도잔고": "공매도잔고", "상장주식수": "상장주식수_공매도"}, SHORT_START),
        ("foreign", pick("get_exhaustion_rates_of_foreign_investment_by_date",
                         "get_exhaustion_rates_of_foreign_investment"),
         {"지분율": "외국인지분율"}, None),
    ]
    return [c for c in cand if c[1] is not None]


def fetch_one(fn, cols, code, start, end, floor, sleep):
    """연도별로 나눠 조회. 실패 연도는 3회 재시도 후 건너뜀."""
    s0 = max(start, floor) if floor is not None else start
    parts = []
    for y in range(s0.year, end.year + 1):
        s = max(s0, pd.Timestamp(f"{y}-01-01"))
        e = min(end, pd.Timestamp(f"{y}-12-31"))
        if s > e:
            continue
        for k in range(RETRY):
            try:
                d = fn(s.strftime("%Y%m%d"), e.strftime("%Y%m%d"), code)
                break
            except Exception:
                d = None
                time.sleep(sleep * (k + 2))
        time.sleep(sleep)
        if d is None or len(d) == 0:
            continue
        d = d.rename(columns=cols)[[v for v in cols.values() if v in d.rename(columns=cols).columns]]
        if len(d.columns):
            parts.append(d)
    if not parts:
        return None
    out = pd.concat(parts)
    out.index = pd.to_datetime(out.index)
    return out[~out.index.duplicated(keep="last")]


def write_csv(df, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    df.to_csv(tmp, encoding="utf-8-sig", index_label="날짜")
    os.replace(tmp, path)


def download_code(stock, code, start, end, force, sleep):
    """반환: 저장한 행 수 (-1 = 이미 최신이라 건너뜀, 0 = 받은 데이터 없음)"""
    path = extra_path(code)
    old = None
    if path.exists() and not force:
        old = pd.read_csv(path, index_col=0, parse_dates=True, encoding="utf-8-sig")
        if len(old):
            last = old.index.max()
            if last >= end - pd.Timedelta(days=3):
                return -1
            start = max(start, last - pd.Timedelta(days=OVERLAP_DAYS))
    frames = []
    for name, fn, cols, floor in _fns(stock):
        d = fetch_one(fn, cols, code, start, end, floor, sleep)
        if d is None:
            print(f"    {code} {name}: 데이터 없음")
            continue
        frames.append(d)
    if not frames:
        return 0
    new = pd.concat(frames, axis=1)
    if "상장주식수_공매도" in new.columns:            # 시총 쪽 상장주식수가 없을 때만 보충
        if "상장주식수" not in new.columns:
            new["상장주식수"] = new["상장주식수_공매도"]
        else:
            new["상장주식수"] = new["상장주식수"].fillna(new["상장주식수_공매도"])
        new = new.drop(columns="상장주식수_공매도")
    new = new.apply(pd.to_numeric, errors="coerce").sort_index()
    if old is not None and len(old):
        new = pd.concat([old[old.index < new.index.min()], new])
    new = new[~new.index.duplicated(keep="last")].sort_index()
    write_csv(new, path)
    return len(new)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="추가 데이터(PER·PBR·시총·공매도·외국인지분율) 다운로드")
    p.add_argument("--universe", default=None, choices=["stocks", "kospi200"])
    p.add_argument("--tickers", nargs="*", default=None)
    p.add_argument("--max-stocks", type=int, default=None)
    p.add_argument("--start", default="2015-01-01")
    p.add_argument("--end", default=pd.Timestamp.today().strftime("%Y-%m-%d"))
    p.add_argument("--force", action="store_true", help="이미 받은 파일도 처음부터 다시")
    p.add_argument("--sleep", type=float, default=1.0, help="KRX 호출 간 대기(초)")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    login()
    from pykrx import stock

    files = select_files(args.universe, args.tickers, args.max_stocks)
    if not files:
        print("processed 폴더에 종목 CSV가 없습니다. STEP 1을 먼저 실행하세요.")
        return 1
    start, end = pd.Timestamp(args.start), pd.Timestamp(args.end)
    print(f"대상 {len(files)}종목, {start.date()} ~ {end.date()}, 저장 위치 {EXTRA_DIR}")
    print(f"조회 항목: {[n for n, *_ in _fns(stock)]}")

    failed, t0 = [], time.time()
    for i, p in enumerate(files, 1):
        code = p.stem
        try:
            n = download_code(stock, code, start, end, args.force, args.sleep)
            msg = "이미 최신 (건너뜀)" if n < 0 else ("받은 데이터 없음" if n == 0 else f"{n}행 저장")
            if n == 0:
                failed.append(code)
        except Exception as e:
            msg = f"실패: {e}"
            failed.append(code)
        print(f"[{i}/{len(files)}] {code} {msg}  (경과 {(time.time() - t0) / 60:.1f}분)", flush=True)

    print(f"\n완료: {len(files) - len(failed)}/{len(files)}종목")
    if failed:
        print(f"실패/데이터 없음 종목: {failed}")
        print("다시 실행하면 실패 종목만 이어서 받습니다.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
