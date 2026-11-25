"""
KRX에서 한 번에 받아 CSV로 저장합니다. 학습 단계는 이 파일들만 씁니다.

python -m rl.download --universe stocks              # stocks.txt 종목
python -m rl.download --universe kospi200            # data_start 시점 KOSPI200
python -m rl.download --universe stocks --update     # 새 거래일만 추가
python -m rl.download --universe stocks --force      # 전부 다시 받기
python -m rl.download --tickers 000660,005930        # 일부만
"""
import argparse
import logging
import os
import time
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .config import BASE_DIR, MANIFEST_CSV, MARKET_CSV, RLConfig
from .storage import RAW_COLUMNS, check_raw, raw_path, read_csv, write_csv, write_universe

logger = logging.getLogger("rl.download")
OVERLAP_DAYS = 20      # 업데이트 때 겹쳐 받는 기간(마지막 날 미집계 수급 보완)


def load_krx_credentials():
    """환경변수가 없으면 .streamlit/secrets.toml에서 KRX_ID/KRX_PW를 읽습니다."""
    if os.environ.get("KRX_ID") and os.environ.get("KRX_PW"):
        return

    path = BASE_DIR / ".streamlit" / "secrets.toml"
    if not path.is_file():
        logger.warning("secrets.toml 이 없습니다: KRX_ID/KRX_PW 환경변수를 설정하세요.")
        return

    try:
        import tomllib   # Python 3.11+
    except ImportError:
        logger.warning("Python 3.10 이하: KRX_ID/KRX_PW 환경변수를 직접 설정하세요.")
        return

    try:
        # utf-8-sig: Visual Studio가 붙이는 BOM(서명)을 자동으로 제거합니다.
        text = path.read_text(encoding="utf-8-sig")
        secrets = tomllib.loads(text)
    except UnicodeDecodeError:
        logger.error(
            "secrets.toml 이 UTF-8이 아닙니다. Visual Studio에서 "
            "'인코딩하여 저장 -> 유니코드(서명 없는 UTF-8)'로 다시 저장하세요."
        )
        return
    except tomllib.TOMLDecodeError as error:
        # 오류 메시지에는 위치만 들어 있고 비밀번호 값은 들어 있지 않습니다.
        logger.error(
            "secrets.toml 문법 오류(%s). 형식 예: KRX_ID = \"아이디\" "
            "(따옴표 필수, 주석은 #).", error
        )
        return

    found = []
    for key in ("KRX_ID", "KRX_PW"):
        if secrets.get(key) and not os.environ.get(key):
            os.environ[key] = str(secrets[key])
            found.append(key)

    if len(found) < 2 and not (os.environ.get("KRX_ID") and os.environ.get("KRX_PW")):
        logger.warning("secrets.toml 에 KRX_ID 또는 KRX_PW 가 없습니다.")

def fetch_raw(ticker, start, end):
    from data_collector import fetch_data   # 인증정보 설정 뒤에 import

    frame, messages = fetch_data(ticker, start.replace("-", ""), end.replace("-", ""),
                                 include_investors=True)
    return frame.reindex(columns=RAW_COLUMNS), messages


def merge_update(old, new):
    """
    겹치는 구간 종가가 다르면 수정주가가 바뀐 것(분할·배당 등)입니다.
    이어 붙이면 과거와 현재의 가격 기준이 섞이므로 None을 돌려 전체를 다시 받게 합니다.
    """
    overlap = old.index.intersection(new.index)

    if len(overlap) == 0:
        return None, "겹치는 날짜 없음"

    if not np.allclose(old.loc[overlap, "종가"], new.loc[overlap, "종가"], rtol=1e-4):
        return None, "수정주가 변경 감지"

    return pd.concat([old.loc[old.index < new.index[0], RAW_COLUMNS], new]), None


def download_one(name, ticker, start, end, mode):
    path = raw_path(ticker)
    note = ""

    if path.is_file() and mode == "skip":
        frame = read_csv(path)
        return frame, "기존 파일 사용", []

    if path.is_file() and mode == "update":
        old = read_csv(path)
        since = (old.index[-1] - pd.Timedelta(days=OVERLAP_DAYS)).strftime("%Y-%m-%d")
        new, messages = fetch_raw(ticker, since, end)
        merged, reason = merge_update(old, new)

        if merged is not None:
            write_csv(merged, path)
            return merged, f"업데이트 +{len(merged) - len(old)}행", messages

        note = f"{reason} -> 전체 다시 받기 / "

    frame, messages = fetch_raw(ticker, start, end)
    write_csv(frame, path)
    return frame, note + "전체 저장", messages


def download_market(start, end):
    from pykrx import stock

    closes = {}
    for code, column in (("1001", "kospi_close"), ("2001", "kosdaq_close")):
        table = stock.get_index_ohlcv(start.replace("-", ""), end.replace("-", ""), code)
        closes[column] = pd.to_numeric(table["종가"], errors="coerce")
        time.sleep(1.0)

    market = pd.DataFrame(closes)
    market.index = pd.to_datetime(market.index).normalize()
    write_csv(market, MARKET_CSV)
    logger.info("시장지수 저장: %d행", len(market))


def build_universe(kind, as_of):
    if kind == "stocks":
        from batch_daily import load_all_stocks
        return load_all_stocks(), "stocks.txt"

    from pykrx import stock

    day = pd.Timestamp(as_of)
    for _ in range(10):   # 휴장일이면 다음 날로
        tickers = stock.get_index_portfolio_deposit_file("1028", day.strftime("%Y%m%d"))
        if len(tickers):
            items = [(stock.get_market_ticker_name(t), t) for t in tickers]
            return items, f"KOSPI200 {day.date()}"
        day += pd.Timedelta(days=1)

    raise ValueError("KOSPI200 구성종목을 가져오지 못했습니다.")


def update_manifest(rows):
    new = pd.DataFrame(rows)
    if MANIFEST_CSV.is_file():
        old = pd.read_csv(MANIFEST_CSV, dtype={"종목코드": str}, encoding="utf-8-sig")
        new = pd.concat([old[~old["종목코드"].isin(new["종목코드"])], new])
    new.sort_values("종목코드").to_csv(MANIFEST_CSV, index=False, encoding="utf-8-sig")


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--universe", choices=["stocks", "kospi200"], default="stocks")
    p.add_argument("--start", default=RLConfig.data_start)
    p.add_argument("--end", default=date.today().isoformat())
    p.add_argument("--tickers", help="쉼표로 구분한 종목코드만 받기")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--update", action="store_true")
    g.add_argument("--force", action="store_true")
    p.add_argument("--skip-market", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    load_krx_credentials()
    mode = "update" if args.update else "force" if args.force else "skip"

    items, source = build_universe(args.universe, args.start)
    write_universe(args.universe, items, source)

    if args.tickers:
        wanted = {t.strip().zfill(6) for t in args.tickers.split(",")}
        items = [(n, t) for n, t in items if t in wanted]

    rows = []
    for i, (name, ticker) in enumerate(items, start=1):
        logger.info("[%d/%d] %s (%s)", i, len(items), name, ticker)
        row = {"종목코드": ticker, "종목명": name,
               "저장시각": datetime.now().isoformat(timespec="seconds")}

        try:
            frame, action, messages = download_one(name, ticker, args.start, args.end, mode)
            warnings = check_raw(frame)
            row.update({"상태": "완료", "처리": action, "행수": len(frame),
                        "시작일": frame.index[0].date(), "종료일": frame.index[-1].date(),
                        "경고": " / ".join(warnings + list(messages or []))})
            if warnings:
                logger.warning("%s: %s", name, " / ".join(warnings))
        except Exception as error:
            row.update({"상태": "실패", "경고": f"{type(error).__name__}: {error}"})
            logger.error("%s 실패: %s", name, error)

        rows.append(row)
        if row.get("처리") != "기존 파일 사용":
            time.sleep(1.5)   # KRX 호출 간격

    update_manifest(rows)

    if not args.skip_market:
        try:
            download_market(args.start, args.end)
        except Exception as error:
            logger.warning("시장지수 실패(%s): 학습 때 종목 평균으로 대신합니다.", error)

    failed = sum(r["상태"] == "실패" for r in rows)
    logger.info("완료 %d / 실패 %d -> %s", len(rows) - failed, failed, MANIFEST_CSV)
    return 0 if failed < len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
