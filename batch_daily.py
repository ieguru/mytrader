# batch_daily.py
import io
import logging
import os
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
from PIL import Image

from analysis_engine import analyze_frame
from charts import make_chart_png
from data_collector import fetch_data


BASE_DIR = Path(__file__).resolve().parent
STOCKS_FILE = BASE_DIR / "stocks.txt"
FINANCIAL_DIR = BASE_DIR / "file"
IMAGE_DIR = BASE_DIR / "img"
OUTPUT_DIR = BASE_DIR / "output"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def load_all_stocks():
    """모든 그룹의 종목을 읽고 중복을 제거합니다."""
    if not STOCKS_FILE.is_file():
        raise FileNotFoundError(f"종목 목록이 없습니다: {STOCKS_FILE}")

    stocks = {}
    current_group = None

    for line_number, raw_line in enumerate(
        STOCKS_FILE.read_text(encoding="utf-8-sig").splitlines(),
        start=1,
    ):
        line = raw_line.strip()

        if not line or line.startswith("#"):
            continue

        if line.startswith("[") and line.endswith("]"):
            current_group = line[1:-1].strip()
            if not current_group:
                raise ValueError(f"{line_number}행: 그룹명이 없습니다.")
            continue

        if current_group is None:
            raise ValueError(
                f"{line_number}행: 종목 위에 [그룹명]을 작성하세요."
            )

        fields = [value.strip() for value in line.split(",")]

        if len(fields) != 2:
            raise ValueError(
                f"{line_number}행: '종목명,종목코드' 형식이어야 합니다."
            )

        name, ticker = fields
        invalid_chars = '<>:"/\\|?*'

        if (
            not name
            or name in {".", ".."}
            or any(char in name for char in invalid_chars)
            or any(ord(char) < 32 for char in name)
            or name.endswith((".", " "))
        ):
            raise ValueError(f"{line_number}행: 잘못된 종목명입니다.")

        if not (
            ticker.isascii()
            and ticker.isdigit()
            and len(ticker) == 6
        ):
            raise ValueError(f"{line_number}행: 잘못된 종목코드입니다.")

        if name in stocks and stocks[name] != ticker:
            raise ValueError(
                f"{line_number}행: {name}의 종목코드가 중복·불일치합니다."
            )

        stocks[name] = ticker

    if not stocks:
        raise ValueError("등록된 종목이 없습니다.")

    return list(stocks.items())


def save_gif(png_bytes, target_path):
    """PNG를 실제 GIF 형식으로 저장합니다."""
    buffer = io.BytesIO()

    with Image.open(io.BytesIO(png_bytes)) as image:
        rgb = image.convert("RGB")
        try:
            rgb.save(buffer, format="GIF")
        finally:
            rgb.close()

    temporary_path = target_path.parent / f".{uuid.uuid4().hex}.tmp"

    try:
        temporary_path.write_bytes(buffer.getvalue())
        os.replace(temporary_path, target_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def main():
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    today = datetime.now(ZoneInfo("Asia/Seoul")).date()

    # 저녁 9시에 실행하므로 당일까지 조회합니다.
    # 토요일·일요일·휴일은 데이터 공급자가 반환한 최근 거래일까지 분석합니다.
    start = today - timedelta(days=365 * 5)
    date_text = today.strftime("%Y%m%d")

    stocks = load_all_stocks()
    records = []

    completed = 0
    failed = 0

    for index, (name, ticker) in enumerate(stocks, start=1):
        logger.info("[%s/%s] %s (%s)", index, len(stocks), name, ticker)

        excel_path = FINANCIAL_DIR / f"{name}.xlsx"

        if not excel_path.is_file():
            logger.warning("%s: 재무 엑셀 없음 — 건너뜀", name)
            records.append({
                "종목명": name,
                "종목코드": ticker,
                "상태": "건너뜀",
                "내용": f"file/{name}.xlsx 없음",
            })
            continue

        try:
            frame, messages = fetch_data(
                ticker,
                start.strftime("%Y%m%d"),
                today.strftime("%Y%m%d"),
                include_investors=True,
            )

            result = analyze_frame(
                frame,
                name,
                excel_file=str(excel_path),
            )

            png_bytes = make_chart_png(
                result["data"],
                name,
                excel_file=str(excel_path),
            )

            chart_path = IMAGE_DIR / f"{name}_{date_text}.gif"
            save_gif(png_bytes, chart_path)

            # 웹에서 최신 상태를 불러오는 기능을 추가할 때 사용할 수 있습니다.
            pd.DataFrame([result["latest"]]).to_json(
                OUTPUT_DIR / f"{name}_{date_text}_latest.json",
                orient="records",
                force_ascii=False,
                indent=2,
            )

            result["metrics"].to_csv(
                OUTPUT_DIR / f"{name}_{date_text}_metrics.csv",
                index=False,
                encoding="utf-8-sig",
            )

            for strategy, trades in result["trades"].items():
                trades.to_csv(
                    OUTPUT_DIR / f"{name}_{date_text}_{strategy}_trades.csv",
                    index=False,
                    encoding="utf-8-sig",
                )

            all_messages = messages + result["messages"]

            records.append({
                "종목명": name,
                "종목코드": ticker,
                "상태": "완료",
                "내용": f"img/{chart_path.name}",
                "데이터 기준일": result["latest"]["기준일"],
                "경고": " / ".join(all_messages),
            })

            completed += 1
            logger.info("%s: 저장 완료 — %s", name, chart_path.name)

        except Exception as error:
            failed += 1
            logger.exception("%s: 분석 실패", name)

            records.append({
                "종목명": name,
                "종목코드": ticker,
                "상태": "실패",
                "내용": str(error),
            })

        finally:
            # 종목 간 최소 간격. 공급자의 호출 제한 정책도 확인하세요.
            time.sleep(2)

    pd.DataFrame(records).to_csv(
        OUTPUT_DIR / f"run_status_{date_text}.csv",
        index=False,
        encoding="utf-8-sig",
    )

    logger.info(
        "전체 실행 완료: 성공 %s / 실패 %s / 전체 %s",
        completed,
        failed,
        len(stocks),
    )

    # 일부 실패도 Actions에서 확인할 수 있도록 실패 상태로 종료합니다.
    # 성공한 결과와 실행 내역은 이미 저장되어 있습니다.
    if failed:
        return 1

    if not completed:
        logger.error("분석 완료된 종목이 없습니다. 재무 파일을 확인하세요.")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
