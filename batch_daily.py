# batch_daily.py
"""
매일 정해진 시각에 모든 종목을 분석하는 배치입니다.
(예: GitHub Actions cron 20:01 KST — .github/workflows/daily-analysis.yml)

산출물
    img/{종목명}_{YYYYMMDD}.png            종합차트(원본)
    output/{종목명}_{YYYYMMDD}_latest.json  종목별 최신 신호
    output/{종목명}_{YYYYMMDD}_metrics.csv  전략별 성과
    output/{종목명}_{YYYYMMDD}_*_trades.csv 거래 내역
    output/run_status_{YYYYMMDD}.csv        실행 내역
    output/last_analysis.json.gz            앱이 재접속 시 자동 복원하는 스냅샷

알림
    TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID 가 있으면
    실행 요약과 신호 종목 차트를 텔레그램으로 보냅니다.
    실패해도 배치 결과와 종료코드에는 영향을 주지 않습니다.

설계 메모
- 재무 엑셀이 없는 종목도 분석합니다(목표가·괴리율·PER만 비움).
  '모든 종목 실행'이 엑셀 유무에 막히지 않게 하기 위해서입니다.
- 종목이 하나 끝날 때마다 스냅샷을 저장(checkpoint)합니다.
  중간에 죽어도 직전 종목까지 남습니다.
- 차트는 PNG로 저장합니다. GIF(256색)는 한글·선이 뭉개져
  스크린샷 용도에 부적합합니다.
- 종료코드: 1종목이라도 성공하면 0. 전부 실패했을 때만 1.
  (한 종목 실패로 매일 빨간불이 되는 것을 막습니다)
"""

import logging
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

import argparse
import sys

from analysis_engine import analyze_frame, recent_signals
from charts import make_chart_png
from data_collector import fetch_data
import notifier
import per_status
import result_store


BASE_DIR = Path(__file__).resolve().parent
STOCKS_FILE = BASE_DIR / "stocks.txt"
FINANCIAL_DIR = BASE_DIR / "file"
IMAGE_DIR = BASE_DIR / "img"
OUTPUT_DIR = BASE_DIR / "output"

KST = ZoneInfo("Asia/Seoul")

# app.py와 반드시 같아야 하는 값 (스냅샷 검증에 사용됩니다)
RECENT_SIGNAL_DAYS = 7
RECENT_SIGNAL_LIMIT = 3

ANALYSIS_YEARS = 5

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


def write_bytes_atomic(path, data):
    """같은 폴더의 임시 파일에 쓴 뒤 교체합니다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.tmp"

    try:
        temporary.write_bytes(data)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def build_latest_with_per(data, latest, excel_path, stock_name, messages):
    """latest에 PER 정보를 더합니다(엑셀이 없으면 사유만 기록)."""
    if excel_path is None or not excel_path.is_file():
        latest.update(per_status.empty_per_status())
        latest["PER 설명"] = (
            f"file/{stock_name}.xlsx 가 없어 PER을 계산하지 않았습니다."
        )
        return latest

    try:
        financial = per_status.read_per_financial(
            str(excel_path),
            stock_name,
        )
        reference = pd.Timestamp(latest["기준일"]).normalize()

        latest.update(
            per_status.build_per_status(
                financial,
                reference,
                float(latest["종가"]),
            )
        )

    except Exception as error:
        latest.update(per_status.empty_per_status())
        latest["PER 설명"] = f"PER 계산 실패: {error}"
        messages.append(f"PER 계산 실패: {error}")

    return latest


def analyze_one(name, ticker, start, end):
    """종목 1개를 분석해 앱 스냅샷용 결과 dict를 반환합니다."""
    excel_path = FINANCIAL_DIR / f"{name}.xlsx"
    messages = []

    if not excel_path.is_file():
        messages.append(
            f"file/{name}.xlsx 가 없어 목표가·괴리율·PER 없이 분석했습니다."
        )
        excel_path = None

    frame, fetch_messages = fetch_data(
        ticker,
        start.strftime("%Y%m%d"),
        end.strftime("%Y%m%d"),
        include_investors=True,
    )

    messages.extend(fetch_messages or [])

    analysis = analyze_frame(
        frame,
        name,
        excel_file=str(excel_path) if excel_path else None,
        recent_signal_days=RECENT_SIGNAL_DAYS,
    )

    messages.extend(analysis.get("messages") or [])

    latest = build_latest_with_per(
        analysis["data"],
        dict(analysis["latest"]),
        excel_path,
        name,
        messages,
    )

    # 앱과 같은 규칙: 전체 기간에서 최근 N건을 '최신순'으로
    weekly = (
        recent_signals(
            analysis["data"],
            combined=False,
            limit=RECENT_SIGNAL_LIMIT,
        )
        .iloc[::-1]
        .reset_index(drop=True)
    )

    combined = (
        recent_signals(
            analysis["data"],
            combined=True,
            limit=RECENT_SIGNAL_LIMIT,
        )
        .iloc[::-1]
        .reset_index(drop=True)
    )

    return {
        "stock_name": name,
        "ticker": ticker,
        "latest": latest,
        "messages": messages,
        "png_bytes": None,
        "image_filename": None,
        "weekly_signals": weekly,
        "combined_signals": combined,
        "metrics": analysis["metrics"],
        "trades": analysis["trades"],
        "data": analysis["data"],
        "excel_path": excel_path,
    }


def main(notify=True):
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    today = datetime.now(KST).date()

    # 저녁 8시 실행: 당일 종가까지 조회합니다.
    # 휴장일이면 데이터 공급자가 준 최근 거래일까지 분석합니다.
    start = today - timedelta(days=365 * ANALYSIS_YEARS)
    date_text = today.strftime("%Y%m%d")

    stocks = load_all_stocks()
    records = []
    results = []

    completed = 0
    failed = 0
    skipped = 0

    logger.info(
        "분석 시작: %s 종목 / 기간 %s ~ %s",
        len(stocks),
        start,
        today,
    )

    for index, (name, ticker) in enumerate(stocks, start=1):
        logger.info("[%s/%s] %s (%s)", index, len(stocks), name, ticker)

        try:
            item = analyze_one(name, ticker, start, today)

            png_bytes = make_chart_png(
                item["data"],
                name,
                excel_file=(
                    str(item["excel_path"])
                    if item["excel_path"]
                    else None
                ),
            )

            chart_name = f"{name}_{date_text}.png"
            write_bytes_atomic(IMAGE_DIR / chart_name, png_bytes)

            item["png_bytes"] = png_bytes
            item["image_filename"] = chart_name

            latest = item["latest"]

            pd.DataFrame([latest]).to_json(
                OUTPUT_DIR / f"{name}_{date_text}_latest.json",
                orient="records",
                force_ascii=False,
                indent=2,
            )

            item["metrics"].to_csv(
                OUTPUT_DIR / f"{name}_{date_text}_metrics.csv",
                index=False,
                encoding="utf-8-sig",
            )

            for strategy, trades in item["trades"].items():
                trades.to_csv(
                    OUTPUT_DIR / f"{name}_{date_text}_{strategy}_trades.csv",
                    index=False,
                    encoding="utf-8-sig",
                )

            records.append({
                "종목명": name,
                "종목코드": ticker,
                "상태": "완료",
                "기준일": latest.get("기준일"),
                "신호": latest.get("Weekly+Monthly 신호"),
                "신호일": latest.get("Weekly+Monthly 신호일"),
                "종가": latest.get("종가"),
                "PER": latest.get("PER"),
                "차트": f"img/{chart_name}",
                "경고": " / ".join(item["messages"]),
            })

            # 앱 스냅샷에는 무거운 원본 프레임을 넣지 않습니다.
            item.pop("data", None)
            item.pop("excel_path", None)
            results.append(item)

            completed += 1
            logger.info("%s: 완료 — %s", name, chart_name)

        except Exception as error:
            failed += 1
            logger.exception("%s: 분석 실패", name)

            records.append({
                "종목명": name,
                "종목코드": ticker,
                "상태": "실패",
                "경고": f"{type(error).__name__}: {error}",
            })

        finally:
            # 공급자 호출 제한을 피하기 위한 최소 간격입니다.
            time.sleep(1.5)

        # -------------------------------------------------
        # 종목마다 체크포인트 저장
        # -------------------------------------------------
        if results:
            try:
                result_store.LocalStore(
                    OUTPUT_DIR / "last_analysis.json.gz"
                ).save(
                    result_store.build_snapshot(
                        results=results,
                        records=records,
                        period=(
                            start.isoformat(),
                            today.isoformat(),
                        ),
                        saved_at=datetime.now(KST).isoformat(
                            timespec="seconds"
                        ),
                        complete=False,
                        source="batch",
                        meta={
                            "recent_signal_days": RECENT_SIGNAL_DAYS,
                            "recent_signal_limit": RECENT_SIGNAL_LIMIT,
                        },
                    )
                )
            except Exception:
                logger.exception("중간 저장 실패")

    pd.DataFrame(records).to_csv(
        OUTPUT_DIR / f"run_status_{date_text}.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # -----------------------------------------------------
    # 최종 스냅샷 (앱이 재접속 시 자동 복원)
    # -----------------------------------------------------
    if results:
        saved_at = datetime.now(KST).isoformat(timespec="seconds")

        payload = result_store.build_snapshot(
            results=results,
            records=records,
            period=(start.isoformat(), today.isoformat()),
            saved_at=saved_at,
            complete=True,
            source="batch",
            meta={
                "recent_signal_days": RECENT_SIGNAL_DAYS,
                "recent_signal_limit": RECENT_SIGNAL_LIMIT,
            },
        )

        size = result_store.LocalStore(
            OUTPUT_DIR / "last_analysis.json.gz"
        ).save(payload)

        logger.info(
            "스냅샷 저장: output/last_analysis.json.gz (%.0f KB)",
            size / 1024,
        )

    logger.info(
        "실행 완료: 성공 %s / 실패 %s / 건너뜀 %s / 전체 %s",
        completed,
        failed,
        skipped,
        len(stocks),
    )

    # -----------------------------------------------------
    # 텔레그램 알림 (실패해도 배치 결과에는 영향 없음)
    # -----------------------------------------------------
    if notify:
        send_notification(records, today)

    if not completed:
        logger.error("분석에 성공한 종목이 없습니다.")
        return 1

    return 0


def send_notification(records, today):
    """텔레그램 요약을 보냅니다. 예외는 여기서 흡수합니다."""
    if notifier.notification_disabled(os.environ):
        logger.info("텔레그램 알림이 비활성화되어 있습니다(TELEGRAM_DISABLE).")
        return

    config = notifier.config_from_env(os.environ)

    if config.channel:
        logger.info("알림 대상: 채널 %s", config.channel)

    if not config.configured:
        logger.warning(
            "텔레그램 설정이 없어 알림을 건너뜁니다"
            "(TELEGRAM_BOT_TOKEN + TELEGRAM_CHANNEL 또는 TELEGRAM_CHAT_ID)."
        )
        return

    try:
        outcome = notifier.notify_daily(
            records,
            run_date=f"{today:%Y-%m-%d}",
            base_dir=BASE_DIR,
            config=config,
        )
    except Exception:
        logger.exception("텔레그램 알림 중 예외")
        return

    if outcome["error"]:
        logger.error("텔레그램 알림 실패: %s", outcome["error"])
    elif outcome["sent"]:
        logger.info(
            "텔레그램 알림 전송: %s / 메시지 %s개 / 차트 %s장",
            outcome.get("target") or config.describe_target(),
            outcome["messages"],
            outcome["photos"],
        )


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description="전 종목 분석 배치",
    )
    parser.add_argument(
        "--no-notify",
        action="store_true",
        help="텔레그램 알림을 보내지 않습니다.",
    )
    parser.add_argument(
        "--notify-test",
        action="store_true",
        help="분석 없이 텔레그램 연결만 확인하고 종료합니다.",
    )
    return parser.parse_args(argv)


def run_notify_test():
    """설정 확인용: 짧은 테스트 메시지를 보냅니다."""
    config = notifier.config_from_env(os.environ)

    if not config.token:
        logger.error("TELEGRAM_BOT_TOKEN 을 설정하세요.")
        return 1

    # 채널이 지정되면 게시 권한까지 검증합니다.
    if config.channel:
        notifier_instance = notifier.TelegramNotifier(config)

        try:
            info = notifier_instance.verify_target()

        except Exception as error:
            logger.error("채널 검증 실패: %s", error)
            return 1

        print("채널 확인:")
        print(f"  이름   {info['title']}")
        print(f"  주소   @{info['username']}" if info["username"] else
              f"  주소   {info['chat_id']}")
        print(f"  유형   {info['type']}")
        print(f"  봇     @{info['bot_username']} / 상태 {info['status']}")

        for warning in info["warnings"]:
            print(f"  주의   {warning}")

        try:
            notifier_instance.send_message(
                "✅ <b>채널 연결 확인</b>\n"
                f"<b>{info['title']}</b> 채널로 매일 분석 결과를 보냅니다."
            )
        except Exception as error:
            logger.error("전송 실패: %s", error)
            return 1

        logger.info("테스트 메시지를 채널에 게시했습니다: %s", info["title"])
        return 0

    # chat_id가 없으면 최근 메시지에서 찾아 안내합니다.
    if not config.chat_id:
        logger.info("TELEGRAM_CHAT_ID 가 없어 자동으로 찾습니다.")

        try:
            found = notifier.TelegramNotifier(config).find_chat_ids()

        except Exception as error:
            logger.error("chat_id 조회 실패: %s", error)
            return 1

        if not found:
            logger.error(
                "chat_id를 찾지 못했습니다. 텔레그램에서 봇에게 "
                "메시지를 한 번 보낸 뒤 다시 실행하세요."
            )
            return 1

        print("찾은 chat_id 후보:")

        for chat_id, label in found:
            print(f"  {chat_id}  {label}")

        config.chat_id = found[0][0]
        logger.info("첫 번째 후보(%s)로 테스트합니다.", config.chat_id)

    try:
        answer = notifier.TelegramNotifier(config).send_message(
            "✅ <b>텔레그램 연결 확인</b>\n"
            "이 메시지가 보이면 배치 알림 설정이 정상입니다."
        )
    except Exception as error:
        logger.error("전송 실패: %s", error)
        return 1

    logger.info("테스트 메시지 전송 완료: %s", answer)
    return 0


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])

    if args.notify_test:
        raise SystemExit(run_notify_test())

    raise SystemExit(main(notify=not args.no_notify))
