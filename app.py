# app.py
import io
import os
import uuid
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st
from PIL import Image

from analysis_engine import analyze_frame, recent_signals
from charts import make_chart_png
from data_collector import fetch_data


# =========================================================
# 기본 설정
# =========================================================

st.set_page_config(
    page_title="MyTrader — 그룹별 종목 분석",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
STOCKS_FILE = BASE_DIR / "stocks.txt"
FINANCIAL_DIR = BASE_DIR / "file"
IMAGE_DIR = BASE_DIR / "img"

TODAY_KST = datetime.now(ZoneInfo("Asia/Seoul")).date()
YESTERDAY_KST = TODAY_KST - timedelta(days=1)


# =========================================================
# TXT 종목 목록 읽기
# =========================================================

def load_stock_groups(path):
    """
    형식:
        [그룹명]
        종목명,6자리종목코드
    """
    if not path.is_file():
        raise FileNotFoundError(
            f"종목 목록 파일이 없습니다: {path.name}"
        )

    groups = {}
    current_group = None
    registered_codes = {}

    text = path.read_text(encoding="utf-8-sig")

    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()

        if not line or line.startswith("#"):
            continue

        if line.startswith("[") and line.endswith("]"):
            current_group = line[1:-1].strip()

            if not current_group:
                raise ValueError(
                    f"stocks.txt {line_number}행: 그룹명이 비어 있습니다."
                )

            groups.setdefault(current_group, [])
            continue

        if current_group is None:
            raise ValueError(
                f"stocks.txt {line_number}행: "
                "종목 위에 [그룹명]을 먼저 작성하세요."
            )

        fields = [value.strip() for value in line.split(",")]

        if len(fields) != 2:
            raise ValueError(
                f"stocks.txt {line_number}행: "
                "'종목명,종목코드' 형식이어야 합니다."
            )

        stock_name, ticker = fields

        # 파일명으로 사용할 수 없는 문자와 경로 이동을 제한합니다.
        invalid_chars = '<>:"/\\|?*'

        if (
            not stock_name
            or stock_name in {".", ".."}
            or any(character in stock_name for character in invalid_chars)
            or any(ord(character) < 32 for character in stock_name)
            or stock_name.endswith((".", " "))
        ):
            raise ValueError(
                f"stocks.txt {line_number}행: 사용할 수 없는 종목명입니다."
            )

        if not (
            ticker.isascii()
            and ticker.isdigit()
            and len(ticker) == 6
        ):
            raise ValueError(
                f"stocks.txt {line_number}행: "
                "종목코드는 숫자 6자리여야 합니다."
            )

        previous_code = registered_codes.get(stock_name)

        if previous_code is not None and previous_code != ticker:
            raise ValueError(
                f"stocks.txt {line_number}행: "
                f"'{stock_name}'에 서로 다른 종목코드가 등록되었습니다."
            )

        registered_codes[stock_name] = ticker
        item = (stock_name, ticker)

        if item not in groups[current_group]:
            groups[current_group].append(item)

    groups = {
        name: items
        for name, items in groups.items()
        if items
    }

    if not groups:
        raise ValueError("stocks.txt에 등록된 종목이 없습니다.")

    return groups


# =========================================================
# 데이터 수집 캐시
# =========================================================

@st.cache_data(ttl=3600, show_spinner=False, max_entries=100)
def cached_fetch(ticker, start_date, end_date, include_investors):
    return fetch_data(
        ticker,
        start_date,
        end_date,
        include_investors,
    )


# =========================================================
# GIF 변환 및 저장
# =========================================================

def save_chart_gif(png_bytes, stock_name, analysis_date):
    """
    기존 PNG 차트를 실제 GIF 형식으로 변환합니다.
    동일 종목·동일 날짜의 파일은 덮어씁니다.
    """
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    filename = f"{stock_name}_{analysis_date}.gif"
    target_path = IMAGE_DIR / filename

    gif_buffer = io.BytesIO()

    with Image.open(io.BytesIO(png_bytes)) as image:
        rgb_image = image.convert("RGB")
        try:
            rgb_image.save(gif_buffer, format="GIF")
        finally:
            rgb_image.close()

    gif_bytes = gif_buffer.getvalue()

    # 저장 도중 다른 요청이 불완전한 파일을 읽지 않도록
    # 임시 파일 작성 후 교체합니다.
    temporary_path = IMAGE_DIR / f".{uuid.uuid4().hex}.tmp"

    try:
        temporary_path.write_bytes(gif_bytes)
        os.replace(temporary_path, target_path)
    finally:
        temporary_path.unlink(missing_ok=True)

    return target_path, gif_bytes


# =========================================================
# 종목별 순차 분석
# =========================================================

def run_batch(selected_stocks, start, end, include_investors):
    results = []
    records = []

    # 실행을 시작할 때 이전 화면 결과를 초기화합니다.
    st.session_state["batch_results"] = results
    st.session_state["batch_records"] = records
    st.session_state["batch_period"] = (
        start.isoformat(),
        end.isoformat(),
    )

    analysis_date = datetime.now(
        ZoneInfo("Asia/Seoul")
    ).strftime("%Y%m%d")

    progress = st.progress(0)
    status = st.empty()

    total = len(selected_stocks)

    for index, (stock_name, ticker) in enumerate(
        selected_stocks, start=1
    ):
        status.info(
            f"[{index}/{total}] {stock_name} ({ticker}) 분석 중..."
        )

        excel_path = FINANCIAL_DIR / f"{stock_name}.xlsx"

        # 요청에 따라 엑셀이 없으면 해당 종목 전체를 건너뜁니다.
        if not excel_path.is_file():
            records.append({
                "종목명": stock_name,
                "종목코드": ticker,
                "상태": "건너뜀",
                "내용": f"file/{stock_name}.xlsx 없음",
            })

            progress.progress(index / total)
            continue

        try:
            frame, fetch_messages = cached_fetch(
                ticker,
                start.strftime("%Y%m%d"),
                end.strftime("%Y%m%d"),
                include_investors,
            )

            result = analyze_frame(
                frame,
                stock_name,
                excel_file=str(excel_path),
            )

            png_bytes = make_chart_png(
                result["data"],
                stock_name,
                excel_file=str(excel_path),
            )

            image_path, gif_bytes = save_chart_gif(
                png_bytes,
                stock_name,
                analysis_date,
            )

            # 전체 일봉 데이터 대신 표시용 결과만 세션에 보관합니다.
            results.append({
                "stock_name": stock_name,
                "ticker": ticker,
                "latest": result["latest"],
                "metrics": result["metrics"],
                "trades": result["trades"],
                "weekly_signals": recent_signals(
                    result["data"],
                    combined=False,
                ),
                "combined_signals": recent_signals(
                    result["data"],
                    combined=True,
                ),
                "messages": fetch_messages + result["messages"],
                "gif_bytes": gif_bytes,
                "image_filename": image_path.name,
            })

            records.append({
                "종목명": stock_name,
                "종목코드": ticker,
                "상태": "완료",
                "내용": f"img/{image_path.name}",
            })

        except Exception as error:
            # 한 종목의 실패가 전체 분석을 중단하지 않게 합니다.
            records.append({
                "종목명": stock_name,
                "종목코드": ticker,
                "상태": "실패",
                "내용": str(error),
            })

        progress.progress(index / total)

    st.session_state["batch_results"] = results
    st.session_state["batch_records"] = records

    completed = len(results)
    skipped = sum(
        record["상태"] == "건너뜀" for record in records
    )
    failed = sum(
        record["상태"] == "실패" for record in records
    )

    status.success(
        f"분석 종료 — 완료 {completed}개 / "
        f"건너뜀 {skipped}개 / 실패 {failed}개"
    )


# =========================================================
# 종목 목록 로드
# =========================================================

try:
    stock_groups = load_stock_groups(STOCKS_FILE)
except Exception as error:
    st.error(str(error))
    st.info(
        "프로젝트 폴더에 stocks.txt를 만들고 "
        "그룹과 종목을 등록하세요."
    )
    st.stop()


# =========================================================
# 왼쪽: 그룹·종목·기간 선택
# =========================================================

with st.sidebar:
    st.header("📋 등록 종목")

    group_names = list(stock_groups)

    selected_groups = st.multiselect(
        "분석할 그룹",
        options=group_names,
        default=group_names[:1],
    )

    available_stocks = []
    seen = set()

    for group_name in selected_groups:
        for item in stock_groups[group_name]:
            if item not in seen:
                seen.add(item)
                available_stocks.append(item)

    st.caption(
        f"전체 {len(stock_groups)}개 그룹 · "
        f"선택 그룹 내 {len(available_stocks)}개 종목"
    )

    if available_stocks:
        registered_table = pd.DataFrame(
            available_stocks,
            columns=["종목명", "종목코드"],
        )

        st.dataframe(
            registered_table,
            hide_index=True,
            use_container_width=True,
        )
    else:
        st.info("그룹을 하나 이상 선택하세요.")

    with st.form("batch_analysis_form"):
        selected_stocks = st.multiselect(
            "실행할 종목 — 복수 선택",
            options=available_stocks,
            default=available_stocks,
            format_func=lambda item: f"{item[0]} ({item[1]})",
        )

        start = st.date_input(
            "시작일",
            value=YESTERDAY_KST - timedelta(days=365 * 5),
            max_value=YESTERDAY_KST,
        )

        end = st.date_input(
            "종료일",
            value=YESTERDAY_KST,
            max_value=YESTERDAY_KST,
        )

        include_investors = st.checkbox(
            "투자자 매매 데이터 포함",
            value=True,
        )

        submitted = st.form_submit_button(
            "분석 실행",
            use_container_width=True,
        )

    st.caption(
        "종목명.xlsx 파일이 없는 종목은 건너뜁니다. "
        "종료일은 어제까지 선택할 수 있으며 "
        "장중 네이버 시세는 사용하지 않습니다."
    )

    st.divider()

    if st.button("가격 데이터 캐시 초기화"):
        cached_fetch.clear()
        st.success("캐시를 초기화했습니다.")

    if st.button("화면 결과 초기화"):
        st.session_state.pop("batch_results", None)
        st.session_state.pop("batch_records", None)
        st.session_state.pop("batch_period", None)

    st.caption(
        "화면 결과를 초기화해도 img 폴더의 파일은 삭제하지 않습니다."
    )


# =========================================================
# 오른쪽: 실행 및 결과 표시
# =========================================================

st.title("📊 MyTrader — 종목별 종합차트")

st.warning(
    "기존 신호 계산 로직을 유지한 웹 버전입니다. "
    "주간·월간 지표의 미래 데이터 참조 위험이 있으므로 "
    "백테스트는 검증 전 참고용으로 사용하세요."
)

if submitted:
    if not selected_stocks:
        st.error("실행할 종목을 하나 이상 선택하세요.")
    elif start > end:
        st.error("시작일은 종료일보다 늦을 수 없습니다.")
    else:
        run_batch(
            selected_stocks,
            start,
            end,
            include_investors,
        )

records = st.session_state.get("batch_records", [])
results = st.session_state.get("batch_results", [])
period = st.session_state.get("batch_period")

if records:
    record_frame = pd.DataFrame(records)

    with st.expander("실행 내역 — 완료 / 건너뜀 / 실패", expanded=True):
        st.dataframe(
            record_frame,
            hide_index=True,
            use_container_width=True,
        )

if not results:
    st.info(
        "왼쪽에서 그룹·종목·기간을 선택하고 "
        "'분석 실행'을 누르세요. "
        "분석된 차트가 여기에 세로로 표시됩니다."
    )
else:
    if period:
        st.caption(
            f"현재 표시 결과의 요청 기간: {period[0]} ~ {period[1]} · "
            f"분석 완료 {len(results)}개 종목"
        )

    for position, result in enumerate(results, start=1):
        name = result["stock_name"]
        ticker = result["ticker"]
        latest = result["latest"]

        st.subheader(f"{position}. {name} ({ticker})")

        st.caption(
            f'데이터 기준일: {latest["기준일"]} · '
            f'차트 파일: img/{result["image_filename"]}'
        )

        price_column, weekly_column, combined_column = st.columns(3)

        price_column.metric(
            "최근 일봉 종가",
            f'{latest["종가"]:,.0f}원',
        )

        weekly_column.metric(
            "최근 일봉 Weekly 신호",
            latest["Weekly 신호"],
        )

        combined_column.metric(
            "최근 일봉 Weekly+Monthly 신호",
            latest["Weekly+Monthly 신호"],
        )

        for message in result["messages"]:
            st.info(message)

        # 종목별 차트를 세로로 표시합니다.
        st.image(
            result["gif_bytes"],
            use_container_width=True,
        )

        st.download_button(
            f"{name} GIF 다운로드",
            data=result["gif_bytes"],
            file_name=result["image_filename"],
            mime="image/gif",
            key=f"gif_download_{position}_{ticker}",
        )

        with st.expander(f"{name} — 최근 신호 / 거래 기록 / 성과"):
            st.markdown("#### Weekly 단독 신호")
            st.dataframe(
                result["weekly_signals"],
                hide_index=True,
            )

            st.markdown("#### Weekly+Monthly 결합 신호")
            st.dataframe(
                result["combined_signals"],
                hide_index=True,
            )

            st.markdown("#### 전략별 성과 — 원본 계산 방식")
            st.dataframe(
                result["metrics"],
                hide_index=True,
            )

            st.caption(
                "총수익률은 거래별 수익률의 단순 합계입니다. "
                "수수료·세금·슬리피지 및 미청산 평가손익은 "
                "반영되지 않았습니다. 손익비도 원본 계산식입니다."
            )

            for strategy, trades in result["trades"].items():
                st.markdown(f"#### {strategy} 거래 기록")

                if trades.empty:
                    st.info("청산 완료된 거래가 없습니다.")
                else:
                    st.dataframe(
                        trades,
                        hide_index=True,
                    )

        st.divider()
