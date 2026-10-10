"""Windows 개인용 MyTrader 웹 MVP."""
import tempfile
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import streamlit as st

from analysis_engine import analyze_frame, recent_signals
from charts import make_chart_png
from data_collector import fetch_data

st.set_page_config(
    page_title="MyTrader",
    page_icon="📊",
    layout="wide",
)

@st.cache_data(ttl=3600, show_spinner=False)
def cached_fetch(ticker, start_date, end_date, include_investors):
    return fetch_data(
        ticker, start_date, end_date, include_investors
    )

def csv_bytes(frame, index=False):
    return frame.to_csv(index=index).encode("utf-8-sig")

st.title("📊 MyTrader — 종목 분석")
st.caption("기존 Python 분석 코드를 재사용한 개인용 웹 MVP")

st.warning(
    "기존 주간·월간 지표 계산에는 과거 행이 이후 가격을 반영하는 "
    "미래 데이터 참조 위험이 있습니다. 백테스트는 검증 전 참고용입니다. "
    "수수료·세금·슬리피지가 반영되지 않았으며, "
    "총수익률(%)은 거래별 수익률의 단순 합계입니다."
)

yesterday = date.today() - timedelta(days=1)

with st.sidebar:
    st.header("분석 설정")
    st.caption("첫 버전은 장중 데이터를 사용하지 않습니다.")

    with st.form("analysis_form"):
        stock_name = st.text_input("종목명 / 엑셀 시트명", "삼성전자")
        ticker = st.text_input("6자리 종목코드", "005930")
        start = st.date_input(
            "시작일",
            value=yesterday - timedelta(days=365 * 5),
            max_value=yesterday,
        )
        end = st.date_input(
            "종료일",
            value=yesterday,
            max_value=yesterday,
        )
        include_investors = st.checkbox(
            "투자자 매매 데이터 수집", value=True
        )
        uploaded = st.file_uploader(
            "재무·목표가 엑셀 (선택)",
            type=["xlsx"],
        )
        submitted = st.form_submit_button(
            "분석 실행", use_container_width=True
        )

    st.caption(
        "엑셀 시트명은 입력 종목명과 같아야 합니다. "
        "필수 컬럼: 분기, 목표가, 영업이익"
    )

    if st.button("가격 데이터 캐시 초기화"):
        cached_fetch.clear()
        st.success("캐시를 초기화했습니다.")

if submitted:
    ticker = ticker.strip()
    stock_name = stock_name.strip()

    if not stock_name:
        st.error("종목명을 입력하세요.")
    elif not (ticker.isascii() and ticker.isdigit() and len(ticker) == 6):
        st.error("종목코드는 숫자 6자리로 입력하세요.")
    elif start > end:
        st.error("시작일은 종료일보다 늦을 수 없습니다.")
    else:
        try:
            with st.spinner("데이터 수집 및 분석 중입니다..."):
                frame, fetch_messages = cached_fetch(
                    ticker,
                    start.strftime("%Y%m%d"),
                    end.strftime("%Y%m%d"),
                    include_investors,
                )

                # 사용자 업로드를 공용 파일명으로 저장하지 않습니다.
                # 분석이 끝나면 임시 디렉터리는 자동 삭제됩니다.
                with tempfile.TemporaryDirectory(prefix="mytrader_") as folder:
                    excel_file = None
                    if uploaded is not None:
                        file_path = Path(folder) / "financial.xlsx"
                        file_path.write_bytes(uploaded.getvalue())
                        excel_file = str(file_path)

                    result = analyze_frame(
                        frame, stock_name, excel_file=excel_file
                    )
                    result["messages"] = (
                        fetch_messages + result["messages"]
                    )

                    try:
                        result["chart"] = make_chart_png(
                            result["data"], stock_name, excel_file
                        )
                    except Exception as chart_error:
                        result["chart"] = None
                        result["messages"].append(
                            f"차트 생성 실패: {chart_error}"
                        )

                result["ticker"] = ticker
                result["requested_period"] = (
                    start.isoformat(), end.isoformat()
                )
                st.session_state["result"] = result

        except Exception as error:
            st.error(f"분석 실패: {error}")
            st.info(
                "이전 분석 결과가 있다면 아래에 그대로 유지됩니다. "
                "표시된 종목과 기준일을 확인하세요."
            )

result = st.session_state.get("result")

if result is None:
    st.info("왼쪽에서 종목과 기간을 설정하고 '분석 실행'을 누르세요.")
    st.stop()

latest = result["latest"]
st.subheader(f'{latest["종목명"]} ({result["ticker"]})')
st.caption(
    f'데이터 기준일: {latest["기준일"]} · '
    f'요청 기간: {result["requested_period"][0]} ~ '
    f'{result["requested_period"][1]}'
)

col1, col2, col3 = st.columns(3)
col1.metric("최근 일봉 종가", f'{latest["종가"]:,.0f}원')
col2.metric("최근 일봉 Weekly 신호", latest["Weekly 신호"])
col3.metric(
    "최근 일봉 Weekly+Monthly 신호",
    latest["Weekly+Monthly 신호"],
)

for message in result["messages"]:
    st.info(message)

tab_chart, tab_signals, tab_trades, tab_data = st.tabs(
    ["종합 차트", "최근 신호", "거래·성과", "원본 데이터"]
)

with tab_chart:
    if result.get("chart"):
        st.image(result["chart"])
        st.download_button(
            "차트 PNG 다운로드",
            data=result["chart"],
            file_name=f'{result["ticker"]}_chart.png',
            mime="image/png",
        )
    else:
        st.info("차트가 생성되지 않았습니다.")

with tab_signals:
    st.markdown("#### Weekly 단독")
    st.dataframe(
        recent_signals(result["data"]),
        hide_index=True,
    )
    st.markdown("#### Weekly+Monthly 결합")
    st.dataframe(
        recent_signals(result["data"], combined=True),
        hide_index=True,
    )

with tab_trades:
    st.markdown("#### 성과 요약 — 원본 계산 방식")
    st.dataframe(result["metrics"], hide_index=True)
    st.caption(
        "현재 '손익비'도 원본 계산식입니다. "
        "일반적인 profit factor와 같은 지표로 해석하지 마세요. "
        "미청산 포지션의 평가손익은 거래 기록에 포함되지 않습니다."
    )
    st.download_button(
        "성과 요약 CSV 다운로드",
        data=csv_bytes(result["metrics"]),
        file_name=f'{result["ticker"]}_metrics.csv',
        mime="text/csv",
        key="metrics_download",
    )

    for strategy, trades in result["trades"].items():
        st.markdown(f"#### {strategy}")
        if trades.empty:
            st.info("청산 완료된 거래가 없습니다.")
        else:
            st.dataframe(trades, hide_index=True)
            st.download_button(
                f"{strategy} 거래 CSV 다운로드",
                data=csv_bytes(trades),
                file_name=f'{result["ticker"]}_{strategy}_trades.csv',
                mime="text/csv",
                key=f"trades_{strategy}",
            )

with tab_data:
    st.dataframe(result["data"])
    st.download_button(
        "전체 분석 데이터 CSV 다운로드",
        data=csv_bytes(result["data"], index=True),
        file_name=f'{result["ticker"]}_analysis.csv',
        mime="text/csv",
        key="data_download",
    )
