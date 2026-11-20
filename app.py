# app.py

import base64
import io
import json
import math
import os
import re
import uuid

from datetime import date, datetime, timedelta
from html import escape
from numbers import Integral, Real
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st

from streamlit.errors import StreamlitSecretNotFoundError


# =========================================================
# 페이지 설정
# =========================================================

st.set_page_config(
    page_title="종목별 분석",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# KRX 인증정보
# data_collector import 전에 설정합니다.
# =========================================================

for credential_key in ("KRX_ID", "KRX_PW"):
    try:
        credential_value = st.secrets.get(credential_key)
    except (StreamlitSecretNotFoundError, KeyError):
        credential_value = None

    if credential_value:
        os.environ[credential_key] = str(credential_value)

# 인증정보를 화면이나 로그에 출력하지 않습니다.
del credential_key
del credential_value


from analysis_engine import analyze_frame, recent_signals
from charts import make_chart_png
from data_collector import fetch_data

import per_status
import result_store
import stock_groups as stock_data
import schedule_config


# =========================================================
# 전역 UI 스타일
# =========================================================

st.markdown(
    """
    <style>
    /* ================================================
       MyTrader 디자인 시스템
       컬러: 잉크 #0f172a / 뮤트 #64748b / 액센트 #4f46e5
       ================================================ */

    @import url('https://cdn.jsdelivr.net/gh/orioncactus/pretendard@v1.3.9/dist/web/variable/pretendardvariable-dynamic-subset.min.css');

    :root {
        --ink:#0f172a; --muted:#64748b; --faint:#94a3b8;
        --accent:#4f46e5; --accent-2:#6366f1;
        --line:#e8ecf3; --line-soft:#eef2f7;
        --surface:#ffffff; --surface-soft:#f8fafc;
        --shadow-xs:0 1px 2px rgba(15,23,42,.05);
        --shadow-sm:0 2px 6px rgba(15,23,42,.07);
        --shadow-md:0 8px 24px rgba(15,23,42,.10);
        --radius:12px; --radius-lg:16px;
    }

    /* ---------- 타이포그래피 & 배경 ---------- */
    html, body, .stApp, [data-testid="stSidebar"], .stApp button {
        font-family:'Pretendard Variable','Pretendard',
            'Noto Sans KR','Apple SD Gothic Neo','Malgun Gothic',
            -apple-system,'Segoe UI',sans-serif;
        letter-spacing:-0.014em;
    }
    .stApp { background:#fbfcfe; color:var(--ink); }
    .stApp p, .stApp li { color:#334155; }

    /* ---------- 메인 레이아웃 ---------- */
    .block-container {
        padding-top:2.6rem; padding-bottom:3.5rem; max-width:1380px;
    }

    /* ---------- 문서 제목 ---------- */
    .stApp h1 {
        font-weight:800; letter-spacing:-0.035em; color:var(--ink);
        font-size:1.72rem;
        padding-bottom:.85rem; margin-bottom:1.35rem;
        border-bottom:1px solid var(--line);
    }

    /* ---------- 섹션 헤더 (종목 상세 등) ---------- */
    .stApp h2 {
        position:relative; padding-left:16px;
        font-weight:800; letter-spacing:-0.025em; color:var(--ink);
        scroll-margin-top:72px;
    }
    .stApp h2::before {
        content:''; position:absolute; left:0; top:6%;
        height:88%; width:5px; border-radius:5px;
        background:linear-gradient(180deg,var(--accent),#8b5cf6);
    }
    .stApp h3 {
        font-weight:800; letter-spacing:-0.018em; color:var(--ink);
        font-size:1.05rem; margin-top:.5rem;
    }

    /* ---------- 사이드바 ---------- */
    section[data-testid="stSidebar"] {
        background:var(--surface-soft);
        border-right:1px solid var(--line);
        padding-top:1.6rem;
    }
    section[data-testid="stSidebar"] h2::before { display:none; }
    section[data-testid="stSidebar"] .stHeader { font-weight:800; }

    /* ---------- 폼 입력 ---------- */
    .stApp [data-baseweb="select"] > div,
    .stApp [data-baseweb="input"] {
        border-radius:10px !important;
        border-color:var(--line) !important;
        box-shadow:var(--shadow-xs) !important;
    }
    .stApp [data-baseweb="select"] > div:hover,
    .stApp [data-baseweb="input"]:hover {
        border-color:var(--accent-2) !important;
    }
    .stApp label { font-weight:600; color:#475569; }

    /* ---------- 버튼 ---------- */
    .stButton > button, .stDownloadButton > button,
    .stFormSubmitButton > button {
        border-radius:10px; font-weight:600; width:100%;
        font-size:.92rem;
        border:1px solid var(--line); background:var(--surface);
        color:#334155; box-shadow:var(--shadow-xs);
        transition:all .17s ease;
    }
    .stButton > button:hover, .stDownloadButton > button:hover {
        border-color:var(--accent); color:var(--accent);
        background:#fdfdff;
        box-shadow:0 4px 12px rgba(79,70,229,.14);
        transform:translateY(-1px);
    }
    .stButton > button:focus-visible,
    .stDownloadButton > button:focus-visible {
        box-shadow:0 0 0 3px rgba(99,102,241,.22) !important;
        border-color:var(--accent) !important;
        outline:none;
    }
    button[kind="primary"] {
        background:linear-gradient(135deg,#4f46e5,#6d28d9);
        color:#ffffff !important; border:none;
        box-shadow:0 3px 10px rgba(79,70,229,.32);
        letter-spacing:-0.01em;
    }

    /* ---------- 분석 실행 버튼: 파스텔 강조 ---------- */
    .stFormSubmitButton > button,
    [data-testid="stFormSubmitButton"] button {
        background:linear-gradient(135deg,#e0e7ff 0%,#fae8ff 52%,#dbeafe 100%) !important;
        color:#3730a3 !important; border:1px solid #c7d2fe !important;
        border-radius:14px !important; font-weight:800 !important;
        font-size:1.02rem !important; letter-spacing:-.01em;
        padding:.92rem 1rem !important;
        box-shadow:0 6px 18px rgba(99,102,241,.20) !important;
        transition:transform .15s ease, box-shadow .15s ease, filter .15s ease;
    }
    .stFormSubmitButton > button:hover,
    [data-testid="stFormSubmitButton"] button:hover {
        filter:brightness(1.02);
        border-color:#a5b4fc !important;
        box-shadow:0 10px 26px rgba(99,102,241,.30) !important;
        transform:translateY(-1px);
    }
    .stFormSubmitButton > button:active,
    [data-testid="stFormSubmitButton"] button:active {
        transform:translateY(0);
    }
    button[kind="primary"]:hover {
        color:#ffffff !important; filter:brightness(1.08);
        box-shadow:0 6px 18px rgba(79,70,229,.4);
        transform:translateY(-1px);
    }

    /* ---------- 지표 카드 ---------- */
    [data-testid="stMetric"] {
        position:relative;
        background:var(--surface); border:1px solid var(--line);
        border-radius:var(--radius-lg); padding:18px 22px 15px;
        box-shadow:var(--shadow-xs);
        overflow:hidden;
        transition:box-shadow .18s ease, transform .18s ease;
    }
    [data-testid="stMetric"]:hover {
        box-shadow:var(--shadow-sm); transform:translateY(-1px);
    }
    [data-testid="stMetric"]::before {
        content:''; position:absolute; left:0; top:0; bottom:0;
        width:4px;
        background:linear-gradient(180deg,var(--accent),#8b5cf6);
    }
    [data-testid="stMetricLabel"] {
        color:var(--muted); font-weight:600; font-size:.87rem;
    }
    [data-testid="stMetricValue"] {
        font-weight:800; letter-spacing:-0.025em; color:var(--ink);
    }

    /* ---------- 최근 종가 · PER · 추세: 같은 행, 같은 높이 ---------- */
    /* 키가 종목마다 달라지므로 접두어로 매칭합니다. */
    [class*="st-key-price_per_row"] {
        --metric-h: 96px;
    }
    [class*="st-key-price_per_row"] [data-testid="stMetric"] {
        min-height:var(--metric-h);
        padding:16px 20px;
        display:flex; flex-direction:column; justify-content:center;
    }
    [class*="st-key-price_per_row"] [data-testid="stMetricValue"] {
        font-size:.88rem; font-weight:700; letter-spacing:-0.01em;
    }

    /* ---------- 페이지 제목 + 실행 시각 ---------- */
    .page-heading h1 {
        display:flex; align-items:center; gap:12px; flex-wrap:wrap;
    }
    .page-heading .heading-time {
        font-size:.8rem; font-weight:700; letter-spacing:-.01em;
        color:var(--muted); background:#eef2ff;
        border:1px solid #e0e7ff; border-radius:999px;
        padding:5px 12px; line-height:1.2; white-space:nowrap;
    }
    .page-heading .heading-time.is-batch { color:#3730a3; background:#eef2ff; }
    .page-heading .heading-time.is-none { color:var(--faint); background:#f8fafc; border-color:var(--line); }

    /* ---------- 최근 종가 카드(등락률 색상 포함) ---------- */
    .price-card {
        position:relative; background:var(--surface);
        border:1px solid var(--line); border-radius:var(--radius-lg);
        padding:16px 20px; box-shadow:var(--shadow-xs);
        overflow:hidden;
        min-height:var(--metric-h, 96px);
        display:flex; flex-direction:column; justify-content:center;
    }
    .price-card::before {
        content:''; position:absolute; left:0; top:0; bottom:0; width:4px;
        background:linear-gradient(180deg,var(--accent),#8b5cf6);
    }
    .price-card .price-label {
        color:var(--muted); font-weight:600; font-size:.87rem;
        margin-bottom:6px;
    }
    .price-row {
        display:flex; align-items:baseline; gap:8px; flex-wrap:wrap;
    }
    .price-value {
        font-size:.88rem; font-weight:700; letter-spacing:-0.01em;
        color:var(--ink);
    }
    .price-delta { font-size:.78rem; font-weight:800; letter-spacing:-.01em; }
    .price-delta.up   { color:#dc2626; }   /* 상승: 빨강 */
    .price-delta.down { color:#2563eb; }   /* 하락: 파랑 */
    .price-delta.flat { color:var(--faint); }

    /* ---------- 일간/주간/월간 추세 신호등 ---------- */
    .trend-card {
        position:relative; background:var(--surface);
        border:1px solid var(--line); border-radius:var(--radius-lg);
        padding:16px 20px; box-shadow:var(--shadow-xs);
        overflow:hidden;
        min-height:var(--metric-h, 96px);
        display:flex; align-items:center; justify-content:center;
    }
    .trend-card::before {
        content:''; position:absolute; left:0; top:0; bottom:0; width:4px;
        background:linear-gradient(180deg,var(--accent),#8b5cf6);
    }
    .trend-lights { display:flex; gap:16px; align-items:flex-start; }
    .trend-light {
        display:flex; flex-direction:column; align-items:center; gap:4px;
        min-width:44px;
    }
    .trend-dot {
        width:18px; height:18px; border-radius:50%; display:inline-block;
        border:2px solid rgba(255,255,255,.9);
        box-shadow:0 0 0 2px var(--line);
    }
    .trend-dot.up   { background:#dc2626; box-shadow:0 0 0 2px #fecaca; }  /* 상승: 빨간 원 */
    .trend-dot.down { background:#2563eb; box-shadow:0 0 0 2px #bfdbfe; }  /* 하락: 파란 원 */
    .trend-dot.flat { background:#e2e8f0; }                                /* 보합 */
    .trend-dot.none { background:#fff; border:2px dashed #cbd5e1; }        /* 값 없음 */
    .trend-name { font-size:.7rem; color:var(--faint); font-weight:600; }


    /* ---------- 펼침 메뉴 ---------- */
    [data-testid="stExpander"] {
        border:1px solid var(--line); border-radius:var(--radius);
        background:var(--surface); overflow:hidden;
        box-shadow:var(--shadow-xs);
    }
    [data-testid="stExpander"] summary {
        font-weight:700; color:var(--ink); font-size:.95rem;
    }
    [data-testid="stExpanderDetails"] { background:transparent; }

    /* ---------- 데이터프레임 ---------- */
    [data-testid="stDataFrame"] {
        border-radius:var(--radius); overflow:hidden;
        border:1px solid var(--line);
        box-shadow:var(--shadow-xs);
        background:var(--surface);
    }

    /* ---------- 알림: 왼쪽 액센트 바 스타일 ---------- */
    [data-testid="stAlert"] {
        border-radius:10px; box-shadow:var(--shadow-xs);
        border:none; overflow:hidden;
    }
    [data-testid="stAlert"] [data-testid="stMarkdownContainer"] {
        font-size:.92rem;
    }

    /* ---------- 구분선 ---------- */
    hr { border-color:var(--line-soft); margin:1.5rem 0; }

    /* ---------- 되돌아가기 링크 ---------- */
    .back-link {
        display:inline-flex; align-items:center; gap:6px;
        padding:5px 15px; margin-bottom:.8rem;
        border:1px solid var(--line); border-radius:999px;
        background:var(--surface); color:#475569;
        font-size:.84rem; font-weight:600; text-decoration:none;
        box-shadow:var(--shadow-xs);
        transition:all .15s ease;
    }
    .back-link:hover {
        border-color:var(--accent); color:var(--accent);
        box-shadow:0 3px 10px rgba(79,70,229,.13);
        text-decoration:none;
        transform:translateY(-1px);
    }

    /* ---------- 신호 카드 ---------- */
    .signal-card {
        background:var(--surface);
        border:1px solid var(--line); border-radius:var(--radius-lg);
        box-shadow:var(--shadow-xs);
        padding:16px 18px;
        display:flex; align-items:center; justify-content:center;
        min-height:88px;
    }
    .signal-card-label {
        font-size:.84rem; font-weight:700; color:var(--muted);
        margin:8px 0 6px 4px; letter-spacing:.01em;
    }

    /* ---------- 개요 테이블 보강 ---------- */
    .stock-overview-table tbody tr { transition:background .12s ease; }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# 기본 설정
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

STOCKS_FILE = BASE_DIR / "stocks.txt"
FINANCIAL_DIR = BASE_DIR / "file"
IMAGE_DIR = BASE_DIR / "img"
OUTPUT_DIR = BASE_DIR / "output"
LAST_RESULT_FILE = OUTPUT_DIR / "last_analysis.json.gz"

TODAY_KST = datetime.now(ZoneInfo("Asia/Seoul")).date()

# 기준일 포함 최근 7일: 거래일이 아닌 달력일입니다.
RECENT_SIGNAL_DAYS = 7

# 최근 매매 신호 표의 최대 행 수입니다.
RECENT_SIGNAL_LIMIT = 3

RESULT_SCHEMA = "png-v9-recent7-per-top3"
SAVED_RESULT_VERSION = result_store.SNAPSHOT_VERSION

# 데이터 수집 코드 변경 전 캐시와 구분합니다.
DATA_CACHE_VERSION = "price-investor-only-v2"

WEEKDAYS = ("월", "화", "수", "목", "금", "토", "일")


def read_result_store_secrets():
    """
    st.secrets의 [result_store] 설정을 읽습니다.

    설정이 없으면 로컬(output/last_analysis.json.gz)을 씁니다.
    Streamlit Cloud에서 재배포에도 결과를 남기려면
    [result_store] backend/url/token을 등록하세요.
    """
    try:
        return dict(st.secrets)

    except Exception:
        return {}


RESULT_STORE = result_store.make_store(
    BASE_DIR,
    read_result_store_secrets(),
)


# =========================================================
# 세션 형식 초기화
# =========================================================

if st.session_state.get("result_schema") != RESULT_SCHEMA:
    for key in (
        "batch_results",
        "batch_records",
        "batch_period",
        "last_saved_at",
        "last_analysis_checked",
        "restore_notice",
    ):
        st.session_state.pop(key, None)

    st.session_state["result_schema"] = RESULT_SCHEMA


# =========================================================
# 등록 종목 읽기
# =========================================================



# =========================================================
# 가격·투자자 데이터 캐시
# =========================================================

@st.cache_data(
    ttl=3600,
    show_spinner=False,
    max_entries=100,
)
def cached_fetch(
    ticker,
    start_date,
    end_date,
    include_investors,
    cache_version,
):
    """
    cache_version:
        데이터 수집 구조 변경 시 이전 캐시와 구분합니다.

    실제 수집 항목은 data_collector.py의 fetch_data가 결정합니다.
    """
    return fetch_data(
        ticker,
        start_date,
        end_date,
        include_investors=include_investors,
    )


# =========================================================
# 날짜·숫자 처리
# =========================================================

def normalize_date(value):
    """한국 날짜 기준의 시간대 없는 Timestamp로 변환합니다."""
    if value is None:
        return pd.NaT

    try:
        timestamp = pd.Timestamp(value)

        if pd.isna(timestamp):
            return pd.NaT

        if timestamp.tzinfo is not None:
            timestamp = (
                timestamp
                .tz_convert("Asia/Seoul")
                .tz_localize(None)
            )

        return timestamp.normalize()

    except (TypeError, ValueError, OverflowError):
        return pd.NaT


def format_short_date(value):
    """mm/dd(요일) 형식으로 표시합니다."""
    timestamp = normalize_date(value)

    if pd.isna(timestamp):
        return ""

    weekday = WEEKDAYS[timestamp.weekday()]
    return f"{timestamp:%m/%d}({weekday})"


def to_finite_number(value):
    """숫자 또는 쉼표 포함 문자열을 유한한 float로 변환합니다."""
    if value is None:
        return None

    try:
        if isinstance(value, str):
            value = value.replace(",", "").strip()

            if not value:
                return None

        number = float(value)

        if not math.isfinite(number):
            return None

        return number

    except (TypeError, ValueError, OverflowError):
        return None


# =========================================================
# PER 계산용 재무 엑셀
# =========================================================

@st.cache_data(
    ttl=3600,
    show_spinner=False,
    max_entries=100,
)
def cached_read_per_financial(
    excel_path,
    stock_name,
    modified_ns,
    file_size,
):
    """
    PER 재무 엑셀을 읽습니다(캐시).

    파일 변경 시 modified_ns/file_size가 달라져 다시 읽습니다.
    검증·계산 규칙은 per_status.py와 공유합니다.
    """
    return per_status.read_per_financial(
        excel_path,
        stock_name,
    )


def build_per_status(
    excel_path,
    stock_name,
    latest,
):
    """
    기준일 종가 기준 PER을 계산합니다.

    계산식은 per_status.build_per_status에 위임합니다
    (batch_daily.py와 동일한 규칙을 쓰기 위해서입니다).
    """
    status = per_status.empty_per_status()

    reference = normalize_date(
        latest.get("기준일")
    )

    close_price = to_finite_number(
        latest.get("종가")
    )

    if pd.isna(reference):
        status["PER 설명"] = (
            "PER 계산에 필요한 기준일이 없습니다."
        )
        return status

    if close_price is None or close_price <= 0:
        status["PER 설명"] = (
            "PER 계산에 필요한 유효한 종가가 없습니다."
        )
        return status

    try:
        file_stat = excel_path.stat()

        financial = cached_read_per_financial(
            str(excel_path),
            stock_name,
            file_stat.st_mtime_ns,
            file_stat.st_size,
        )

    except Exception as error:
        status["PER 설명"] = (
            f"PER 계산용 재무 데이터 확인 필요: {error}"
        )
        return status

    return per_status.build_per_status(
        financial,
        reference,
        close_price,
    )


# =========================================================
# 최근 7일 신호 계산
# =========================================================

def build_recent_signal_status(data, latest):
    """
    가격 데이터 기준일 포함 최근 7일의 마지막 신호를 조회합니다.

    거래일이 아닌 주말·공휴일 포함 달력일 기준입니다.
    """
    if data is None or data.empty:
        raise ValueError(
            "신호를 조회할 분석 데이터가 없습니다."
        )

    required = {
        "Weekly_Buy",
        "Weekly_Sell",
        "Weekly_Monthly_Buy",
        "Weekly_Monthly_Sell",
    }

    missing = required - set(data.columns)

    if missing:
        raise ValueError(
            "신호 계산에 필요한 컬럼이 없습니다: "
            + ", ".join(sorted(missing))
        )

    frame = data.copy()
    frame.index = pd.to_datetime(
        frame.index,
        errors="coerce",
    )

    frame = frame.loc[
        ~frame.index.isna()
    ].copy()

    if frame.empty:
        raise ValueError(
            "분석 데이터에 유효한 날짜가 없습니다."
        )

    if frame.index.tz is not None:
        frame.index = (
            frame.index
            .tz_convert("Asia/Seoul")
            .tz_localize(None)
        )

    frame.index = frame.index.normalize()
    frame = frame.sort_index()

    frame = frame.loc[
        ~frame.index.duplicated(keep="last")
    ]

    reference = frame.index[-1]

    start_date = reference - pd.Timedelta(
        days=RECENT_SIGNAL_DAYS - 1
    )

    recent = frame.loc[
        (frame.index >= start_date)
        & (frame.index <= reference)
    ]

    def find_last_signal(buy_col, sell_col):
        buys = pd.to_numeric(
            recent[buy_col],
            errors="coerce",
        ).eq(1)

        sells = pd.to_numeric(
            recent[sell_col],
            errors="coerce",
        ).eq(1)

        matched = recent.loc[buys | sells]

        if matched.empty:
            return "신호 없음", None

        last_row = matched.iloc[-1]

        is_buy = (
            to_finite_number(last_row[buy_col]) == 1
        )

        signal = "매수" if is_buy else "매도"

        signal_date = (
            matched.index[-1]
            .strftime("%Y-%m-%d")
        )

        return signal, signal_date

    weekly_signal, weekly_date = find_last_signal(
        "Weekly_Buy",
        "Weekly_Sell",
    )

    combined_signal, combined_date = find_last_signal(
        "Weekly_Monthly_Buy",
        "Weekly_Monthly_Sell",
    )

    status = dict(latest)

    status.update({
        "기준일": reference.strftime("%Y-%m-%d"),
        "Weekly 신호": weekly_signal,
        "Weekly 신호일": weekly_date,
        "Weekly+Monthly 신호": combined_signal,
        "Weekly+Monthly 신호일": combined_date,
        "신호 조회 시작일": start_date.strftime("%Y-%m-%d"),
        "신호 조회 종료일": reference.strftime("%Y-%m-%d"),
        "신호 조회 일수": RECENT_SIGNAL_DAYS,
    })

    return status


# =========================================================
# 신호 표시
# =========================================================

def get_signal_display(
    signal,
    signal_date,
    reference_date,
):
    """신호 기호·색상·날짜를 반환합니다."""
    parsed_date = normalize_date(signal_date)
    reference = normalize_date(reference_date)

    valid = (
        signal in ("매수", "매도")
        and pd.notna(parsed_date)
        and pd.notna(reference)
    )

    if valid:
        start_date = reference - pd.Timedelta(
            days=RECENT_SIGNAL_DAYS - 1
        )

        valid = (
            start_date <= parsed_date <= reference
        )

    if not valid:
        return {
            "valid": False,
            "symbol": "×",
            "color": "#9ca3af",
            "date_text": "",
        }

    return {
        "valid": True,
        "symbol": "↑" if signal == "매수" else "↓",
        "color": (
            "#dc2626"
            if signal == "매수"
            else "#2563eb"
        ),
        "date_text": format_short_date(parsed_date),
    }


def signal_html(
    signal,
    signal_date,
    reference_date,
    compact=False,
):
    """빨간 위 화살표 / 파란 아래 화살표와 날짜."""
    display = get_signal_display(
        signal,
        signal_date,
        reference_date,
    )

    symbol = display["symbol"]
    color = display["color"]

    if not display["valid"]:
        font_size = 26 if compact else 52

        return (
            f'<span style="color:{color};'
            f'font-size:{font_size}px;'
            'font-weight:800;line-height:1;">'
            f'{symbol}'
            '</span>'
        )

    arrow_size = 32 if compact else 58
    date_size = 18 if compact else 27
    gap = 8 if compact else 12

    date_text = escape(
        display["date_text"]
    )

    return (
        '<span style="display:inline-flex;'
        'align-items:center;flex-wrap:wrap;'
        f'column-gap:{gap}px;row-gap:4px;'
        f'color:{color};">'
        f'<span style="font-size:{arrow_size}px;'
        'font-weight:900;line-height:1;">'
        f'{symbol}'
        '</span>'
        f'<span style="font-size:{date_size}px;'
        'font-weight:800;line-height:1.3;'
        'white-space:nowrap;">'
        f'{date_text}'
        '</span>'
        '</span>'
    )


def render_signal_card(
    title,
    signal,
    signal_date,
    reference_date,
):
    st.markdown(
        f'<div class="signal-card-label">'
        f'{escape(title)}</div>',
        unsafe_allow_html=True,
    )

    content = signal_html(
        signal,
        signal_date,
        reference_date,
        compact=False,
    )

    st.markdown(
        '<div class="signal-card">'
        + content
        + '</div>',
        unsafe_allow_html=True,
    )


# =========================================================
# 전체 결과 요약표
# =========================================================

DELTA_STATE = (("up", "▲", "상승"), ("down", "▼", "하락"))


def price_delta_state(value):
    """등락률 부호를 CSS 상태로 바꿉니다."""
    number = to_finite_number(value)

    if number is None:
        return "flat"

    if number > 0:
        return "up"

    if number < 0:
        return "down"

    return "flat"


def format_run_stamp(value):
    """
    스냅샷 saved_at(ISO)을 '2026-11-20(금) 20:05' 형태로 바꿉니다.

    파싱에 실패하면 원본 문자열을 그대로 돌려줍니다(정보를 숨기지 않음).
    """
    if not value:
        return None

    text = str(value).strip()

    try:
        moment = datetime.fromisoformat(text)

        if moment.tzinfo is None:
            moment = moment.replace(tzinfo=ZoneInfo("Asia/Seoul"))

        moment = moment.astimezone(ZoneInfo("Asia/Seoul"))
        weekday = WEEKDAYS[moment.weekday()]

        return f"{moment:%Y-%m-%d}({weekday}) {moment:%H:%M}"

    except ValueError:
        return text


def render_page_heading(title, saved_at, source):
    """
    제목 옆에 마지막 분석 실행 날짜·시간을 붙입니다.

    source가 'batch'면 예약 실행, 그 외는 앱에서 직접 실행한 결과입니다.
    실행 기록이 없으면 배지를 붙이지 않습니다(잘못된 시각을 보여주지 않기 위해).
    """
    stamp = format_run_stamp(saved_at)

    if not stamp:
        st.markdown(
            f'<div class="page-heading"><h1>{escape(title)} '
            '<span class="heading-time is-none">실행 기록 없음</span>'
            "</h1></div>",
            unsafe_allow_html=True,
        )
        return

    label = "자동 실행" if str(source) == "batch" else "직접 실행"
    css = "is-batch" if str(source) == "batch" else ""

    st.markdown(
        f'<div class="page-heading"><h1>{escape(title)}'
        f'<span class="heading-time {css}">{escape(stamp)} '
        f"· {label}</span></h1></div>",
        unsafe_allow_html=True,
    )


def render_price_card(latest):
    """
    최근 종가 카드 — 값과 전일대비 등락률을 함께 표시합니다.

    상승은 빨강, 하락은 파랑(국내 증시 표기와 동일).
    값이 없으면 회색 '—'로 표시해, 계산 불가를 숨기지 않습니다.
    """
    close_price = to_finite_number(latest.get("종가"))
    change = to_finite_number(latest.get("전일대비(%)"))

    value_text = (
        f"{close_price:,.0f}원" if close_price is not None else "—"
    )

    state = price_delta_state(change)

    if change is None:
        delta_text = "전일대비 —"
    else:
        arrow = "▲" if state == "up" else ("▼" if state == "down" else "")
        delta_text = f"{arrow} {abs(change):.2f}%"

    title = "전일 종가 대비 등락률"

    st.markdown(
        '<div class="price-card">'
        '<div class="price-label">최근 종가</div>'
        '<div class="price-row">'
        f'<span class="price-value">{value_text}</span>'
        f'<span class="price-delta {state}" title="{title}">'
        f"{delta_text}</span>"
        "</div></div>",
        unsafe_allow_html=True,
    )


def trend_lights_cell(latest):
    """
    요약표 한 칸에 들어갈 추세 신호등(일·주·월) HTML을 만듭니다.

    상세 화면의 카드와 같은 색 규칙을 씁니다.
    상승=빨강 / 하락=파랑 / 보합=회색 / 값 없음=점선
    """
    parts = []

    for name, key in TREND_LIGHTS:
        css = trend_state(latest.get(key))

        parts.append(
            '<span class="ov-trend">'
            f'<span class="ov-dot {css}"></span>'
            f'<span class="ov-name">{name}</span>'
            "</span>"
        )

    return '<span class="ov-trends">' + "".join(parts) + "</span>"


def render_nomad_buy_table(results):
    """
    '유목민 매수 종목' 표를 그립니다.

    조건: 전일 대비 거래량 2배 이상 + 상승 -> 다음날 거래량 25% 이하 + 음봉
    조건을 충족한 종목이 없으면 아무것도 그리지 않습니다.
    """
    rows = []

    for position, result in enumerate(results, start=1):
        latest = result.get("latest") or {}

        if not latest.get("유목민 매수"):
            continue

        name = escape(str(result["stock_name"]))
        ticker = escape(str(result["ticker"]))
        anchor = f"stock-detail-{position}"

        multiple = to_finite_number(latest.get("유목민 거래량배수"))
        dry = to_finite_number(latest.get("유목민 급감비율"))
        change = to_finite_number(latest.get("유목민 음봉등락률"))
        close_price = to_finite_number(latest.get("종가"))
        per_value = to_finite_number(latest.get("PER"))

        state = price_delta_state(change)
        arrow = "▲" if state == "up" else ("▼" if state == "down" else "")
        change_text = "-" if change is None else f"{arrow} {abs(change):.2f}%"

        def fmt_cell(value, spec, suffix=""):
            return f"{value:{spec}}{suffix}" if value is not None else "-"

        surge_day = escape(str(latest.get("유목민 폭등일") or "-"))
        dry_day = escape(str(latest.get("유목민 음봉일") or "-"))

        rows.append(
            "<tr>"
            f'<td><a href="#{anchor}" target="_self" '
            'style="font-weight:700;color:inherit;text-decoration:none;">'
            f"{name} "
            '<span style="color:#808080;font-size:12px;font-weight:400;">'
            f"({ticker})</span></a></td>"
            f"<td>{surge_day}</td>"
            f'<td class="num">{fmt_cell(multiple, ",.2f", "배")}</td>'
            f"<td>{dry_day}</td>"
            f'<td class="num">{fmt_cell(dry, ",.1f", "%")}</td>'
            f'<td class="num price-delta {state}">{change_text}</td>'
            f'<td class="num">{fmt_cell(close_price, ",.0f", "원")}</td>'
            f'<td class="num">{fmt_cell(per_value, ",.1f")}</td>'
            "</tr>"
        )

    if not rows:
        return

    css = (
        ".nomad-wrap { margin:4px 0 2px; }"
        ".nomad-head { display:flex; align-items:center; gap:10px; margin-bottom:6px; }"
        ".nomad-head .nomad-title { font-weight:800; font-size:1.05rem;"
        " color:#0f172a; letter-spacing:-.018em; }"
        ".nomad-head .nomad-count { font-size:.78rem; font-weight:800;"
        " color:#92400e; background:#fef3c7; border:1px solid #fde68a;"
        " border-radius:999px; padding:3px 10px; }"
        ".nomad-rule { font-size:.8rem; color:#64748b; margin:0 0 10px 2px; }"
        ".nomad-table { width:100%; border-collapse:separate; border-spacing:0;"
        " font-size:14px; background:#fff; border:1px solid #fde68a;"
        " border-radius:14px; overflow:hidden;"
        " box-shadow:0 2px 10px rgba(180,83,9,.10); }"
        ".nomad-table th { padding:11px 16px; text-align:left; white-space:nowrap;"
        " font-weight:700; color:#92400e; font-size:12px; letter-spacing:.03em;"
        " background:#fffbeb; border-bottom:1px solid #fde68a; }"
        ".nomad-table td { padding:11px 16px; border-bottom:1px solid #fffbeb;"
        " vertical-align:middle; }"
        ".nomad-table tbody tr:last-child td { border-bottom:none; }"
        ".nomad-table tbody tr:hover { background:#fffdf5; }"
        ".nomad-table .num { text-align:right; font-variant-numeric:tabular-nums; }"
        ".nomad-table th.num { text-align:right; }"
        ".nomad-table .price-delta { font-weight:800; }"
        ".nomad-table .price-delta.up { color:#dc2626; }"
        ".nomad-table .price-delta.down { color:#2563eb; }"
        ".nomad-table .price-delta.flat { color:#475569; }"
    )

    head = (
        '<div class="nomad-wrap">'
        '<div class="nomad-head">'
        '<span class="nomad-title">🐪 유목민 매수 종목</span>'
        f'<span class="nomad-count">{len(rows)}종목</span>'
        "</div>"
        '<div class="nomad-rule">'
        "전일 대비 거래량 2배 이상 + 상승 &rarr; 다음날 거래량 25% 이하 + 음봉"
        "</div>"
        '<div style="overflow-x:auto;">'
        '<table class="nomad-table"><thead><tr>'
        "<th>종목</th><th>폭등일</th>"
        '<th class="num">거래량 배수</th>'
        "<th>음봉일</th>"
        '<th class="num">급감 비율</th>'
        '<th class="num">음봉 등락률</th>'
        '<th class="num">종가</th>'
        '<th class="num">PER</th>'
        "</tr></thead><tbody>"
    )

    st.markdown(
        "<style>" + css + "</style>" + head + "".join(rows) + "</tbody></table></div></div>",
        unsafe_allow_html=True,
    )


def render_results_overview(results):
    """종목명·신호 클릭 시 해당 종목 상세분석으로 이동."""
    if not results:
        return

    st.subheader(
        "전체 결과 요약",
        anchor="results-overview",
    )

    rows = []

    for position, result in enumerate(
        results,
        start=1,
    ):
        latest = result["latest"]

        name = escape(
            str(result["stock_name"])
        )

        ticker = escape(
            str(result["ticker"])
        )

        anchor = f"stock-detail-{position}"

        weekly_html = signal_html(
            latest.get(
                "Weekly 신호",
                "신호 없음",
            ),
            latest.get("Weekly 신호일"),
            latest.get("기준일"),
            compact=True,
        )

        combined_html = signal_html(
            latest.get(
                "Weekly+Monthly 신호",
                "신호 없음",
            ),
            latest.get(
                "Weekly+Monthly 신호일"
            ),
            latest.get("기준일"),
            compact=True,
        )

        name_link = (
            f'<a href="#{anchor}" target="_self" '
            'style="display:block;color:inherit;'
            'text-decoration:none;font-weight:700;">'
            f'{name} '
            '<span style="color:#808080;'
            'font-size:13px;font-weight:400;">'
            f'({ticker})'
            '</span></a>'
        )

        weekly_link = (
            f'<a href="#{anchor}" target="_self" '
            'style="display:block;'
            'text-decoration:none;">'
            f'{weekly_html}</a>'
        )

        combined_link = (
            f'<a href="#{anchor}" target="_self" '
            'style="display:block;'
            'text-decoration:none;">'
            f'{combined_html}</a>'
        )

        trend_link = (
            f'<a href="#{anchor}" target="_self" '
            'style="display:block;text-decoration:none;">'
            f'{trend_lights_cell(latest)}</a>'
        )

        rows.append(
            '<tr>'
            f'<td>{name_link}</td>'
            f'<td>{weekly_link}</td>'
            f'<td>{combined_link}</td>'
            f'<td>{trend_link}</td>'
            '</tr>'
        )

    html = (
        '<style>'
        '.stock-overview-table {'
        'width:100%;border-collapse:separate;border-spacing:0;'
        'font-size:15.5px;'
        'background:#ffffff;'
        'border:1px solid #e8ecf3;border-radius:16px;overflow:hidden;'
        'box-shadow:0 2px 8px rgba(15,23,42,.07);'
        '}'
        '.stock-overview-table th {'
        'padding:13px 20px;text-align:left;'
        'white-space:nowrap;font-weight:700;color:#64748b;'
        'font-size:12.5px;text-transform:uppercase;letter-spacing:.05em;'
        'border-bottom:1px solid #e8ecf3;'
        'background:#f8fafc;'
        '}'
        '.stock-overview-table td {'
        'padding:13px 20px;'
        'vertical-align:middle;'
        'border-bottom:1px solid #eef2f7;'
        '}'
        '.stock-overview-table tbody tr:last-child td {'
        'border-bottom:none;'
        '}'
        '.stock-overview-table tbody tr:hover {'
        'background:#f5f6ff;'
        '}'
        '.stock-overview-table .ov-trends {'
        'display:inline-flex;gap:12px;align-items:flex-start;'
        '}'
        '.stock-overview-table .ov-trend {'
        'display:inline-flex;flex-direction:column;'
        'align-items:center;gap:3px;min-width:16px;'
        '}'
        '.stock-overview-table .ov-dot {'
        'width:12px;height:12px;border-radius:50%;'
        'display:inline-block;'
        '}'
        '.stock-overview-table .ov-dot.up {'
        'background:#dc2626;box-shadow:0 0 0 2px #fecaca;'
        '}'
        '.stock-overview-table .ov-dot.down {'
        'background:#2563eb;box-shadow:0 0 0 2px #bfdbfe;'
        '}'
        '.stock-overview-table .ov-dot.flat {'
        'background:#e2e8f0;box-shadow:0 0 0 2px #eef2f7;'
        '}'
        '.stock-overview-table .ov-dot.none {'
        'background:#fff;border:2px dashed #cbd5e1;'
        '}'
        '.stock-overview-table .ov-name {'
        'font-size:10px;line-height:1;font-weight:700;'
        'color:#94a3b8;'
        '}'
        '</style>'
        '<div style="overflow-x:auto;">'
        '<table class="stock-overview-table">'
        '<thead><tr>'
        '<th>종목</th>'
        '<th>최근 주간신호</th>'
        '<th>최근 주간·월간신호</th>'
        '<th>일간/주간/월간 추세</th>'
        '</tr></thead><tbody>'
        + "".join(rows)
        + '</tbody></table></div>'
    )

    st.markdown(
        html,
        unsafe_allow_html=True,
    )

    # 요약표 바로 아래에 유목민 매수 종목을 붙입니다.
    render_nomad_buy_table(results)

    st.divider()


# =========================================================
# 차트 저장
# =========================================================

def save_chart_png(
    png_bytes,
    stock_name,
    analysis_date,
):
    """같은 종목·날짜의 PNG는 덮어씁니다."""
    if not png_bytes:
        raise ValueError(
            "저장할 차트 이미지가 없습니다."
        )

    IMAGE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    filename = (
        f"{stock_name}_{analysis_date}.png"
    )

    target_path = IMAGE_DIR / filename

    temporary_path = (
        IMAGE_DIR
        / f".{uuid.uuid4().hex}.tmp"
    )

    try:
        temporary_path.write_bytes(png_bytes)

        os.replace(
            temporary_path,
            target_path,
        )

    finally:
        temporary_path.unlink(
            missing_ok=True
        )

    return target_path, png_bytes


# =========================================================
# 표 스타일
# =========================================================

def apply_style_map(
    styled,
    function,
    subset,
):
    """pandas 신·구 버전의 Styler를 지원합니다."""
    if hasattr(styled, "map"):
        return styled.map(
            function,
            subset=subset,
        )

    return styled.applymap(
        function,
        subset=subset,
    )


def korean_signal_table(frame):
    table = frame.copy()

    trend_columns = {
        "Daily_Trend": "일간 추세",
        "Weekly_Trend": "주간 추세",
        "Monthly_Trend": "월간 추세",
    }

    trend_labels = {
        1: "● 상승",
        -1: "● 하락",
        0: "● 중립",
    }

    for column in trend_columns:
        if column in table.columns:
            table[column] = (
                pd.to_numeric(
                    table[column],
                    errors="coerce",
                )
                .map(trend_labels)
                .fillna("없음")
            )

    table = table.rename(
        columns=trend_columns
    )

    if "날짜" in table.columns:
        table["날짜"] = table["날짜"].map(
            format_short_date
        )

    if "신호" in table.columns:
        table["신호"] = table["신호"].replace({
            "매수": "↑",
            "매도": "↓",
            "신호 없음": "×",
            "↑ 매수": "↑",
            "↓ 매도": "↓",
        })

    for column in (
        "종가",
        "목표가",
        "괴리율(%)",
    ):
        if column in table.columns:
            table[column] = table[column].map(
                to_finite_number
            )

    def divergence_color(value):
        number = to_finite_number(value)

        if number is None:
            return ""

        if number > 0:
            return (
                "color:#2563eb;"
                "font-weight:bold;"
            )

        if number < 0:
            return (
                "color:#dc2626;"
                "font-weight:bold;"
            )

        return ""

    def signal_color(value):
        if value == "↑":
            return (
                "color:#dc2626;"
                "font-weight:bold;"
            )

        if value == "↓":
            return (
                "color:#2563eb;"
                "font-weight:bold;"
            )

        if value == "×":
            return "color:#9ca3af;"

        return ""

    def trend_color(value):
        if value == "● 상승":
            return (
                "color:#dc2626;"
                "font-weight:bold;"
            )

        if value == "● 하락":
            return (
                "color:#2563eb;"
                "font-weight:bold;"
            )

        if value == "● 중립":
            return "color:#6b7280;"

        return ""

    formats = {
        column: "{:,.0f}"
        for column in ("종가", "목표가")
        if column in table.columns
    }

    if "괴리율(%)" in table.columns:
        formats["괴리율(%)"] = "{:.1f}%"

    styled = table.style.format(
        formats,
        na_rep="",
    )

    if "괴리율(%)" in table.columns:
        styled = apply_style_map(
            styled,
            divergence_color,
            subset=["괴리율(%)"],
        )

    if "신호" in table.columns:
        styled = apply_style_map(
            styled,
            signal_color,
            subset=["신호"],
        )

    trend_subset = [
        column
        for column in (
            "일간 추세",
            "주간 추세",
            "월간 추세",
        )
        if column in table.columns
    ]

    if trend_subset:
        styled = apply_style_map(
            styled,
            trend_color,
            subset=trend_subset,
        )

    return styled


def korean_metrics_table(frame):
    table = frame.copy()

    if "전략" in table.columns:
        table["전략"] = (
            table["전략"]
            .astype(str)
            .str.replace(
                "Weekly+Monthly",
                "주간·월간",
                regex=False,
            )
            .str.replace(
                "Weekly",
                "주간",
                regex=False,
            )
        )

    return table


def profit_color(value):
    if isinstance(value, str):
        value = (
            value
            .replace(",", "")
            .replace("%", "")
            .strip()
        )

    number = to_finite_number(value)

    if number is None:
        return ""

    if number > 0:
        return (
            "color:#dc2626;"
            "font-weight:bold;"
        )

    if number < 0:
        return (
            "color:#2563eb;"
            "font-weight:bold;"
        )

    return ""


def make_trade_table(trades, strategy):
    table = trades.copy().drop(
        columns=["전략"],
        errors="ignore",
    )

    if strategy == "Weekly":
        table = table.drop(
            columns=[
                "수익(원)",
                "보유기간(일)",
                "정산사유",
                "청산사유",
            ],
            errors="ignore",
        )

    for column in ("매수일", "매도일"):
        if column in table.columns:
            table[column] = (
                pd.to_datetime(
                    table[column],
                    errors="coerce",
                )
                .dt.strftime("%Y-%m-%d")
            )

    numeric_columns = (
        "매수가",
        "매도가",
        "수익(원)",
        "수익률(%)",
    )

    for column in numeric_columns:
        if column in table.columns:
            table[column] = table[column].map(
                to_finite_number
            )

    formats = {
        column: "{:,.0f}"
        for column in (
            "매수가",
            "매도가",
            "수익(원)",
        )
        if column in table.columns
    }

    if "수익률(%)" in table.columns:
        formats["수익률(%)"] = "{:.2f}%"

    styled = table.style.format(
        formats,
        na_rep="",
    )

    if "수익률(%)" in table.columns:
        styled = apply_style_map(
            styled,
            profit_color,
            subset=["수익률(%)"],
        )

    return styled


# =========================================================
# 宗목 상세 결과
# =========================================================

TREND_LIGHTS = (
    ("일", "Daily_Trend"),
    ("주", "Weekly_Trend"),
    ("월", "Monthly_Trend"),
)


TREND_STATE_LABEL = {
    "up": "상승",
    "down": "하락",
    "flat": "보합",
    "none": "값 없음",
}


def trend_state(value):
    """
    추세 값을 화면 상태로 바꿉니다.

    up(상승·빨강) / down(하락·파랑) / flat(보합) / none(값 없음)
    """
    number = to_finite_number(value)

    if number is None:
        return "none"

    if number > 0:
        return "up"

    if number < 0:
        return "down"

    return "flat"


def render_trend_lights(latest):
    """
    일간·주간·월간 추세를 신호등(원) 3개만 표시합니다.

    - 상승: 빨간 원 / 하락: 파란 원 / 보합: 회색 원 / 값 없음: 점선 원
    - 카드 라벨과 상태 문구는 넣지 않습니다(높이를 다른 항목과 맞추기 위해).
    """
    lights = []
    missing = False

    for name, key in TREND_LIGHTS:
        if key not in latest:
            missing = True

        css = trend_state(latest.get(key))

        lights.append(
            '<div class="trend-light">'
            f'<span class="trend-dot {css}" '
            f'title="{name}간 추세: {TREND_STATE_LABEL[css]}"></span>'
            f'<span class="trend-name">{name}</span>'
            "</div>"
        )

    st.markdown(
        '<div class="trend-card">'
        '<div class="trend-lights">'
        + "".join(lights)
        + "</div></div>",
        unsafe_allow_html=True,
    )

    if missing:
        st.caption(
            "추세 값이 없는 저장본입니다. "
            "‘분석 실행’ 또는 다음 자동 실행 후 표시됩니다."
        )


def render_stock_result(result, position):
    name = result["stock_name"]
    ticker = result["ticker"]
    latest = result["latest"]

    reference_date = normalize_date(
        latest.get("기준일")
    )

    if pd.isna(reference_date):
        raise ValueError(
            "가격 데이터 기준일이 올바르지 않습니다."
        )

    st.subheader(
        f"{position}. {name} ({ticker})",
        anchor=f"stock-detail-{position}",
    )

    st.markdown(
        '<a class="back-link" '
        'href="#results-overview" '
        'target="_self">↑ 전체 결과 요약으로 돌아가기</a>',
        unsafe_allow_html=True,
    )

    # -----------------------------------------------------
    # 첫 번째 줄: 최근 종가 / PER / 일간·주간·월간 추세
    # -----------------------------------------------------

    # 위젯 키는 문서 전체에서 유일해야 합니다.
    # 여러 종목을 한 화면에 그리므로 position·ticker를 붙입니다.
    with st.container(key=f"price_per_row_{position}_{ticker}"):
        price_column, per_column, trend_column = st.columns(3)

        with price_column:
            render_price_card(latest)

        with per_column:
            per_value = to_finite_number(
                latest.get("PER")
            )

            st.metric(
                "PER",
                (
                    f"{per_value:,.2f}배"
                    if per_value is not None
                    else "—"
                ),
            )

        with trend_column:
            render_trend_lights(latest)

    # -----------------------------------------------------
    # 두 번째 줄: 최근 7일 매매 신호
    # -----------------------------------------------------

    weekly_column, combined_column = st.columns(2)

    with weekly_column:
        render_signal_card(
            title="최근 주간신호",
            signal=latest.get(
                "Weekly 신호",
                "신호 없음",
            ),
            signal_date=latest.get(
                "Weekly 신호일"
            ),
            reference_date=reference_date,
        )

    with combined_column:
        render_signal_card(
            title="최근 주간·월간신호",
            signal=latest.get(
                "Weekly+Monthly 신호",
                "신호 없음",
            ),
            signal_date=latest.get(
                "Weekly+Monthly 신호일"
            ),
            reference_date=reference_date,
        )

    for message in result.get("messages", []):
        st.info(message)

    # -----------------------------------------------------
    # 최근 신호 표
    # -----------------------------------------------------

    st.markdown("### 최근 매매 신호")

    st.markdown("**주간 단독 신호**")

    weekly_signals = result["weekly_signals"].head(
        RECENT_SIGNAL_LIMIT
    )

    if weekly_signals.empty:
        st.info(
            "해당 기간에 발생한 "
            "주간 매매 신호가 없습니다."
        )

    else:
        st.dataframe(
            korean_signal_table(weekly_signals),
            hide_index=True,
            use_container_width=True,
        )

    st.markdown("**주간·월간 결합 신호**")

    combined_signals = result["combined_signals"].head(
        RECENT_SIGNAL_LIMIT
    )

    if combined_signals.empty:
        st.info(
            "해당 기간에 발생한 "
            "주간·월간 결합 신호가 없습니다."
        )

    else:
        st.dataframe(
            korean_signal_table(combined_signals),
            hide_index=True,
            use_container_width=True,
        )

    # -----------------------------------------------------
    # 종합차트
    # -----------------------------------------------------

    st.markdown("### 종합차트")

    png_bytes = result.get("png_bytes")

    if png_bytes:
        st.image(
            png_bytes,
            use_container_width=True,
        )

        image_filename = (
            result.get("image_download_name")
            or result.get("image_filename")
        )

        is_preview = bool(
            result.get("image_is_preview")
        )

        st.download_button(
            (
                f"{name} 차트 다운로드 (미리보기)"
                if is_preview
                else f"{name} 고해상도 차트 다운로드"
            ),
            data=png_bytes,
            file_name=(
                image_filename
                or f"{name}_chart.png"
            ),
            mime=result.get(
                "image_mime",
                "image/png",
            ),
            key=f"chart_{position}_{ticker}",
        )

        if result.get("image_note"):
            st.caption(result["image_note"])

    else:
        st.warning(
            "종합차트를 생성하거나 저장하지 못했습니다. "
            "최근 신호와 거래 기록은 확인할 수 있습니다."
        )

    # -----------------------------------------------------
    # 전략별 성과: 항상 표시
    # -----------------------------------------------------

    st.markdown("### 전략별 성과")

    metrics = result["metrics"]

    if metrics.empty:
        st.info(
            "표시할 전략별 성과가 없습니다."
        )

    else:
        st.dataframe(
            korean_metrics_table(metrics),
            hide_index=True,
            use_container_width=True,
        )

    # -----------------------------------------------------
    # 거래 기록: 전략별 기본 접힘
    # -----------------------------------------------------

    st.markdown("### 거래 기록")

    strategy_names = {
        "Weekly": "주간 단독 전략",
        "Weekly+Monthly": "주간·월간 결합 전략",
    }

    trades_by_strategy = result.get(
        "trades",
        {},
    )

    strategy_order = [
        "Weekly",
        "Weekly+Monthly",
    ]

    strategy_order.extend(
        strategy
        for strategy in trades_by_strategy
        if strategy not in strategy_order
    )

    for strategy in strategy_order:
        trades = trades_by_strategy.get(
            strategy,
            pd.DataFrame(),
        )

        label = strategy_names.get(
            strategy,
            strategy,
        )

        with st.expander(
            f"{label} 거래 기록",
            expanded=False,
        ):
            if trades.empty:
                st.info(
                    "청산이 완료된 거래가 없습니다."
                )

            else:
                st.dataframe(
                    make_trade_table(
                        trades,
                        strategy,
                    ),
                    hide_index=True,
                    use_container_width=True,
                )

    st.divider()


# =========================================================
# 결과 직렬화
# =========================================================

def dataframe_to_json_data(frame):
    return json.loads(
        frame.to_json(
            orient="split",
            date_format="iso",
            force_ascii=False,
        )
    )


def json_data_to_dataframe(data):
    return pd.read_json(
        io.StringIO(
            json.dumps(
                data,
                ensure_ascii=False,
            )
        ),
        orient="split",
    )


def json_safe(value):
    """
    pandas/numpy 값과 결측값을
    엄격한 JSON으로 저장 가능한 형태로 바꿉니다.
    """
    if value is None or value is pd.NA or value is pd.NaT:
        return None

    if isinstance(value, dict):
        return {
            str(key): json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            json_safe(item)
            for item in value
        ]

    if isinstance(value, (datetime, date, pd.Timestamp)):
        return value.isoformat()

    if isinstance(value, bool):
        return value

    if isinstance(value, Integral):
        return int(value)

    if isinstance(value, Real):
        number = float(value)

        return (
            number
            if math.isfinite(number)
            else None
        )

    if isinstance(value, str):
        return value

    # numpy scalar 등
    if hasattr(value, "item"):
        return json_safe(value.item())

    raise TypeError(
        "JSON으로 저장할 수 없는 자료형입니다: "
        f"{type(value).__name__}"
    )


# =========================================================
# 마지막 결과 저장
# =========================================================

def save_last_analysis(
    results,
    records,
    period,
    complete=False,
):
    """
    마지막 실행 결과를 스냅샷으로 저장합니다.

    종목이 하나 끝날 때마다 complete=False로 호출하므로,
    브라우저를 닫거나 앱이 잠들어도 직전까지의 결과가 남습니다.
    마지막 호출은 complete=True로 전체 완료를 표시합니다.

    스냅샷 구성
    - 표·수치: JSON(orient="split") — 인덱스까지 보존
    - 차트: 원본 PNG는 img/ 폴더에 두고,
            스냅샷에는 저용량 JPEG 미리보기만 담습니다.
            (300dpi PNG를 base64로 넣으면 20종목에서 수십 MB가 되어
             Cloud 메모리·응답시간을 잡아먹습니다)
    """
    if not results:
        return False

    saved_at = datetime.now(
        ZoneInfo("Asia/Seoul")
    ).isoformat(timespec="seconds")

    try:
        payload = result_store.build_snapshot(
            results=results,
            records=records,
            period=period,
            saved_at=saved_at,
            complete=complete,
            meta={
                "recent_signal_days": RECENT_SIGNAL_DAYS,
                "recent_signal_limit": RECENT_SIGNAL_LIMIT,
            },
            source="app",
        )

        RESULT_STORE.save(payload)

    except Exception as error:
        # 저장 실패가 분석 자체를 막지는 않습니다.
        st.session_state["save_error"] = (
            f"{type(error).__name__}: {error}"
        )
        return False

    st.session_state["last_saved_at"] = saved_at
    st.session_state.pop("save_error", None)
    return True


def apply_restored_result(restored):
    """복원된 결과를 세션 상태에 반영합니다."""
    st.session_state["batch_results"] = restored["results"]
    st.session_state["batch_records"] = restored["records"]
    st.session_state["batch_period"] = restored["period"]
    st.session_state["last_saved_at"] = restored["saved_at"]
    st.session_state["result_source"] = restored.get("source", "app")

    notes = list(restored.get("notes") or [])

    if notes:
        st.session_state["restore_notice"] = " ".join(notes)


def load_last_analysis():
    """
    재접속(새 세션) 시 마지막 저장 결과를 복원합니다.

    Streamlit Cloud는 세션이 끊기면 st.session_state가 사라지므로,
    이 함수가 '다시 접속했을 때 결과가 보이는' 경로입니다.
    """
    try:
        payload = RESULT_STORE.load()

    except Exception as error:
        st.session_state["restore_notice"] = (
            "저장된 결과를 읽지 못했습니다"
            f"({type(error).__name__}). "
            "저장소 설정을 확인해 주세요."
        )
        return False

    if not payload:
        return False

    try:
        restored = result_store.restore_snapshot(
            payload,
            image_dir=IMAGE_DIR,
        )

    except result_store.SnapshotError as error:
        st.session_state["restore_notice"] = (
            f"{error} '분석 실행'을 다시 눌러 주세요."
        )
        return False

    except Exception as error:
        st.session_state["restore_notice"] = (
            "저장된 결과 형식이 올바르지 않습니다"
            f"({type(error).__name__}). "
            "'분석 실행'을 다시 눌러 주세요."
        )
        return False

    apply_restored_result(restored)
    return True


# =========================================================
# 종목별 순차 분석
# =========================================================

def run_batch(
    selected_stocks,
    start,
    end,
    include_investors,
    overview_container,
    progress_container,
    results_container,
    nomad_container=None,
):
    results = []
    records = []

    st.session_state["batch_results"] = results
    st.session_state["batch_records"] = records

    st.session_state["batch_period"] = (
        start.isoformat(),
        end.isoformat(),
    )

    st.session_state.pop(
        "last_saved_at",
        None,
    )

    st.session_state.pop(
        "restore_notice",
        None,
    )

    analysis_date = datetime.now(
        ZoneInfo("Asia/Seoul")
    ).strftime("%Y%m%d")

    with progress_container:
        progress = st.progress(0)
        status = st.empty()

    total = len(selected_stocks)

    for index, (stock_name, ticker) in enumerate(
        selected_stocks,
        start=1,
    ):
        status.info(
            f"[{index}/{total}] "
            f"{stock_name} ({ticker}) 분석 중입니다."
        )

        excel_path = (
            FINANCIAL_DIR
            / f"{stock_name}.xlsx"
        )

        # 재무 엑셀이 없는 종목도 분석합니다.
        # (배치와 같은 규칙: 목표가·괴리율·PER만 비우고 신호·차트는 정상 산출)
        excel_missing = not excel_path.is_file()

        if excel_missing:
            excel_path = None

        # -------------------------------------------------
        # 데이터 조회 및 분석
        # -------------------------------------------------

        try:
            frame, fetch_messages = cached_fetch(
                ticker,
                start.strftime("%Y%m%d"),
                end.strftime("%Y%m%d"),
                include_investors,
                DATA_CACHE_VERSION,
            )

            analysis = analyze_frame(
                frame,
                stock_name,
                excel_file=(str(excel_path) if excel_path else None),
            )

            # 엔진 내부 기본값과 관계없이
            # app.py에서 최근 7일 신호를 다시 계산합니다.
            latest = build_recent_signal_status(
                data=analysis["data"],
                latest=analysis["latest"],
            )

            # PER은 원본 엑셀에서 별도로 읽습니다.
            if excel_path is None:
                latest.update(per_status.empty_per_status())
                latest["PER 설명"] = (
                    f"file/{stock_name}.xlsx 가 없어 "
                    "PER을 계산하지 않았습니다."
                )
            else:
                latest.update(
                    build_per_status(
                        excel_path=excel_path,
                        stock_name=stock_name,
                        latest=latest,
                    )
                )

            if excel_missing:
                messages_note = (
                    f"file/{stock_name}.xlsx 가 없어 "
                    "목표가·괴리율·PER 없이 분석했습니다."
                )
                analysis.setdefault("messages", [])
                if messages_note not in analysis["messages"]:
                    analysis["messages"].append(messages_note)

            # recent_signals는 기존 함수와 동일하게
            # 날짜 오름차순 반환을 전제로 역순 표시합니다.
            weekly_signals = (
                recent_signals(
                    analysis["data"],
                    combined=False,
                    limit=RECENT_SIGNAL_LIMIT,
                )
                .iloc[::-1]
                .reset_index(drop=True)
            )

            combined_signals = (
                recent_signals(
                    analysis["data"],
                    combined=True,
                    limit=RECENT_SIGNAL_LIMIT,
                )
                .iloc[::-1]
                .reset_index(drop=True)
            )

            display_result = {
                "stock_name": stock_name,
                "ticker": ticker,
                "latest": latest,
                "metrics": analysis["metrics"],
                "trades": analysis["trades"],
                "weekly_signals": weekly_signals,
                "combined_signals": combined_signals,
                "messages": (
                    list(fetch_messages or [])
                    + list(
                        analysis.get("messages") or []
                    )
                ),
                "png_bytes": None,
                "image_filename": None,
            }

        except Exception as error:
            records.append({
                "종목명": stock_name,
                "종목코드": ticker,
                "상태": "분석 실패",
                "내용": str(error),
            })

            st.session_state["batch_records"] = records

            progress.progress(
                index / total
            )
            continue

        # -------------------------------------------------
        # 차트 생성 및 저장
        # -------------------------------------------------

        chart_error = None

        try:
            png_bytes = make_chart_png(
                analysis["data"],
                stock_name,
                excel_file=(str(excel_path) if excel_path else None),
            )

            image_path, saved_png = save_chart_png(
                png_bytes,
                stock_name,
                analysis_date,
            )

            display_result["png_bytes"] = (
                saved_png
            )

            display_result["image_filename"] = (
                image_path.name
            )

        except Exception as error:
            chart_error = str(error)

            display_result["messages"].append(
                "차트 생성 또는 저장 실패: "
                f"{chart_error}"
            )

        results.append(
            display_result
        )

        records.append({
            "종목명": stock_name,
            "종목코드": ticker,
            "상태": (
                "완료"
                if chart_error is None
                else "분석 완료·차트 실패"
            ),
            "내용": (
                f'img/{display_result["image_filename"]}'
                if chart_error is None
                else chart_error
            ),
        })

        st.session_state["batch_results"] = results
        st.session_state["batch_records"] = records

        # 완료된 종목부터 상세 결과 표시
        try:
            with results_container:
                render_stock_result(
                    display_result,
                    position=len(results),
                )

        except Exception as error:
            records[-1]["상태"] = (
                "분석 완료·화면 표시 실패"
            )

            records[-1]["내용"] = str(error)

            with results_container:
                st.error(
                    f"{stock_name} 결과 표시 실패: "
                    f"{error}"
                )

        # 상단 요약표 갱신
        with overview_container.container():
            render_results_overview(results)

        del analysis
        del frame

        # -------------------------------------------------
        # 중간 저장(checkpoint)
        # 긴 실행 중 브라우저를 닫거나 앱이 잠들어도
        # 직전 종목까지의 결과가 저장소에 남습니다.
        # -------------------------------------------------

        save_last_analysis(
            results=results,
            records=records,
            period=(
                start.isoformat(),
                end.isoformat(),
            ),
            complete=False,
        )

        progress.progress(
            index / total
        )

    skipped = sum(
        row["상태"] == "건너뜀"
        for row in records
    )

    failed = sum(
        row["상태"] == "분석 실패"
        for row in records
    )

    chart_failed = sum(
        row["상태"] == "분석 완료·차트 실패"
        for row in records
    )

    display_failed = sum(
        row["상태"] == "분석 완료·화면 표시 실패"
        for row in records
    )

    status.info(
        f"분석 종료 — 결과 {len(results)}개 / "
        f"건너뜀 {skipped}개 / "
        f"분석 실패 {failed}개 / "
        f"차트 실패 {chart_failed}개 / "
        f"화면 표시 실패 {display_failed}개"
    )

    if results:
        # save_last_analysis는 예외를 흡수하고 False를 돌려줍니다.
        # 반환값을 확인해야 실패를 화면에 알릴 수 있습니다.
        saved_ok = save_last_analysis(
            results=results,
            records=records,
            period=(
                start.isoformat(),
                end.isoformat(),
            ),
            complete=True,
        )

        if not saved_ok:
            with progress_container:
                st.error(
                    "분석은 완료했지만 결과 저장에 실패했습니다. "
                    f"({st.session_state.get('save_error') or '원인 미상'})"
                )
                st.caption(
                    f"저장소: {RESULT_STORE.describe()} · "
                    "사이드바 '마지막 결과 보관·복원'의 "
                    "'저장 진단'으로 원인을 확인할 수 있습니다."
                )

    else:
        with progress_container:
            st.warning(
                "저장할 분석 결과가 없습니다. "
                "실행한 종목이 모두 실패했는지 확인하세요."
            )


# =========================================================
# 새 세션에서 마지막 결과 복원
# =========================================================

if not st.session_state.get(
    "last_analysis_checked",
    False,
):
    st.session_state["last_analysis_checked"] = True

    if not st.session_state.get("batch_results"):
        try:
            load_last_analysis()

        except Exception as error:
            st.warning(
                "마지막 분석 결과를 불러오지 "
                f"못했습니다: {error}"
            )


# =========================================================
# 등록 종목 불러오기
# =========================================================

try:
    stock_groups = stock_data.load_stock_groups(
        STOCKS_FILE
    )

except Exception as error:
    st.error(str(error))

    st.info(
        "app.py와 같은 폴더에 stocks.txt를 만들고 "
        "[그룹명] 아래에 "
        "'종목명,종목코드'를 등록하세요."
    )

    st.stop()


# =========================================================
# 왼쪽: 설정
# =========================================================







with st.sidebar:
    st.header("등록 종목")

    credentials_present = bool(
        os.environ.get("KRX_ID")
        and os.environ.get("KRX_PW")
    )

    if not credentials_present:
        st.warning(
            "KRX 인증정보가 설정되지 않았습니다. "
            "인증이 필요한 데이터 조회가 "
            "실패할 수 있습니다."
        )

    group_names = list(stock_groups)

    selected_groups = st.multiselect(
        "분석할 그룹",
        options=group_names,
        default=group_names[:1],
    )

    available_stocks = []
    seen = set()

    for group_name in selected_groups:
        for stock_item in stock_groups[group_name]:
            if stock_item not in seen:
                seen.add(stock_item)
                available_stocks.append(
                    stock_item
                )

    if available_stocks:
        st.dataframe(
            pd.DataFrame(
                available_stocks,
                columns=[
                    "종목명",
                    "종목코드",
                ],
            ),
            hide_index=True,
            use_container_width=True,
        )

    else:
        st.info(
            "그룹을 하나 이상 선택하세요."
        )

    # -----------------------------------------------------
    # 그룹 편집·순서 변경은 전용 페이지에서 수행합니다.
    # -----------------------------------------------------

    st.page_link(
        "pages/1_groups.py",
        label="관심 그룹 편집 · 그룹 순서 변경",
        icon="🗂",
    )

    with st.form("batch_analysis_form"):
        selected_stocks = st.multiselect(
            "실행할 종목 — 복수 선택",
            options=available_stocks,
            default=available_stocks,
            format_func=lambda item: (
                f"{item[0]} ({item[1]})"
            ),
        )

        start = st.date_input(
            "시작일",
            value=(
                pd.Timestamp(TODAY_KST)
                - pd.DateOffset(years=3)
            ).date(),
            max_value=TODAY_KST,
        )

        end = st.date_input(
            "종료일",
            value=TODAY_KST,
            max_value=TODAY_KST,
        )

        include_investors = st.checkbox(
            "투자자 순매수 수량 포함",
            value=True,
        )

        submitted = st.form_submit_button(
            "분석 실행",
            use_container_width=True,
        )

    st.divider()

    if st.button(
        "가격·수급 데이터 캐시 초기화"
    ):
        cached_fetch.clear()

        st.success(
            "데이터 캐시를 초기화했습니다. "
            "'분석 실행'을 누르면 새로 조회합니다."
        )

    if st.button(
        "재무 엑셀 캐시 초기화"
    ):
        cached_read_per_financial.clear()

        st.success(
            "PER용 재무 엑셀 캐시를 초기화했습니다. "
            "'분석 실행'을 다시 눌러주세요."
        )

    if st.button(
        "화면 결과 초기화"
    ):
        for key in (
            "batch_results",
            "batch_records",
            "batch_period",
            "last_saved_at",
            "restore_notice",
        ):
            st.session_state.pop(
                key,
                None,
            )

        st.session_state[
            "last_analysis_checked"
        ] = True

    # -----------------------------------------------------
    # 마지막 결과 보관·복원
    # -----------------------------------------------------

    st.divider()

    with st.expander("마지막 결과 보관·복원"):
        st.caption(
            f"저장 위치: {RESULT_STORE.describe()}"
        )

        saved_at_text = st.session_state.get(
            "last_saved_at"
        )

        if saved_at_text:
            st.caption(
                f"마지막 저장: {saved_at_text}"
            )

        save_error = st.session_state.get(
            "save_error"
        )

        if save_error:
            st.warning(
                f"마지막 저장 실패: {save_error}"
            )

        # 1) 내려받기: 재배포로 컨테이너가 초기화되어도
        #    이 파일만 있으면 그대로 복원할 수 있습니다.
        try:
            snapshot_bytes = RESULT_STORE.download_bytes()

        except Exception as error:
            snapshot_bytes = None
            st.warning(
                "저장 파일을 내려받지 못했습니다: "
                f"{type(error).__name__}"
            )

        if snapshot_bytes:
            file_stamp = str(
                saved_at_text or ""
            ).replace("-", "").replace(":", "").replace("T", "_")[:15]

            st.download_button(
                "결과 파일 내려받기 (.json.gz)",
                data=snapshot_bytes,
                file_name=(
                    f"last_analysis_{file_stamp}.json.gz"
                    if file_stamp
                    else "last_analysis.json.gz"
                ),
                mime="application/gzip",
                key="download_last_analysis",
                use_container_width=True,
            )

        else:
            st.caption(
                "아직 저장된 결과가 없습니다."
            )

        # 2) 올리기: 내려받은 파일로 복원
        uploaded = st.file_uploader(
            "결과 파일 올려서 복원",
            type=["gz", "json"],
            key="upload_last_analysis",
        )

        if uploaded is not None and st.button(
            "업로드한 결과 복원",
            key="restore_from_upload",
            use_container_width=True,
        ):
            try:
                payload = result_store.loads(
                    uploaded.getvalue()
                )

                restored = result_store.restore_snapshot(
                    payload,
                    image_dir=IMAGE_DIR,
                )

                apply_restored_result(restored)
                RESULT_STORE.save(payload)

                st.success(
                    "업로드한 결과를 복원했습니다."
                )
                st.rerun()

            except Exception as error:
                st.error(
                    "복원에 실패했습니다: "
                    f"{error}"
                )

        # 2-1) 저장 진단: 실제로 쓰고 읽어 확인합니다.
        if st.button(
            "저장 진단",
            key="diagnose_store",
            use_container_width=True,
            help="테스트 결과를 저장한 뒤 즉시 읽어, 저장소 권한·경로를 확인합니다.",
        ):
            try:
                probe_payload = result_store.build_snapshot(
                    results=[{
                        "stock_name": "저장진단",
                        "ticker": "000000",
                        "latest": {"종목명": "저장진단", "기준일": "2026-01-01",
                                   "종가": 1.0, "PER": None, "PER 분기": None,
                                   "Weekly 신호": "신호 없음",
                                   "Weekly+Monthly 신호": "신호 없음",
                                   "Weekly 신호일": None,
                                   "Weekly+Monthly 신호일": None,
                                   "신호 조회 일수": RECENT_SIGNAL_DAYS},
                        "messages": [],
                        "image_filename": None,
                        "png_bytes": None,
                        "weekly_signals": pd.DataFrame(),
                        "combined_signals": pd.DataFrame(),
                        "metrics": pd.DataFrame(),
                        "trades": {},
                    }],
                    records=[],
                    period=("2026-01-01", "2026-01-01"),
                    complete=False,
                    meta={
                        "recent_signal_days": RECENT_SIGNAL_DAYS,
                        "recent_signal_limit": RECENT_SIGNAL_LIMIT,
                    },
                    source="app",
                )

                RESULT_STORE.save(probe_payload)
                back = RESULT_STORE.load()
                version = (back or {}).get("version")

                st.success(
                    "저장소 정상 — 쓰기·읽기 모두 성공 "
                    f"(v{version}, {RESULT_STORE.describe()})"
                )

            except Exception as error:
                st.error(
                    f"저장 실패: {type(error).__name__}: {error}"
                )
                st.caption(
                    "backend=github 설정이면 결과는 배치가 커밋하며 "
                    "앱은 로컬에만 씁니다. "
                    "http 설정이면 URL이 PUT을 허용하는지 확인하세요."
                )

        # 3) 저장소에서 다시 읽기
        if st.button(
            "저장된 결과 다시 불러오기",
            key="reload_last_analysis",
            use_container_width=True,
        ):
            if load_last_analysis():
                st.rerun()

            else:
                st.info(
                    "불러올 저장 결과가 없습니다."
                )


# =========================================================
# 오른쪽: 분석 실행 및 결과
# =========================================================

render_page_heading(
    "종목별 분석 결과",
    st.session_state.get("last_saved_at"),
    st.session_state.get("result_source"),
)

restore_notice = st.session_state.pop(
    "restore_notice",
    None,
)

if restore_notice:
    st.info(restore_notice)

# 가장 위에 전체 결과 요약표 위치를 확보합니다.
overview_container = st.empty()
nomad_container = st.empty()
progress_container = st.container()
summary_container = st.container()
results_container = st.container()

batch_started = False

if submitted:
    if not selected_stocks:
        st.error(
            "분석할 종목을 하나 이상 선택하세요."
        )

    elif start > end:
        st.error(
            "시작일은 종료일보다 늦을 수 없습니다."
        )

    else:
        batch_started = True

        run_batch(
            selected_stocks=selected_stocks,
            start=start,
            end=end,
            include_investors=include_investors,
            overview_container=overview_container,
            progress_container=progress_container,
            results_container=results_container,
            nomad_container=nomad_container,
        )

records = st.session_state.get(
    "batch_records",
    [],
)

results = st.session_state.get(
    "batch_results",
    [],
)

period = st.session_state.get(
    "batch_period"
)

saved_at = st.session_state.get(
    "last_saved_at"
)

# =========================================================
# 자동 실행 결과 상태 패널
# =========================================================

def read_schedule_summary():
    """스케줄 설정 요약과 다음 실행 예정을 반환합니다."""
    try:
        schedule = schedule_config.load_config(
            BASE_DIR / "config" / "schedule.json"
        )

    except Exception:
        schedule = dict(schedule_config.DEFAULT_CONFIG)

    upcoming = schedule_config.next_runs(schedule, count=2)

    return schedule, upcoming


def render_run_status(results, saved_at, source):
    """
    '자동 실행 결과를 보고 있다'를 한눈에 알 수 있게 요약합니다.

    contributions
        - 언제 만들어진 결과인지(자동/수동)
        - 몇 종목·어느 기준일 데이터인지
        - 다음 자동 실행 예정 시각
        - 최신 결과를 다시 가져오는 버튼
    """
    if not saved_at and not results:
        return

    is_batch = str(source or "app") == "batch"

    with st.container(border=True):
        head_left, head_right = st.columns([3, 1])

        with head_left:
            badge = (
                "🤖 자동 실행 결과"
                if is_batch
                else "🖥 앱에서 실행한 결과"
            )
            st.markdown(f"#### {badge}")

            if is_batch:
                st.caption(
                    "예약된 배치가 생성한 결과입니다. "
                    "접속만 하면 항상 최신 저장본을 불러옵니다."
                )
            else:
                st.caption(
                    "이 브라우저에서 '분석 실행'으로 만든 결과입니다."
                )

        with head_right:
            if st.button(
                "🔄 최신 결과 확인",
                use_container_width=True,
                key="refresh_last_result",
            ):
                if load_last_analysis():
                    st.rerun()

                else:
                    st.toast(
                        "새로 불러올 저장 결과가 없습니다.",
                        icon="ℹ️",
                    )

        info_cols = st.columns(4)
        done = [row for row in results if row.get("latest")]

        basis_dates = sorted({
            str(row["latest"].get("기준일"))
            for row in done
            if row["latest"].get("기준일")
        })

        signal_count = sum(
            1 for row in done
            if row["latest"].get("Weekly+Monthly 신호") in ("매수", "매도")
        )

        info_cols[0].metric("종목 수", f"{len(results)}개")
        info_cols[1].metric("매수·매도 신호", f"{signal_count}개")
        info_cols[2].metric(
            "데이터 기준일",
            basis_dates[-1] if basis_dates else "-",
        )
        info_cols[3].metric(
            "저장 시각",
            str(saved_at).replace("T", " ")[:16] if saved_at else "-",
        )

        schedule, upcoming = read_schedule_summary()

        schedule_text = schedule_config.describe(schedule)

        if not schedule.get("enabled"):
            schedule_text += " · 일시 중지"

        st.caption(
            f"⏰ 다음 자동 실행: {schedule_text}"
            + (
                f" → {upcoming[0]:%m/%d %H:%M}"
                if upcoming
                else ""
            )
            + f" · 저장소: {RESULT_STORE.describe()}"
        )

        if is_batch and basis_dates:
            today = datetime.now(
                ZoneInfo("Asia/Seoul")
            ).date()

            try:
                gap = (
                    today - datetime.strptime(
                        basis_dates[-1], "%Y-%m-%d"
                    ).date()
                ).days

            except ValueError:
                gap = 0

            if gap >= 5:
                st.warning(
                    f"데이터 기준일이 {gap}일 전입니다. "
                    "휴장이거나 배치·데이터 수집에 문제가 있을 수 있습니다."
                )


render_run_status(
    results,
    st.session_state.get("last_saved_at"),
    st.session_state.get("result_source"),
)

# 저장 결과 복원 또는 화면 재실행 시 요약표 표시
if not batch_started:
    with overview_container.container():
        render_results_overview(
            results
        )

with summary_container:
    if records and not results:
        st.error(
            "표시할 분석 결과가 없습니다. "
            "실행 내역의 '상태'와 '내용'을 확인하세요."
        )

        st.info(
            "모든 종목이 '건너뜀'이면 "
            "file 폴더에 종목명.xlsx 파일이 "
            "있는지 확인하세요. "
            "등록 종목명, 엑셀 파일명, 시트명이 "
            "정확히 같아야 합니다."
        )

# 이번 분석에서 이미 표시한 상세 결과는 중복 표시하지 않습니다.
if not batch_started:
    with results_container:
        for position, result in enumerate(
            results,
            start=1,
        ):
            try:
                render_stock_result(
                    result,
                    position,
                )

            except Exception as error:
                st.error(
                    f'{result.get("stock_name", "종목")} '
                    f"결과 표시 실패: {error}"
                )

if not records and not results:
    with results_container:
        st.info(
            "왼쪽에서 그룹·종목·기간을 선택하고 "
            "'분석 실행'을 누르세요. "
            "분석이 끝난 종목부터 결과가 표시됩니다."
        )
