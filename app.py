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
LAST_RESULT_FILE = OUTPUT_DIR / "last_analysis.json"

TODAY_KST = datetime.now(ZoneInfo("Asia/Seoul")).date()
YESTERDAY_KST = TODAY_KST - timedelta(days=1)

# 기준일 포함 최근 7일: 거래일이 아닌 달력일입니다.
RECENT_SIGNAL_DAYS = 7

# 최근 매매 신호 표의 최대 행 수입니다.
RECENT_SIGNAL_LIMIT = 3

RESULT_SCHEMA = "png-v8-recent7-per-top3"
SAVED_RESULT_VERSION = 8

# 데이터 수집 코드 변경 전 캐시와 구분합니다.
DATA_CACHE_VERSION = "price-investor-only-v2"

WEEKDAYS = ("월", "화", "수", "목", "금", "토", "일")


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

def load_stock_groups(path):
    """
    stocks.txt 형식:

    [반도체]
    삼성전자,005930
    SK하이닉스,000660

    [자동차]
    현대차,005380
    """
    if not path.is_file():
        raise FileNotFoundError(
            f"등록 종목 파일을 찾을 수 없습니다: {path.name}"
        )

    groups = {}
    registered_codes = {}
    current_group = None

    text = path.read_text(encoding="utf-8-sig")

    for line_number, raw_line in enumerate(
        text.splitlines(),
        start=1,
    ):
        line = raw_line.strip()

        if not line or line.startswith("#"):
            continue

        if line.startswith("[") and line.endswith("]"):
            current_group = line[1:-1].strip()

            if not current_group:
                raise ValueError(
                    f"stocks.txt {line_number}행: "
                    "그룹명이 비어 있습니다."
                )

            groups.setdefault(current_group, [])
            continue

        if current_group is None:
            raise ValueError(
                f"stocks.txt {line_number}행: "
                "종목 위에 [그룹명]을 먼저 작성하세요."
            )

        fields = [
            value.strip()
            for value in line.split(",")
        ]

        if len(fields) != 2:
            raise ValueError(
                f"stocks.txt {line_number}행: "
                "'종목명,종목코드' 형식이어야 합니다."
            )

        stock_name, ticker = fields
        invalid_chars = '<>:"/\\|?*'

        if (
            not stock_name
            or stock_name in {".", ".."}
            or any(
                char in stock_name
                for char in invalid_chars
            )
            or any(
                ord(char) < 32
                for char in stock_name
            )
            or stock_name.endswith((".", " "))
        ):
            raise ValueError(
                f"stocks.txt {line_number}행: "
                "파일명으로 사용할 수 없는 종목명입니다."
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

        if (
            previous_code is not None
            and previous_code != ticker
        ):
            raise ValueError(
                f"stocks.txt {line_number}행: "
                f"'{stock_name}'에 서로 다른 "
                "종목코드가 등록되었습니다."
            )

        registered_codes[stock_name] = ticker
        stock_item = (stock_name, ticker)

        if stock_item not in groups[current_group]:
            groups[current_group].append(stock_item)

    groups = {
        group_name: stocks
        for group_name, stocks in groups.items()
        if stocks
    }

    if not groups:
        raise ValueError(
            "stocks.txt에 등록된 종목이 없습니다."
        )

    return groups


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
    파일 변경 시 modified_ns/file_size가 달라져 다시 읽습니다.

    필수 컬럼:
        분기, 상장주식수, 4Q_income

    단위:
        상장주식수: 주
        4Q_income: 억원
    """
    with pd.ExcelFile(excel_path) as workbook:
        if stock_name not in workbook.sheet_names:
            raise ValueError(
                f"엑셀에 '{stock_name}' 시트가 없습니다."
            )

        frame = pd.read_excel(
            workbook,
            sheet_name=stock_name,
        )

    frame = frame.dropna(how="all").copy()
    frame.columns = [
        str(column).strip()
        for column in frame.columns
    ]

    if frame.columns.duplicated().any():
        raise ValueError(
            "PER 재무 데이터에 중복된 컬럼이 있습니다."
        )

    required = {
        "분기",
        "상장주식수",
        "4Q_income",
    }

    missing = required - set(frame.columns)

    if missing:
        raise ValueError(
                "PER 계산에 필요한 엑셀 컬럼이 없습니다: "
                + ", ".join(sorted(missing))
        )

    frame["분기"] = (
        frame["분기"]
        .astype("string")
        .str.strip()
    )

    valid_quarter = frame["분기"].str.fullmatch(
        r"\d{4}\.[1-4]Q",
        na=False,
    )

    if not valid_quarter.all():
        raise ValueError(
            "PER 재무 데이터의 분기는 "
            "'2026.1Q' 형식이어야 합니다."
        )

    if frame["분기"].duplicated().any():
        raise ValueError(
            "PER 재무 데이터에 중복된 분기가 있습니다."
        )

    quarter_ends = []

    for quarter_label in frame["분기"]:
        matched = re.fullmatch(
            r"(\d{4})\.([1-4])Q",
            quarter_label,
        )

        year = int(matched.group(1))
        quarter = int(matched.group(2))

        quarter_end = pd.Period(
            f"{year}Q{quarter}",
            freq="Q-DEC",
        ).end_time.normalize()

        quarter_ends.append(quarter_end)

    frame["_분기종료일"] = pd.to_datetime(
        quarter_ends
    )

    for column in ("상장주식수", "4Q_income"):
        frame[column] = frame[column].map(
            to_finite_number
        )

    return (
        frame
        .sort_values("_분기종료일")
        .reset_index(drop=True)
    )


def build_per_status(excel_path, stock_name, latest):
    """
    가격 기준일까지 종료된 분기 중 가장 최근 행을 사용합니다.

    PER =
        종가 × 상장주식수
        ÷ (4Q_income × 100,000,000)

    가장 최근 종료 분기 값이 누락된 경우,
    더 오래된 분기로 임의 대체하지 않습니다.
    """
    status = {
        "PER": None,
        "PER 분기": None,
        "PER 상장주식수": None,
        "PER 4Q_income": None,
        "PER 설명": "",
    }

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

    completed = financial.loc[
        financial["_분기종료일"] <= reference
    ]

    if completed.empty:
        status["PER 설명"] = (
            "기준일까지 종료된 분기 데이터가 없습니다."
        )
        return status

    row = completed.iloc[-1]

    shares = to_finite_number(
        row["상장주식수"]
    )
    income = to_finite_number(
        row["4Q_income"]
    )

    status.update({
        "PER 분기": str(row["분기"]),
        "PER 상장주식수": shares,
        "PER 4Q_income": income,
    })

    if shares is None or shares <= 0:
        status["PER 설명"] = (
            f'{row["분기"]}: '
            "상장주식수가 없거나 0 이하입니다."
        )
        return status

    if not shares.is_integer():
        status["PER 설명"] = (
            f'{row["분기"]}: '
            "상장주식수는 주 단위 정수여야 합니다."
        )
        return status

    if income is None or income <= 0:
        status["PER 설명"] = (
            f'{row["분기"]}: '
            "4Q_income이 없거나 0 이하입니다."
        )
        return status

    per_value = (
        close_price
        * shares
        / (income * 100_000_000)
    )

    if not math.isfinite(per_value):
        status["PER 설명"] = (
            "PER 계산 결과가 유효하지 않습니다."
        )
        return status

    status["PER"] = per_value
    status["PER 설명"] = (
        f'{row["분기"]} 엑셀 값 기준 · '
        "4Q_income 단위: 억원"
    )

    return status


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

        rows.append(
            '<tr>'
            f'<td>{name_link}</td>'
            f'<td>{weekly_link}</td>'
            f'<td>{combined_link}</td>'
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
        '</style>'
        '<div style="overflow-x:auto;">'
        '<table class="stock-overview-table">'
        '<thead><tr>'
        '<th>종목</th>'
        f'<th>주간 신호 · 최근 {RECENT_SIGNAL_DAYS}일</th>'
        '<th>주간·월간 결합 신호 · '
        f'최근 {RECENT_SIGNAL_DAYS}일</th>'
        '</tr></thead><tbody>'
        + "".join(rows)
        + '</tbody></table></div>'
    )

    st.markdown(
        html,
        unsafe_allow_html=True,
    )

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
    # 첫 번째 줄: 최근 종가 / PER
    # -----------------------------------------------------

    price_column, per_column = st.columns(2)

    with price_column:
        close_price = to_finite_number(
            latest.get("종가")
        )

        st.metric(
            "최근 종가",
            (
                f"{close_price:,.0f}원"
                if close_price is not None
                else "—"
            ),
        )

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

    # -----------------------------------------------------
    # 두 번째 줄: 최근 7일 매매 신호
    # -----------------------------------------------------

    weekly_column, combined_column = st.columns(2)

    with weekly_column:
        render_signal_card(
            title=(
                "주간 신호 · "
                f"최근 {RECENT_SIGNAL_DAYS}일"
            ),
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
            title=(
                "주간·월간 결합 신호 · "
                f"최근 {RECENT_SIGNAL_DAYS}일"
            ),
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

        image_filename = result.get(
            "image_filename"
        )

        st.download_button(
            f"{name} 고해상도 차트 다운로드",
            data=png_bytes,
            file_name=(
                image_filename
                or f"{name}_chart.png"
            ),
            mime="image/png",
            key=f"chart_{position}_{ticker}",
        )

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
):
    if not results:
        return

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    saved_results = []

    for result in results:
        png_bytes = result.get("png_bytes")

        saved_results.append({
            "stock_name": result["stock_name"],
            "ticker": result["ticker"],
            "latest": result["latest"],
            "messages": result.get(
                "messages",
                [],
            ),
            "image_filename": result.get(
                "image_filename"
            ),
            "png_base64": (
                base64.b64encode(
                    png_bytes
                ).decode("ascii")
                if png_bytes
                else None
            ),
            "weekly_signals": dataframe_to_json_data(
                result["weekly_signals"]
            ),
            "combined_signals": dataframe_to_json_data(
                result["combined_signals"]
            ),
            "metrics": dataframe_to_json_data(
                result["metrics"]
            ),
            "trades": {
                strategy: dataframe_to_json_data(frame)
                for strategy, frame
                in result["trades"].items()
            },
        })

    saved_at = datetime.now(
        ZoneInfo("Asia/Seoul")
    ).isoformat(
        timespec="seconds"
    )

    payload = {
        "version": SAVED_RESULT_VERSION,
        "recent_signal_days": RECENT_SIGNAL_DAYS,
        "recent_signal_limit": RECENT_SIGNAL_LIMIT,
        "saved_at": saved_at,
        "period": list(period),
        "records": records,
        "results": saved_results,
    }

    temporary_path = (
        OUTPUT_DIR
        / f".last_analysis_{uuid.uuid4().hex}.tmp"
    )

    try:
        temporary_path.write_text(
            json.dumps(
                json_safe(payload),
                ensure_ascii=False,
                allow_nan=False,
            ),
            encoding="utf-8",
        )

        os.replace(
            temporary_path,
            LAST_RESULT_FILE,
        )

    finally:
        temporary_path.unlink(
            missing_ok=True
        )

    st.session_state["last_saved_at"] = saved_at


# =========================================================
# 마지막 결과 복원
# =========================================================

def load_last_analysis():
    if not LAST_RESULT_FILE.is_file():
        return False

    payload = json.loads(
        LAST_RESULT_FILE.read_text(
            encoding="utf-8"
        )
    )

    if (
        payload.get("version")
        != SAVED_RESULT_VERSION
        or payload.get("recent_signal_days")
        != RECENT_SIGNAL_DAYS
        or payload.get("recent_signal_limit")
        != RECENT_SIGNAL_LIMIT
    ):
        st.session_state["restore_notice"] = (
            "이전 형식의 저장 결과가 있습니다. "
            "PER·최근 7일 신호·최근 3건 표시를 "
            "적용하려면 '분석 실행'을 다시 눌러주세요."
        )
        return False

    restored = []

    for item in payload.get("results", []):
        latest = item["latest"]

        required = {
            "Weekly 신호",
            "Weekly+Monthly 신호",
            "Weekly 신호일",
            "Weekly+Monthly 신호일",
            "신호 조회 일수",
            "PER",
            "PER 분기",
        }

        if (
            not required.issubset(latest)
            or latest["신호 조회 일수"]
            != RECENT_SIGNAL_DAYS
        ):
            st.session_state["restore_notice"] = (
                "저장 결과에 최신 신호 또는 PER 정보가 "
                "없습니다. '분석 실행'을 다시 눌러주세요."
            )
            return False

        encoded_png = item.get(
            "png_base64"
        )

        restored.append({
            "stock_name": item["stock_name"],
            "ticker": item["ticker"],
            "latest": latest,
            "messages": item.get(
                "messages",
                [],
            ),
            "image_filename": item.get(
                "image_filename"
            ),
            "png_bytes": (
                base64.b64decode(
                    encoded_png,
                    validate=True,
                )
                if encoded_png
                else None
            ),
            "weekly_signals": json_data_to_dataframe(
                item["weekly_signals"]
            ),
            "combined_signals": json_data_to_dataframe(
                item["combined_signals"]
            ),
            "metrics": json_data_to_dataframe(
                item["metrics"]
            ),
            "trades": {
                strategy: json_data_to_dataframe(frame_data)
                for strategy, frame_data
                in item["trades"].items()
            },
        })

    if not restored:
        return False

    st.session_state["batch_results"] = restored

    st.session_state["batch_records"] = (
        payload.get("records", [])
    )

    st.session_state["batch_period"] = (
        payload.get("period")
    )

    st.session_state["last_saved_at"] = (
        payload.get("saved_at")
    )

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

        if not excel_path.is_file():
            records.append({
                "종목명": stock_name,
                "종목코드": ticker,
                "상태": "건너뜀",
                "내용": (
                    "재무 파일 없음: "
                    f"file/{stock_name}.xlsx"
                ),
            })

            st.session_state["batch_records"] = records

            progress.progress(
                index / total
            )
            continue

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
                excel_file=str(excel_path),
            )

            # 엔진 내부 기본값과 관계없이
            # app.py에서 최근 7일 신호를 다시 계산합니다.
            latest = build_recent_signal_status(
                data=analysis["data"],
                latest=analysis["latest"],
            )

            # PER은 원본 엑셀에서 별도로 읽습니다.
            latest.update(
                build_per_status(
                    excel_path=excel_path,
                    stock_name=stock_name,
                    latest=latest,
                )
            )

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
                excel_file=str(excel_path),
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
        try:
            save_last_analysis(
                results=results,
                records=records,
                period=(
                    start.isoformat(),
                    end.isoformat(),
                ),
            )

        except Exception as error:
            with progress_container:
                st.warning(
                    "분석은 완료했으나 결과 저장에 "
                    f"실패했습니다: {error}"
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
    stock_groups = load_stock_groups(
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

def save_group_section(path, group_name, items):
    """
    stocks.txt의 [그룹명] 섹션만 교체하고
    나머지 그룹과 주석은 그대로 유지합니다.

    항목 형식: (종목명, 종목코드) 튜플 목록
    """
    lines = path.read_text(
        encoding="utf-8-sig"
    ).splitlines()

    section_header = f"[{group_name}]"

    new_lines = []
    in_section = False
    section_written = False

    for line in lines:
        stripped = line.strip()

        if stripped.startswith("[") and stripped.endswith("]"):
            if in_section and not section_written:
                for name, ticker in items:
                    new_lines.append(f"{name},{ticker}")
                section_written = True

            in_section = stripped == section_header
            new_lines.append(line)
            continue

        if in_section:
            continue

        new_lines.append(line)

    if in_section and not section_written:
        for name, ticker in items:
            new_lines.append(f"{name},{ticker}")
        section_written = True

    if not section_written:
        new_lines.extend(
            ["", section_header]
            + [f"{name},{ticker}" for name, ticker in items]
        )

    path.write_text(
        "\n".join(new_lines) + "\n",
        encoding="utf-8",
    )


def save_watch_group(path, watch_stocks):
    """[관심] 섹션 교체 — save_group_section의 래퍼입니다."""
    save_group_section(
        path,
        "관심",
        watch_stocks,
    )


def render_group_reorder(group_name, items, key_prefix=""):
    """
    그룹 종목의 순서를 ↑ / ↓ 버튼으로 바꾸고
    즉시 stocks.txt에 저장합니다.

    key_prefix: 같은 그룹을 화면에 여러 번 그릴 때
    위젯 키가 겹치지 않도록 호출 위치별로 다르게 지정합니다.
    """
    if not items:
        st.info(f"[{group_name}] 그룹이 비어 있습니다.")
        return

    safe_group = re.sub(
        r"\W+",
        "_",
        group_name,
    )

    key_prefix = f"{key_prefix}_" if key_prefix else ""

    for order_index, item in enumerate(items):
        name, ticker = item

        left, middle, right = st.columns([5, 1, 1])

        with left:
            st.markdown(
                f'<div style="padding:6px 0;'
                'font-weight:600;">'
                f'{escape(str(order_index + 1))}. '
                f'{escape(name)} '
                '<span style="color:#808080;'
                'font-size:13px;font-weight:400;">'
                f'({escape(ticker)})'
                '</span></div>',
                unsafe_allow_html=True,
            )

        with middle:
            if st.button(
                "위",
                key=f"{key_prefix}reorder_{safe_group}_up_{order_index}",
                disabled=(order_index == 0),
                use_container_width=True,
            ):
                reordered = list(items)
                reordered[order_index - 1], reordered[order_index] = (
                    reordered[order_index],
                    reordered[order_index - 1],
                )

                save_group_section(
                    STOCKS_FILE,
                    group_name,
                    reordered,
                )

                st.rerun()

        with right:
            if st.button(
                "아래",
                key=f"{key_prefix}reorder_{safe_group}_down_{order_index}",
                disabled=(order_index == len(items) - 1),
                use_container_width=True,
            ):
                reordered = list(items)
                reordered[order_index + 1], reordered[order_index] = (
                    reordered[order_index],
                    reordered[order_index + 1],
                )

                save_group_section(
                    STOCKS_FILE,
                    group_name,
                    reordered,
                )

                st.rerun()


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
    # 관심 그룹 편집: 다른 그룹 종목 추가 / 기존 종목 삭제
    # -----------------------------------------------------

    with st.expander("관심 그룹 편집"):
        watch_stocks = stock_groups.get("관심", [])

        other_stocks = []
        other_seen = set()

        for group_name, items in stock_groups.items():
            if group_name == "관심":
                continue

            for item in items:
                if item in other_seen:
                    continue

                if item in watch_stocks:
                    continue

                other_seen.add(item)
                other_stocks.append(item)

        add_choices = st.multiselect(
            "추가할 종목",
            options=other_stocks,
            format_func=lambda item: (
                f"{item[0]} ({item[1]})"
            ),
        )


        remove_choices = st.multiselect(
            "삭제할 종목",
            options=watch_stocks,
            format_func=lambda item: (
                f"{item[0]} ({item[1]})"
            ),
        )

        # -------------------------------------------------
        # 순서 변경: ↑ / ↓ 버튼으로 즉시 저장합니다.
        # -------------------------------------------------

        st.markdown("**순서 변경**")

        render_group_reorder(
            "관심",
            watch_stocks,
            key_prefix="watch_editor",
        )

        if st.button("관심 그룹 저장"):
            updated_watch = [
                item
                for item in watch_stocks
                if item not in remove_choices
            ] + [
                item
                for item in add_choices
                if item not in watch_stocks
            ]

            save_watch_group(
                STOCKS_FILE,
                updated_watch,
            )

            # stocks.txt에 저장되므로 다시 접속해도
            # 마지막 저장 상태가 유지됩니다.
            st.toast(
                "관심 그룹을 stocks.txt에 저장했습니다. "
                "다음 접속부터 저장된 구성이 유지됩니다.",
                icon="💾",
            )

            st.rerun()

    # -----------------------------------------------------
    # 다른 그룹 순서 변경: 보유·매매 후보 등 모든 그룹 지원
    # -----------------------------------------------------

    with st.expander("다른 그룹 순서 변경"):
        reorder_group = st.selectbox(
            "그룹 선택",
            options=list(stock_groups),
        )

        render_group_reorder(
            reorder_group,
            stock_groups.get(reorder_group, []),
            key_prefix="other_groups",
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
            max_value=YESTERDAY_KST,
        )

        end = st.date_input(
            "종료일",
            value=YESTERDAY_KST,
            max_value=YESTERDAY_KST,
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


# =========================================================
# 오른쪽: 분석 실행 및 결과
# =========================================================

st.title("종목별 분석 결과")

restore_notice = st.session_state.pop(
    "restore_notice",
    None,
)

if restore_notice:
    st.info(restore_notice)

# 가장 위에 전체 결과 요약표 위치를 확보합니다.
overview_container = st.empty()
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

# 저장 결과 복원 또는 화면 재실행 시 요약표 표시
if not batch_started:
    with overview_container.container():
        render_results_overview(
            results
        )

with summary_container:
    if records:
        st.markdown(
            "### 종목별 실행 내역"
        )

        st.dataframe(
            pd.DataFrame(records),
            hide_index=True,
            use_container_width=True,
        )

        if not results:
            st.error(
                "표시할 분석 결과가 없습니다. "
                "실행 내역의 '상태'와 "
                "'내용'을 확인하세요."
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
