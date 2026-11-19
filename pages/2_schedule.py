# pages/2_schedule.py
"""
스케줄 설정 페이지 — 주기(매시간/매일/매주)와 시각을 정합니다.

저장한 설정은 config/schedule.json 에 기록되고,
워크플로(.github/workflows/daily_analysis.yml)가 매시간 이 파일을 읽어
실행 여부를 판정합니다(schedule_gate.py).
"""

import json
from pathlib import Path
from zoneinfo import ZoneInfo

import streamlit as st

import schedule_config as sc

BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = BASE_DIR / "config" / "schedule.json"
IMAGE_DIR = BASE_DIR / "img"


def format_moment(value):
    return (
        f"{value:%Y-%m-%d}({sc.WEEKDAY_NAMES[value.weekday()]}) "
        f"{value:%H:%M}"
    )


st.set_page_config(
    page_title="스케줄 설정",
    page_icon="⏰",
    layout="centered",
)

st.title("⏰ 스케줄 설정")
st.caption(
    "분석 배치가 언제 얼마나 자주 실행될지 정합니다. "
    "저장 즉시 config/schedule.json 이 갱신됩니다."
)

# ---------------------------------------------------------
# 현재 설정
# ---------------------------------------------------------

try:
    config = sc.load_config(CONFIG_PATH)

except sc.ScheduleConfigError as error:
    st.error(f"설정 파일을 읽지 못했습니다: {error}")
    config = dict(sc.DEFAULT_CONFIG)

st.markdown(
    f"**현재 설정** · {sc.describe(config)} · "
    f"{'알림 사용' if config['notify'] else '알림 끔'} · "
    f"{'사용 중' if config['enabled'] else '일시 중지'}"
)

if config.get("updated_at"):
    st.caption(f"마지막 저장: {config['updated_at']}")

# ---------------------------------------------------------
# 입력 폼
# ---------------------------------------------------------

st.markdown("#### 주기")

# 폼 밖에 두어 주기를 바꾸면 즉시 '요일' 선택 가능 여부가 갱신됩니다.
period = st.radio(
    "얼마나 자주 실행할까요?",
    options=sc.PERIODS,
    index=sc.PERIODS.index(config["period"]),
    format_func=lambda item: sc.PERIOD_LABELS[item],
    horizontal=True,
)

with st.form("schedule_form"):
    st.markdown("#### 시각")

    hour_col, minute_col, weekday_col = st.columns(3)

    with hour_col:
        hour = st.selectbox(
            "시",
            options=list(range(24)),
            index=config["hour"],
            format_func=lambda value: f"{value:02d}시",
        )

    with minute_col:
        minute = st.selectbox(
            "분",
            options=list(range(60)),
            index=config["minute"],
            format_func=lambda value: f"{value:02d}분",
        )

    with weekday_col:
        weekday = st.selectbox(
            "요일 (매주일 때)",
            options=list(range(7)),
            index=config["weekday"],
            format_func=lambda value: f"{sc.WEEKDAY_NAMES[value]}요일",
            disabled=(period != "weekly"),
        )

    st.markdown("#### 옵션")

    notify = st.checkbox(
        "텔레그램으로 결과 알림 받기",
        value=config["notify"],
    )

    enabled = st.checkbox(
        "스케줄 사용 (끄면 자동 실행을 멈춥니다)",
        value=config["enabled"],
    )

    submitted = st.form_submit_button(
        "저장",
        use_container_width=True,
        type="primary",
    )

# ---------------------------------------------------------
# 미리보기 (저장 전에도 즉시 갱신)
# ---------------------------------------------------------

preview = sc.normalize_config({
    **config,
    "period": period,
    "hour": hour,
    "minute": minute,
    "weekday": weekday,
    "notify": notify,
    "enabled": enabled,
})

if submitted:
    try:
        saved = sc.save_config(CONFIG_PATH, preview)
        st.success(f"저장했습니다 — {sc.describe(saved)}")
        config = saved

    except (sc.ScheduleConfigError, OSError) as error:
        st.error(f"저장하지 못했습니다: {error}")

st.divider()

st.markdown("### 적용 결과")

summary_col, cron_col = st.columns([3, 2])

with summary_col:
    st.markdown(f"**주기** · {sc.describe(preview)}")
    st.markdown(
        f"**알림** · {'텔레그램 사용' if preview['notify'] else '사용 안 함'}"
    )

    upcoming = sc.next_runs(preview, count=3)

    if upcoming:
        st.markdown("**다음 실행 예정 (KST)**")

        for index, moment in enumerate(upcoming, start=1):
            st.markdown(f"{index}. {format_moment(moment)}")

    elif not preview["enabled"]:
        st.info("스케줄이 꺼져 있어 자동 실행되지 않습니다.")

with cron_col:
    st.markdown("**고정 cron을 쓸 경우 (UTC)**")
    st.code(sc.cron_expression(preview), language="text")
    st.caption(sc.utc_note(preview))

if preview["period"] == "hourly":
    st.warning(
        "매시간 주기는 하루 24회 수집합니다. "
        "pykrx 호출 제한과 GitHub Actions 사용량이 커지니 "
        "장중 감시가 필요할 때만 쓰세요. "
        "또한 매시간 판정이므로 '분' 값은 고정 cron을 쓸 때만 의미가 있습니다."
    )

if preview["period"] in {"daily", "weekly"}:
    st.info(
        "GitHub cron은 수 분~수십 분 지연될 수 있어, "
        f"{preview['hour']:02d}시대(00~59분) 안에 한 번 실행되면 정상입니다."
    )

# ---------------------------------------------------------
# 백업 / 복원
# ---------------------------------------------------------

st.divider()

with st.expander("설정 백업·복원"):
    st.caption(
        "Streamlit Cloud 컨테이너는 재배포 시 초기화됩니다. "
        "설정을 유지하려면 이 파일을 저장소에 커밋하세요 "
        "(config/schedule.json)."
    )

    if CONFIG_PATH.is_file():
        st.download_button(
            "설정 파일 내려받기 (schedule.json)",
            data=CONFIG_PATH.read_bytes(),
            file_name="schedule.json",
            mime="application/json",
            use_container_width=True,
        )

    uploaded = st.file_uploader(
        "설정 파일 올리기",
        type=["json"],
        key="upload_schedule",
    )

    if uploaded is not None and st.button(
        "업로드한 설정 적용",
        use_container_width=True,
    ):
        try:
            raw = json.loads(uploaded.getvalue().decode("utf-8"))
            saved = sc.save_config(CONFIG_PATH, raw)
            st.success(f"적용했습니다 — {sc.describe(saved)}")
            st.rerun()

        except (ValueError, OSError, sc.ScheduleConfigError) as error:
            st.error(f"적용하지 못했습니다: {error}")

# ---------------------------------------------------------
# 워크플로 연동 상태
# ---------------------------------------------------------

st.divider()

workflow = BASE_DIR / ".github" / "workflows" / "daily_analysis.yml"

if workflow.is_file():
    text = workflow.read_text(encoding="utf-8")
    gate_mode = "schedule_gate.py" in text

    st.markdown("### 워크플로 연결 상태")

    if gate_mode:
        st.success(
            "게이트 방식으로 연결되어 있습니다. "
            "워크플로가 매시간 이 설정을 확인하고, "
            "실행 시각이면 분석을 시작합니다."
        )
    else:
        st.warning(
            "워크플로가 고정 cron으로 실행되고 있습니다. "
            "위 cron 값을 .github/workflows/daily_analysis.yml 에 "
            "직접 반영하거나, 게이트 방식(schedule_gate.py)으로 바꿔 주세요."
        )

    state_path = BASE_DIR / "output" / "schedule_state.json"

    if state_path.is_file():
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
            st.caption(
                f"마지막 실행 슬롯: {state.get('last_slot', '-')} "
                f"({state.get('saved_at', '-')})"
            )

        except ValueError:
            pass

else:
    st.caption("워크플로 파일을 찾지 못했습니다.")
