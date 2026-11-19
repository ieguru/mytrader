# pages/1_groups.py
"""
그룹(관심·반도체·태양광 …) 편집 화면.

stocks.txt를 직접 열지 않아도 되게 만든 페이지입니다.
- 관심 그룹에 다른 그룹 종목을 추가/삭제
- 그룹 안 종목 순서 변경(↑/↓)
- 그룹 자체의 표시 순서 변경(↑/↓)
- 파일 백업·복원

저장은 stocks.txt의 해당 [섹션]만 교체합니다.
주석과 다른 그룹은 손대지 않습니다.
"""

from pathlib import Path

import pandas as pd
import streamlit as st

import stock_groups

BASE_DIR = Path(__file__).resolve().parent.parent
STOCKS_FILE = BASE_DIR / "stocks.txt"


def load_groups():
    try:
        return stock_groups.load_stock_groups(STOCKS_FILE)

    except Exception as error:
        st.error(f"stocks.txt를 읽지 못했습니다: {error}")
        st.stop()


st.set_page_config(
    page_title="그룹 편집",
    page_icon="🗂",
    layout="centered",
)

st.title("🗂 그룹 편집")
st.caption(
    "stocks.txt를 직접 열지 않고 그룹 구성을 관리합니다. "
    "저장하면 즉시 파일에 반영되고, 다음 분석부터 새 구성이 적용됩니다."
)

st.markdown(
    """
**이 화면에서 하는 일**

| 항목 | 설명 |
|---|---|
| 관심 그룹 편집 | 다른 그룹에 있는 종목을 관심 그룹에 담거나 뺍니다 |
| 종목 순서 변경 | 그룹 안에서 종목이 표시되는 순서를 바꿉니다 |
| 그룹 순서 변경 | 분석 화면·선택 목록에 그룹이 나오는 순서를 바꿉니다 |

저장 방식은 **해당 `[그룹명]` 섹션만 교체**하는 방식입니다.
파일 맨 위 주석과 다른 그룹은 그대로 유지되며,
잘못 저장했을 때를 대비해 아래에서 **백업 파일**을 내려받을 수 있습니다.
"""
)

groups = load_groups()

# ---------------------------------------------------------
# 현재 구성
# ---------------------------------------------------------

st.divider()
st.markdown("### 현재 구성")

summary = pd.DataFrame([
    {
        "그룹": name,
        "종목 수": len(items),
        "종목": ", ".join(n for n, _ in items),
    }
    for name, items in groups.items()
])

st.dataframe(
    summary,
    hide_index=True,
    use_container_width=True,
    column_config={
        "종목": st.column_config.TextColumn(width="large"),
    },
)

# ---------------------------------------------------------
# 관심 그룹 편집
# ---------------------------------------------------------

st.divider()
st.markdown("### 관심 그룹 편집")

st.caption(
    "관심 그룹은 매매 후보를 모아 두는 목록입니다. "
    "다른 그룹에만 있는 종목을 여기로 옮겨 담을 수 있습니다."
)

watch_stocks = list(groups.get("관심", []))

if not watch_stocks:
    st.info("관심 그룹이 비어 있습니다. 아래에서 종목을 추가하세요.")

other_stocks = []
seen = set()

for group_name, items in groups.items():
    if group_name == "관심":
        continue

    for item in items:
        if item in seen or item in watch_stocks:
            continue

        seen.add(item)
        other_stocks.append(item)

label_of = lambda item: f"{item[0]} ({item[1]})"  # noqa: E731

add_choices = st.multiselect(
    "추가할 종목",
    options=other_stocks,
    format_func=label_of,
    key="watch_add",
    help="다른 그룹에 있는 종목 중 관심 그룹에 담을 종목을 고릅니다.",
)

remove_choices = st.multiselect(
    "삭제할 종목",
    options=watch_stocks,
    format_func=label_of,
    key="watch_remove",
    help="관심 그룹에서 뺄 종목을 고릅니다.",
)

st.markdown("**관심 그룹 종목 순서**")
st.caption("↑ / ↓ 를 누르면 즉시 저장됩니다.")

stock_groups.render_group_reorder(
    "관심",
    watch_stocks,
    key_prefix="watch_page",
)

if st.button("관심 그룹 저장", type="primary", use_container_width=True):
    updated = [
        item for item in watch_stocks if item not in remove_choices
    ] + [
        item for item in add_choices if item not in watch_stocks
    ]

    try:
        stock_groups.save_watch_group(STOCKS_FILE, updated)
        st.success(
            f"관심 그룹을 {len(updated)}종목으로 저장했습니다."
        )
        st.rerun()

    except OSError as error:
        st.error(f"저장하지 못했습니다: {error}")

# ---------------------------------------------------------
# 그룹 안 종목 순서
# ---------------------------------------------------------

st.divider()
st.markdown("### 그룹 안 종목 순서 변경")

reorder_group = st.selectbox(
    "그룹 선택",
    options=list(groups),
    key="reorder_group_select",
)

st.caption(
    f"[{reorder_group}] 의 종목 순서입니다. "
    "↑ / ↓ 를 누르면 바로 저장됩니다."
)

stock_groups.render_group_reorder(
    reorder_group,
    groups.get(reorder_group, []),
    key_prefix="group_page",
)

# ---------------------------------------------------------
# 그룹 표시 순서
# ---------------------------------------------------------

st.divider()
st.markdown("### 그룹 표시 순서 변경")

st.caption(
    "사이드바 '분석할 그룹' 목록에 보이는 그룹 순서입니다. "
    "↑ / ↓ 를 누르면 바로 저장됩니다."
)

stock_groups.render_group_order(
    STOCKS_FILE,
    key_prefix="group_order_page",
)

# ---------------------------------------------------------
# 백업 / 복원
# ---------------------------------------------------------

st.divider()

with st.expander("백업 · 복원"):
    st.caption(
        "Streamlit Cloud 컨테이너는 재배포 시 초기화됩니다. "
        "구성을 확실히 남기려면 내려받은 stocks.txt를 "
        "저장소에 커밋하세요."
    )

    if STOCKS_FILE.is_file():
        st.download_button(
            "stocks.txt 내려받기",
            data=STOCKS_FILE.read_bytes(),
            file_name="stocks.txt",
            mime="text/plain",
            use_container_width=True,
        )

    uploaded = st.file_uploader(
        "stocks.txt 올려서 복원",
        type=["txt"],
        key="upload_stocks",
    )

    if uploaded is not None and st.button(
        "업로드한 구성 적용",
        use_container_width=True,
    ):
        raw = uploaded.getvalue()
        temporary = STOCKS_FILE.with_suffix(".upload.tmp")

        try:
            temporary.write_bytes(raw)
            parsed = stock_groups.load_stock_groups(temporary)

            if not parsed:
                raise ValueError("등록된 그룹이 없습니다.")

            STOCKS_FILE.write_bytes(raw)
            st.success(
                f"{len(parsed)}개 그룹으로 복원했습니다."
            )
            st.rerun()

        except Exception as error:
            st.error(f"적용하지 못했습니다: {error}")

        finally:
            temporary.unlink(missing_ok=True)
