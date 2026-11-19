# stock_groups.py
"""
stocks.txt(그룹별 종목 목록)의 읽기·편집을 담당합니다.

파일 형식
    # 주석
    [그룹명]
    종목명,종목코드

원칙
- 그룹 하나를 저장할 때 그 [섹션]만 교체하고 나머지 그룹·주석은 그대로 둡니다.
- 그룹 블록 순서 변경은 그룹 집합이 그대로일 때만 수행합니다.
- 쓸 때는 임시 파일에 기록 후 교체하여, 중간 실패에도 원본이 남습니다.
"""

from __future__ import annotations

import os
import re
import uuid
from html import escape
from pathlib import Path

import pandas as pd
import streamlit as st


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

# =========================================================
# 그룹 블록 순서 변경
# =========================================================

def split_blocks(path):
    """stocks.txt를 (머리말, 그룹 블록 목록)으로 나눕니다."""
    lines = Path(path).read_text(encoding="utf-8-sig").splitlines()

    preamble, blocks, current = [], [], None

    for line in lines:
        stripped = line.strip()

        if stripped.startswith("[") and stripped.endswith("]"):
            current = {"name": stripped[1:-1].strip(), "lines": [line]}
            blocks.append(current)
            continue

        if current is None:
            preamble.append(line)
        else:
            current["lines"].append(line)

    return preamble, blocks


def save_group_order(path, ordered_names):
    """그룹 블록의 순서만 바꿉니다(집합이 같을 때만)."""
    path = Path(path)
    preamble, blocks = split_blocks(path)

    current = [block["name"] for block in blocks]
    wanted = list(ordered_names)

    if sorted(current) != sorted(wanted):
        raise ValueError(
            "그룹 목록이 파일과 다릅니다. 새로고침 후 다시 시도하세요."
        )

    by_name = {block["name"]: block for block in blocks}
    new_lines = list(preamble)

    for name in wanted:
        new_lines.extend(by_name[name]["lines"])

    text = "\n".join(line.rstrip() for line in new_lines).rstrip() + "\n"
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"

    try:
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)

    return wanted


def render_group_order(path, key_prefix="group_order"):
    """그룹 블록 자체의 순서를 ↑/↓ 로 바꿉니다."""
    try:
        _, blocks = split_blocks(path)

    except OSError as error:
        st.error(f"stocks.txt를 읽지 못했습니다: {error}")
        return

    names = [block["name"] for block in blocks]

    if len(names) < 2:
        st.info("순서를 바꿀 그룹이 2개 이상 필요합니다.")
        return

    st.caption("  →  ".join(names))

    for index, name in enumerate(names):
        row = st.columns([5, 1, 1])
        count = sum(
            1 for line in blocks[index]["lines"][1:]
            if line.strip() and not line.strip().startswith("#")
        )
        row[0].markdown(f"**{name}** · {count}종목")

        if index > 0 and row[1].button(
            "↑", key=f"{key_prefix}_up_{index}", help="위로"
        ):
            order = names[:]
            order[index - 1], order[index] = order[index], order[index - 1]
            save_group_order(path, order)
            st.rerun()

        if index < len(names) - 1 and row[2].button(
            "↓", key=f"{key_prefix}_down_{index}", help="아래로"
        ):
            order = names[:]
            order[index + 1], order[index] = order[index], order[index + 1]
            save_group_order(path, order)
            st.rerun()
