# charts.py
"""
5패널 종합차트를 고해상도 PNG bytes로 반환합니다.

패널 1: 주가·목표가·매매 신호
패널 2: 거래량
패널 3: 외국인·기관·개인 누적 순매수 수량
패널 4: 매출액·영업이익·순이익
패널 5: MACD

공매도·대차거래 데이터는 사용하지 않습니다.

주의:
- 누적 순매수 수량은 전체 조회 기간에서 먼저 계산합니다.
- 가격·수급 차트는 최근 730일을 표시합니다.
- 재무 차트는 엑셀에 입력된 최근 8개 분기를 표시합니다.
- 누락값은 0으로 보정하지 않습니다.
"""

import io
import os
import platform
import threading
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import FuncFormatter

import numpy as np
import pandas as pd


_CHART_LOCK = threading.RLock()

# 실제 엑셀의 단위에 맞춰 변경하세요.
# 이 설정은 축의 표시만 바꾸며 숫자를 환산하지 않습니다.
FINANCIAL_UNIT_LABEL = "엑셀 입력 단위"

DISPLAY_DAYS = 730
FINANCIAL_QUARTERS = 8

VOLUME_COLUMNS = {
    "외국인합계_순매수": "외국인_누적순매수수량",
    "기관합계_순매수": "기관_누적순매수수량",
    "개인_순매수": "개인_누적순매수수량",
}


# =========================================================
# 공통 함수
# =========================================================

def to_numeric(series):
    """숫자 문자열과 결측값을 정리합니다."""
    if not pd.api.types.is_numeric_dtype(series):
        series = (
            series.astype("string")
            .str.replace(",", "", regex=False)
            .str.strip()
        )

    return (
        pd.to_numeric(series, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .astype(float)
    )


def number_formatter(value, position):
    return f"{value:,.0f}"


def panel_message(ax, message):
    ax.text(
        0.5,
        0.45,
        message,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=9,
        color="#6b7280",
        linespacing=1.5,
    )


def set_panel_style(ax, title, ylabel):
    ax.set_title(
        title,
        loc="left",
        fontsize=10,
        fontweight="bold",
        pad=3,
    )

    ax.set_ylabel(
        ylabel,
        fontsize=9,
        fontweight="bold",
        labelpad=5,
    )

    ax.yaxis.set_major_formatter(
        FuncFormatter(number_formatter)
    )

    ax.tick_params(
        axis="both",
        labelsize=8,
    )

    ax.grid(
        True,
        alpha=0.22,
        linewidth=0.6,
    )

    ax.set_axisbelow(True)


# =========================================================
# 차트 클래스
# =========================================================

class ChartPlotter:

    # -----------------------------------------------------
    # 한글 폰트
    # -----------------------------------------------------

    @staticmethod
    def setup_korean_font():
        base_dir = Path(__file__).resolve().parent

        candidates = [
            base_dir / "fonts" / "NanumGothic.ttf",
            base_dir / "fonts" / "NotoSansKR-Regular.ttf",
        ]

        system = platform.system()

        if system == "Windows":
            windows_dir = Path(
                os.environ.get("WINDIR", r"C:\Windows")
            )

            candidates.extend([
                windows_dir / "Fonts" / "malgun.ttf",
                windows_dir / "Fonts" / "malgunbd.ttf",
            ])

        elif system == "Darwin":
            candidates.extend([
                Path(
                    "/System/Library/Fonts/"
                    "Supplemental/AppleGothic.ttf"
                ),
                Path("/Library/Fonts/AppleGothic.ttf"),
            ])

        else:
            candidates.extend([
                Path(
                    "/usr/share/fonts/truetype/"
                    "nanum/NanumGothic.ttf"
                ),
                Path(
                    "/usr/share/fonts/truetype/"
                    "nanum/NanumBarunGothic.ttf"
                ),
            ])

        for font_path in candidates:
            if not font_path.is_file():
                continue

            font_manager.fontManager.addfont(
                str(font_path)
            )

            font_name = font_manager.FontProperties(
                fname=str(font_path)
            ).get_name()

            plt.rcParams["font.family"] = [font_name]
            plt.rcParams["font.sans-serif"] = [font_name]
            plt.rcParams["axes.unicode_minus"] = False

            return font_name

        installed = {
            font.name
            for font in font_manager.fontManager.ttflist
        }

        for font_name in (
            "NanumGothic",
            "NanumBarunGothic",
            "Malgun Gothic",
            "AppleGothic",
            "Noto Sans CJK KR",
            "Noto Sans KR",
        ):
            if font_name not in installed:
                continue

            plt.rcParams["font.family"] = [font_name]
            plt.rcParams["font.sans-serif"] = [font_name]
            plt.rcParams["axes.unicode_minus"] = False

            return font_name

        raise RuntimeError(
            "한글 폰트를 찾지 못했습니다. "
            "Community Cloud에서는 packages.txt에 "
            "fonts-nanum을 추가하거나, "
            "fonts 폴더에 NanumGothic.ttf를 넣으세요."
        )

    # -----------------------------------------------------
    # 데이터 준비
    # -----------------------------------------------------

    @staticmethod
    def prepare_chart_frame(df):
        if df is None or df.empty:
            raise ValueError("차트 데이터가 없습니다.")

        frame = df.copy()

        if frame.columns.duplicated().any():
            raise ValueError(
                "차트 데이터에 중복된 컬럼이 있습니다."
            )

        if isinstance(frame.index, pd.DatetimeIndex):
            dates = frame.index

        elif "날짜" in frame.columns:
            date_values = frame["날짜"]

            if pd.api.types.is_numeric_dtype(
                date_values.dtype
            ):
                date_values = (
                    date_values
                    .astype("Int64")
                    .astype("string")
                )

            dates = pd.DatetimeIndex(
                pd.to_datetime(
                    date_values,
                    errors="coerce",
                )
            )

        else:
            raise ValueError(
                "날짜 인덱스 또는 '날짜' 컬럼이 필요합니다."
            )

        if dates.tz is not None:
            dates = (
                dates
                .tz_convert("Asia/Seoul")
                .tz_localize(None)
            )

        frame.index = dates.normalize()

        frame = frame.loc[
            ~frame.index.isna()
        ].copy()

        frame = frame.sort_index()

        frame = frame.loc[
            ~frame.index.duplicated(keep="last")
        ].copy()

        if frame.empty:
            raise ValueError(
                "유효한 날짜 데이터가 없습니다."
            )

        required = {
            "시가",
            "고가",
            "저가",
            "종가",
            "거래량",
        }

        missing = required - set(frame.columns)

        if missing:
            raise ValueError(
                "차트 필수 컬럼 누락: "
                + ", ".join(sorted(missing))
            )

        for column in required:
            frame[column] = to_numeric(
                frame[column]
            )

        frame["날짜"] = frame.index

        frame["분기"] = [
            f"{date.year}.{(date.month - 1) // 3 + 1}Q"
            for date in frame.index
        ]

        # 최근 표시 구간을 자르기 전에 전체 기간을 누적합니다.
        for source, target in VOLUME_COLUMNS.items():
            if source not in frame.columns:
                continue

            values = to_numeric(
                frame[source]
            )

            frame[source] = values

            # 누락값 이후의 정확한 누적값은 알 수 없으므로
            # 이후 누적값도 NaN으로 유지합니다.
            frame[target] = values.cumsum(
                skipna=False
            )

        return frame

    @staticmethod
    def filter_last_2_years(df):
        if df is None or df.empty:
            return pd.DataFrame()

        cutoff = (
            df.index.max()
            - pd.Timedelta(days=DISPLAY_DAYS)
        )

        return (
            df.loc[df.index >= cutoff]
            .copy()
            .reset_index(drop=True)
        )

    # -----------------------------------------------------
    # 패널 1: 주가·목표가·신호
    # -----------------------------------------------------

    @staticmethod
    def plot_candlestick(ax, df, width=0.6):
        plot_df = df.dropna(
            subset=["시가", "고가", "저가", "종가"]
        )

        if plot_df.empty:
            panel_message(
                ax,
                "표시할 캔들 데이터가 없습니다.",
            )
            return

        up = plot_df.loc[
            plot_df["종가"] >= plot_df["시가"]
        ]

        down = plot_df.loc[
            plot_df["종가"] < plot_df["시가"]
        ]

        for subset, color, edgecolor in (
            (up, "#dc2626", "#991b1b"),
            (down, "#2563eb", "#1e40af"),
        ):
            if subset.empty:
                continue

            ax.vlines(
                subset.index,
                subset["저가"],
                subset["고가"],
                color=color,
                linewidth=0.7,
                zorder=2,
            )

            bottom = subset[
                ["시가", "종가"]
            ].min(axis=1)

            height = (
                subset["종가"] - subset["시가"]
            ).abs()

            ax.bar(
                subset.index,
                height,
                width=width,
                bottom=bottom,
                color=color,
                edgecolor=edgecolor,
                linewidth=0.4,
                zorder=3,
            )

            doji = subset.loc[
                height.eq(0)
            ]

            if not doji.empty:
                ax.hlines(
                    doji["종가"],
                    doji.index - width / 2,
                    doji.index + width / 2,
                    color=color,
                    linewidth=0.9,
                    zorder=3,
                )

        last_index = plot_df.index[-1]

        ax.axvspan(
            last_index - 0.5,
            last_index + 0.5,
            color="gold",
            alpha=0.08,
        )

    @staticmethod
    def plot_signals(ax, df):
        specs = [
            (
                "Weekly_Buy",
                "^",
                "#dc2626",
                80,
                "주간 매수",
                "저가",
                0.97,
            ),
            (
                "Weekly_Sell",
                "v",
                "#2563eb",
                80,
                "주간 매도",
                "고가",
                1.03,
            ),
            (
                "Weekly_Monthly_Buy",
                "D",
                "#16a34a",
                100,
                "주간·월간 매수",
                "저가",
                0.94,
            ),
            (
                "Weekly_Monthly_Sell",
                "X",
                "#9740e8",
                100,
                "주간·월간 매도",
                "고가",
                1.06,
            ),
        ]

        for (
            column,
            marker,
            color,
            size,
            label,
            price_column,
            factor,
        ) in specs:
            if column not in df.columns:
                continue

            signals = pd.to_numeric(
                df[column],
                errors="coerce",
            )

            subset = df.loc[
                signals.eq(1)
            ].dropna(
                subset=[price_column]
            )

            if subset.empty:
                continue

            ax.scatter(
                subset.index,
                subset[price_column] * factor,
                marker=marker,
                color=color,
                s=size,
                edgecolors="black",
                linewidth=0.5,
                label=label,
                zorder=6,
            )

    @staticmethod
    def plot_price_panel(ax, df, stock_name):
        set_panel_style(
            ax,
            f"1. {stock_name} — 주가·목표가·매매 신호",
            "가격 (원)",
        )

        ChartPlotter.plot_candlestick(
            ax,
            df,
        )

        if "목표가" in df.columns:
            target = to_numeric(
                df["목표가"]
            ).clip(lower=0)

            valid = target.dropna()

            if not valid.empty:
                ax.plot(
                    df.index,
                    target,
                    color="#ff7f0e",
                    linestyle="--",
                    linewidth=1.7,
                    label="목표가",
                )

                last_index = valid.index[-1]
                last_value = float(valid.iloc[-1])

                ax.annotate(
                    f"목표가 {last_value:,.0f}",
                    xy=(last_index, last_value),
                    xytext=(-5, 6),
                    textcoords="offset points",
                    color="#c45a00",
                    fontsize=8,
                    fontweight="bold",
                    ha="right",
                    va="bottom",
                )

        ChartPlotter.plot_signals(
            ax,
            df,
        )

        handles, labels = (
            ax.get_legend_handles_labels()
        )

        if handles:
            ax.legend(
                handles,
                labels,
                loc="upper left",
                ncol=3,
                fontsize=7.5,
                framealpha=0.9,
            )

        ax.margins(y=0.13)

    # -----------------------------------------------------
    # 패널 2: 거래량
    # -----------------------------------------------------

    @staticmethod
    def plot_volume_panel(ax, df):
        set_panel_style(
            ax,
            "2. 거래량",
            "거래량 (주)",
        )

        colors = np.where(
            df["종가"] >= df["시가"],
            "#dc2626",
            "#2563eb",
        )

        ax.bar(
            df.index,
            df["거래량"],
            color=colors,
            alpha=0.7,
            width=0.7,
        )

        ax.set_ylim(bottom=0)

    # -----------------------------------------------------
    # 패널 3: 누적 순매수 수량
    # -----------------------------------------------------

    @staticmethod
    def plot_investor_volume_panel(
        ax,
        df,
        cumulative_start,
    ):
        set_panel_style(
            ax,
            "3. 외국인·기관·개인 누적 순매수 수량"
            f" — 누적 시작 {cumulative_start}",
            "누적 순매수 (주)",
        )

        specs = [
            (
                "외국인_누적순매수수량",
                "외국인",
                "#dc2626",
            ),
            (
                "기관_누적순매수수량",
                "기관",
                "#111827",
            ),
            (
                "개인_누적순매수수량",
                "개인",
                "#2563eb",
            ),
        ]

        plotted = False
        incomplete = False
        valid_dates = []

        for column, label, color in specs:
            if column not in df.columns:
                incomplete = True
                continue

            values = to_numeric(
                df[column]
            )

            if values.isna().any():
                incomplete = True

            valid = values.dropna()

            if valid.empty:
                incomplete = True
                continue

            last_index = valid.index[-1]
            last_value = float(valid.iloc[-1])

            valid_dates.append(
                pd.Timestamp(
                    df.loc[last_index, "날짜"]
                )
            )

            ax.plot(
                df.index,
                values,
                color=color,
                linewidth=1.8,
                label=f"{label} {last_value:,.0f}주",
            )

            plotted = True

        ax.axhline(
            0,
            color="#6b7280",
            linewidth=0.8,
            linestyle="--",
        )

        if not plotted:
            panel_message(
                ax,
                "누적 순매수 수량을 표시할 수 없습니다.\n"
                "투자자 데이터 포함 여부와 조회 누락을 확인하세요.",
            )
            return

        ax.legend(
            loc="upper left",
            ncol=3,
            fontsize=8,
            framealpha=0.9,
        )

        note = (
            "표시 시리즈의 공통 유효 종료일: "
            + min(valid_dates).strftime("%Y-%m-%d")
        )

        if incomplete:
            note += " · 일부 데이터 누락"

        ax.text(
            0.99,
            0.03,
            note,
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=7.5,
            color="#6b7280",
            bbox={
                "facecolor": "white",
                "alpha": 0.8,
                "edgecolor": "none",
                "pad": 1,
            },
        )

        ax.margins(y=0.22)

    # -----------------------------------------------------
    # 패널 4: 매출액·영업이익·순이익
    # -----------------------------------------------------

    @staticmethod
    def plot_financial_panel(
        ax,
        df,
        stock_name,
        excel_file=None,
    ):
        """
        최근 8개 입력 분기의 재무 데이터를 표시합니다.

        왼쪽 축: 매출액 막대
        오른쪽 축: 영업이익·순이익 선

        원본에 실적/추정 구분이 없으므로 이를 임의로 판단하지 않습니다.
        """
        set_panel_style(
            ax,
            "4. 분기 재무 — 매출액·영업이익·순이익",
            f"매출액\n({FINANCIAL_UNIT_LABEL})",
        )

        if (
            not excel_file
            or not Path(excel_file).is_file()
        ):
            panel_message(
                ax,
                "재무 엑셀 파일이 없습니다.",
            )
            ax.set_xticks([])
            return

        financial = pd.read_excel(
            excel_file,
            sheet_name=stock_name,
        )

        financial.columns = [
            str(column).strip()
            for column in financial.columns
        ]

        if financial.columns.duplicated().any():
            raise ValueError(
                "재무 엑셀에 중복된 컬럼이 있습니다."
            )

        required = {
            "분기",
            "매출액",
            "영업이익",
            "순이익",
        }

        missing = (
            required - set(financial.columns)
        )

        if missing:
            raise ValueError(
                "재무 컬럼 누락: "
                + ", ".join(sorted(missing))
            )

        if financial.empty:
            raise ValueError(
                "재무 엑셀에 데이터가 없습니다."
            )

        financial = financial.copy()

        financial["분기"] = (
            financial["분기"]
            .astype("string")
            .str.strip()
        )

        valid_quarters = (
            financial["분기"]
            .str.fullmatch(r"\d{4}\.[1-4]Q")
            .fillna(False)
        )

        if not valid_quarters.all():
            raise ValueError(
                "분기는 YYYY.NQ 형식이어야 합니다."
            )

        if financial["분기"].duplicated().any():
            raise ValueError(
                "재무 엑셀에 중복된 분기가 있습니다."
            )

        financial["_year"] = (
            financial["분기"]
            .str[:4]
            .astype(int)
        )

        financial["_quarter"] = (
            financial["분기"]
            .str[5]
            .astype(int)
        )

        for column in (
            "매출액",
            "영업이익",
            "순이익",
        ):
            original = financial[column]
            converted = to_numeric(original)

            supplied = (
                original.notna()
                & original.astype("string")
                .str.strip()
                .ne("")
            )

            if (
                supplied & converted.isna()
            ).any():
                raise ValueError(
                    f"'{column}'에는 숫자 또는 빈칸만 입력하세요."
                )

            financial[column] = converted

        financial = (
            financial
            .sort_values(
                ["_year", "_quarter"]
            )
            .tail(FINANCIAL_QUARTERS)
            .reset_index(drop=True)
        )

        financial_values = financial[
            ["매출액", "영업이익", "순이익"]
        ]

        if not (
            financial_values
            .notna()
            .any()
            .any()
        ):
            raise ValueError(
                "표시할 재무 숫자 데이터가 없습니다."
            )

        x_values = np.arange(
            len(financial)
        )

        quarters = (
            financial["분기"].tolist()
        )

        revenue = financial[
            "매출액"
        ].to_numpy(dtype=float)

        operating = financial[
            "영업이익"
        ].to_numpy(dtype=float)

        net_income = financial[
            "순이익"
        ].to_numpy(dtype=float)

        reference_date = pd.Timestamp(
            df["날짜"].max()
        ).normalize()

        quarter_starts = pd.to_datetime({
            "year": financial["_year"],
            "month": (
                financial["_quarter"] - 1
            ) * 3 + 1,
            "day": np.ones(
                len(financial),
                dtype=int,
            ),
        })

        future_mask = (
            quarter_starts > reference_date
        ).to_numpy()

        valid_revenue = np.isfinite(
            revenue
        )

        revenue_colors = np.asarray([
            "#cbd5e1" if future else "#93c5fd"
            for future in future_mask
        ])

        if valid_revenue.any():
            bars = ax.bar(
                x_values[valid_revenue],
                revenue[valid_revenue],
                width=0.48,
                color=revenue_colors[valid_revenue],
                edgecolor="#3b82f6",
                linewidth=0.7,
                alpha=0.75,
                label="매출액",
            )

            for bar, value in zip(
                bars,
                revenue[valid_revenue],
            ):
                ax.annotate(
                    f"{value:,.0f}",
                    xy=(
                        bar.get_x()
                        + bar.get_width() / 2,
                        value,
                    ),
                    xytext=(
                        0,
                        3 if value >= 0 else -3,
                    ),
                    textcoords="offset points",
                    ha="center",
                    va=(
                        "bottom"
                        if value >= 0
                        else "top"
                    ),
                    fontsize=7.5,
                    color="#1e40af",
                )

        right_ax = ax.twinx()

        right_ax.set_ylabel(
            "영업이익·순이익\n"
            f"({FINANCIAL_UNIT_LABEL})",
            fontsize=9,
            fontweight="bold",
            labelpad=5,
        )

        right_ax.yaxis.set_major_formatter(
            FuncFormatter(number_formatter)
        )

        right_ax.tick_params(
            axis="y",
            labelsize=8,
        )

        for series, label, color, marker in (
            (
                operating,
                "영업이익",
                "#dc2626",
                "o",
            ),
            (
                net_income,
                "순이익",
                "#16a34a",
                "s",
            ),
        ):
            if not np.isfinite(series).any():
                continue

            right_ax.plot(
                x_values,
                series,
                label=label,
                color=color,
                marker=marker,
                markersize=4.5,
                linewidth=1.8,
            )

            for x_value, value in zip(
                x_values,
                series,
            ):
                if not np.isfinite(value):
                    continue

                offset = (
                    6
                    if label == "영업이익"
                    else -10
                )

                right_ax.annotate(
                    f"{value:,.0f}",
                    xy=(x_value, value),
                    xytext=(0, offset),
                    textcoords="offset points",
                    ha="center",
                    va=(
                        "bottom"
                        if offset > 0
                        else "top"
                    ),
                    fontsize=7.5,
                    color=color,
                )

        for x_value, future in zip(
            x_values,
            future_mask,
        ):
            if future:
                ax.axvspan(
                    x_value - 0.48,
                    x_value + 0.48,
                    color="#9ca3af",
                    alpha=0.10,
                    zorder=0,
                )

        ax.set_xticks(x_values)

        # 회전하지 않아 아래 MACD 패널과의 간격을 줄일 수 있습니다.
        ax.set_xticklabels(
            quarters,
            rotation=0,
            fontsize=8,
        )

        for tick, future in zip(
            ax.get_xticklabels(),
            future_mask,
        ):
            if future:
                tick.set_color("#6b7280")

        ax.set_xlim(
            -0.6,
            len(financial) - 0.4,
        )

        # 숫자 라벨과 범례가 잘리지 않도록 위아래 공간 확보
        ax.margins(y=0.32)
        right_ax.margins(y=0.40)

        ax.axhline(
            0,
            color="#94a3b8",
            linewidth=0.6,
        )

        handles_left, labels_left = (
            ax.get_legend_handles_labels()
        )

        handles_right, labels_right = (
            right_ax.get_legend_handles_labels()
        )

        ax.legend(
            handles_left + handles_right,
            labels_left + labels_right,
            loc="upper left",
            ncol=3,
            fontsize=7.5,
            framealpha=0.9,
        )

        ax.text(
            0.99,
            0.98,
            "최근 8개 입력 분기 · 실적/추정 구분은 원본 미제공\n"
            "회색 배경: 가격 기준일보다 뒤에 시작하는 분기",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=7,
            color="#6b7280",
            bbox={
                "facecolor": "white",
                "alpha": 0.85,
                "edgecolor": "none",
                "pad": 1,
            },
        )

    # -----------------------------------------------------
    # 패널 5: MACD
    # -----------------------------------------------------

    @staticmethod
    def plot_macd_panel(ax, df):
        set_panel_style(
            ax,
            "5. 일간·주간·월간 MACD Oscillator",
            "MACD\nOscillator",
        )

        specs = [
            (
                "Daily_Oscillator",
                "일간",
                "#111827",
            ),
            (
                "Weekly_Oscillator",
                "주간",
                "#dc2626",
            ),
            (
                "Monthly_Oscillator",
                "월간",
                "#16a34a",
            ),
        ]

        plotted = False

        x_values = (
            df.index.to_numpy(dtype=float)
        )

        for column, label, color in specs:
            if column not in df.columns:
                continue

            values = to_numeric(
                df[column]
            ).to_numpy(dtype=float)

            valid = np.isfinite(
                values
            )

            if not valid.any():
                continue

            ax.plot(
                x_values,
                values,
                color=color,
                linewidth=1.3,
                label=label,
            )

            ax.fill_between(
                x_values,
                values,
                0,
                where=valid,
                color=color,
                alpha=0.10,
            )

            plotted = True

        ax.axhline(
            0,
            color="#374151",
            linewidth=0.7,
            linestyle="--",
        )

        if plotted:
            ax.legend(
                loc="upper left",
                ncol=3,
                fontsize=8,
            )
        else:
            panel_message(
                ax,
                "MACD 데이터가 없습니다.",
            )

    # -----------------------------------------------------
    # 종합차트 생성
    # -----------------------------------------------------

    @staticmethod
    def generate_chart(
        df,
        stock_name,
        excel_file=None,
        save_chart=False,
        interactive=False,
        img_dir="img",
    ):
        """
        5패널 Figure를 생성합니다.

        interactive는 기존 호출 호환용으로만 유지합니다.
        실제 렌더링은 Agg 백엔드를 사용합니다.
        """
        ChartPlotter.setup_korean_font()

        full_frame = (
            ChartPlotter.prepare_chart_frame(df)
        )

        cumulative_start = (
            full_frame.index[0]
            .strftime("%Y-%m-%d")
        )

        plot_df = (
            ChartPlotter.filter_last_2_years(
                full_frame
            )
        )

        if plot_df.empty:
            raise ValueError(
                "차트에 표시할 데이터가 없습니다."
            )

        fig = plt.figure(
            figsize=(18, 14)
        )

        try:
            grid = fig.add_gridspec(
                5,
                1,
                height_ratios=[
                    2.5,
                    0.7,
                    1.2,
                    1.65,
                    1.1,
                ],
                hspace=0.20,
            )

            ax1 = fig.add_subplot(
                grid[0]
            )

            ax2 = fig.add_subplot(
                grid[1],
                sharex=ax1,
            )

            ax3 = fig.add_subplot(
                grid[2],
                sharex=ax1,
            )

            # 재무는 독립적인 분기 X축 사용
            ax4 = fig.add_subplot(
                grid[3]
            )

            ax5 = fig.add_subplot(
                grid[4],
                sharex=ax1,
            )

            ChartPlotter.plot_price_panel(
                ax1,
                plot_df,
                stock_name,
            )

            ChartPlotter.plot_volume_panel(
                ax2,
                plot_df,
            )

            ChartPlotter.plot_investor_volume_panel(
                ax3,
                plot_df,
                cumulative_start,
            )

            # 재무 오류가 다른 패널 생성을 막지 않도록 처리
            axes_before_financial = set(
                fig.axes
            )

            try:
                ChartPlotter.plot_financial_panel(
                    ax4,
                    plot_df,
                    stock_name,
                    excel_file,
                )

            except Exception as error:
                # 재무 패널에서 생성한 보조축만 제거
                for extra_ax in list(fig.axes):
                    if extra_ax not in axes_before_financial:
                        extra_ax.remove()

                ax4.clear()

                set_panel_style(
                    ax4,
                    "4. 분기 재무",
                    FINANCIAL_UNIT_LABEL,
                )

                panel_message(
                    ax4,
                    "재무 데이터를 표시하지 못했습니다.\n"
                    + str(error),
                )

                ax4.set_xticks([])

            ChartPlotter.plot_macd_panel(
                ax5,
                plot_df,
            )

            tick_count = min(
                10,
                len(plot_df),
            )

            tick_positions = np.unique(
                np.linspace(
                    0,
                    len(plot_df) - 1,
                    tick_count,
                    dtype=int,
                )
            )

            tick_labels = [
                pd.Timestamp(
                    plot_df.loc[position, "날짜"]
                ).strftime("%Y-%m-%d")
                for position in tick_positions
            ]

            ax5.set_xticks(
                tick_positions
            )

            ax5.set_xticklabels(
                tick_labels,
                rotation=25,
                ha="right",
                fontsize=8,
            )

            ax5.set_xlabel(
                "거래일",
                fontsize=9,
                labelpad=3,
            )

            ax1.set_xlim(
                -1,
                len(plot_df),
            )

            # 공통 날짜는 마지막 MACD 패널에만 표시
            for ax in (
                ax1,
                ax2,
                ax3,
            ):
                ax.tick_params(
                    axis="x",
                    labelbottom=False,
                )

            fig.subplots_adjust(
                left=0.095,
                right=0.90,
                top=0.975,
                bottom=0.065,
            )

            if save_chart:
                output_dir = Path(
                    img_dir
                )

                output_dir.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                save_date = pd.Timestamp.now(
                    tz="Asia/Seoul"
                ).strftime("%Y%m%d")

                safe_name = "".join(
                    "_"
                    if (
                        char in '<>:"/\\|?*'
                        or ord(char) < 32
                    )
                    else char
                    for char in str(stock_name)
                ).strip(" .")

                if not safe_name:
                    safe_name = "stock"

                filename = (
                    output_dir
                    / f"{safe_name}_{save_date}.png"
                )

                fig.savefig(
                    filename,
                    format="png",
                    dpi=300,
                    bbox_inches="tight",
                    facecolor="white",
                )

            return fig

        except Exception:
            plt.close(fig)
            raise


# =========================================================
# Streamlit / 배치 실행용 PNG 반환
# =========================================================

def make_chart_png(
    df,
    stock_name,
    excel_file=None,
):
    """5패널 종합차트를 PNG bytes로 반환합니다."""
    with _CHART_LOCK:
        fig = None

        try:
            fig = ChartPlotter.generate_chart(
                df,
                stock_name,
                excel_file=excel_file,
                save_chart=False,
                interactive=False,
            )

            with io.BytesIO() as buffer:
                fig.savefig(
                    buffer,
                    format="png",
                    dpi=300,
                    bbox_inches="tight",
                    facecolor="white",
                    transparent=False,
                )

                return buffer.getvalue()

        finally:
            if fig is not None:
                plt.close(fig)
