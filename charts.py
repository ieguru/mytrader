# charts.py
"""
5패널 종합차트를 고해상도 PNG bytes로 반환합니다.

패널 1:
    주가·목표가·매매 신호

패널 2:
    거래량

패널 3:
    일간·주간·월간 MACD Oscillator

패널 4:
    외국인·기관·개인 누적 순매수 수량

패널 5:
    매출액·영업이익·순이익

실제 데이터 표시 기간:
    마지막 가격 데이터 날짜 기준 최근 18개월

분기 재무:
    현재 분기를 포함한 최근 6개 분기를 표시합니다.
    엑셀에 숫자가 입력된 미래 분기도 추가합니다.
    분기 시작월의 첫 가용 거래일 위치에 표시합니다.

    종료 분기:
        매출액: 회색 채움 막대
        영업이익·순이익: 실선, 채운 마커

    진행 중·미래 분기:
        매출액: 연한 회색 빗금 막대
        영업이익·순이익: 점선, 빈 마커
        분기명에 '(예상)' 표시

미래 날짜 축:
    모든 패널의 X축을 동일하게 확장합니다.
    미래 구간은 월~금 기준의 표시용 날짜를 사용합니다.
    미래 KRX 휴장일을 반영한 실제 거래일 달력은 아닙니다.

    미래 구간에는 가격·거래량·MACD·수급 데이터를
    생성하거나 연장하지 않습니다.

주의:
    분기 시작 위치는 시각적 배치입니다.
    해당 날짜에 실적이 공개되었다는 의미가 아닙니다.

    엑셀에 공시일·실적/추정 구분이 없으므로,
    종료 분기는 실적 표시로 간주합니다.
    종료 분기의 값이 실제 확정 실적인지 검증하지는 않습니다.

    투자자 누적 순매수는 전체 조회 기간에서 계산한 후,
    표시 구간만 최근 18개월로 제한합니다.

    공매도 및 대차거래 데이터는 사용하지 않습니다.
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
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

import numpy as np
import pandas as pd


# =========================================================
# 기본 설정
# =========================================================

_CHART_LOCK = threading.RLock()

# 축의 단위 표시만 바꾸며 숫자를 환산하지 않습니다.
# 모든 재무 입력값이 억원 단위이면 "억원"으로 변경하세요.
FINANCIAL_UNIT_LABEL = "엑셀 입력 단위"

DISPLAY_MONTHS = 18
FINANCIAL_QUARTERS = 6

DAILY_MACD_LINEWIDTH = 0.65
WEEKLY_MACD_LINEWIDTH = 1.30
MONTHLY_MACD_LINEWIDTH = 2.60

VOLUME_COLUMNS = {
    "외국인합계_순매수": "외국인_누적순매수수량",
    "기관합계_순매수": "기관_누적순매수수량",
    "개인_순매수": "개인_누적순매수수량",
}


# =========================================================
# 공통 함수
# =========================================================

# =========================================================
# 차트에 표시할 매매 신호 (키, 화면 라벨)
# =========================================================

SIGNAL_OPTIONS = (
    ("Weekly_Buy", "주간 매수"),
    ("Weekly_Sell", "주간 매도"),
    ("Weekly_Monthly_Buy", "주간·월간 매수"),
    ("Weekly_Monthly_Sell", "주간·월간 매도"),
    ("유목민신호", "유목민 매수"),
    ("유목민매도신호", "유목민 매도"),
    ("이동평균매수", "이동평균 매수"),
    ("이동평균매도", "이동평균 매도"),
    ("이동평균매수차단", "이동평균 매수 차단"),
    ("쌍바닥신호", "쌍바닥 매수"),
    ("쌍고점신호", "쌍고점 매도"),
)

# 기본값: 전체 표시
DEFAULT_SIGNALS = tuple(key for key, _ in SIGNAL_OPTIONS)


def signal_labels():
    """(키, 라벨) 목록을 돌려줍니다(화면 체크박스 순서)."""
    return list(SIGNAL_OPTIONS)


def normalize_signals(signals=None):
    """
    선택된 신호 키 집합으로 정규화합니다.

    None      -> 전체 신호
    빈 목록   -> 아무 신호도 표시하지 않음
    모르는 키 -> 무시
    """
    if signals is None:
        return set(DEFAULT_SIGNALS)

    if isinstance(signals, str):
        signals = [signals]

    valid = {key for key, _ in SIGNAL_OPTIONS}

    return {str(key) for key in signals} & valid


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
                os.environ.get(
                    "WINDIR",
                    r"C:\Windows",
                )
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

            font_name = (
                font_manager.FontProperties(
                    fname=str(font_path)
                )
                .get_name()
            )

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
            (
                f"{timestamp.year}."
                f"{(timestamp.month - 1) // 3 + 1}Q"
            )
            for timestamp in frame.index
        ]

        # 표시 기간을 자르기 전에 전체 조회 기간을 누적합니다.
        for source, target in VOLUME_COLUMNS.items():
            if source not in frame.columns:
                continue

            values = to_numeric(frame[source])

            frame[source] = values

            # 누락값을 0으로 대체하지 않습니다.
            # 누락 이후 정확한 누적값은 알 수 없으므로 NaN 유지.
            frame[target] = values.cumsum(
                skipna=False
            )

        return frame

    @staticmethod
    def filter_recent_period(df):
        """실제 데이터의 최근 18개월을 반환합니다."""
        if df is None or df.empty:
            return pd.DataFrame()

        reference_date = df.index.max().normalize()

        cutoff = (
            reference_date
            - pd.DateOffset(months=DISPLAY_MONTHS)
        )

        return (
            df.loc[
                (df.index >= cutoff)
                & (df.index <= reference_date)
            ]
            .copy()
            .reset_index(drop=True)
        )

    @staticmethod
    def filter_last_2_years(df):
        """
        이전 외부 호출과의 호환용 함수.

        이름과 관계없이 최근 18개월을 적용합니다.
        """
        return ChartPlotter.filter_recent_period(df)

    # -----------------------------------------------------
    # 패널 1: 주가·목표가·매매 신호
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

            doji = subset.loc[height.eq(0)]

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
    def plot_signals(ax, df, signals=None):
        """
        매매 신호를 표시합니다.

        signals가 None이면 전체 신호, 목록이면 그 키의 신호만 그립니다.
        """
        selected = normalize_signals(signals)

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
            (
                "유목민신호",
                "*",
                "#f59e0b",
                340,
                "유목민 매수",
                "저가",
                0.90,
            ),
            (
                "유목민매도신호",
                "*",
                "#2563eb",
                340,
                "유목민 매도 (급감 52%)",
                "고가",
                1.10,
            ),
            (
                "이동평균매수",
                "o",
                "#b91c1c",
                62,
                "이동평균 매수 (MA5↑MA20·기울기 전환)",
                "저가",
                0.955,
            ),
            (
                "이동평균매도",
                "o",
                "#1e40af",
                62,
                "이동평균 매도 (기울기↓·음봉·MA5 이탈)",
                "고가",
                1.045,
            ),
            (
                "이동평균매수차단",
                "x",
                "#94a3b8",
                58,
                "이동평균 매수 차단 (MA20 기울기<0)",
                "저가",
                0.955,
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
            if column not in selected or column not in df.columns:
                continue

            signals = pd.to_numeric(
                df[column],
                errors="coerce",
            )

            subset = (
                df.loc[signals.eq(1)]
                .dropna(subset=[price_column])
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

        # 더블 패턴은 원(circle)으로 표시합니다.
        for column, color, label, price_column, factor in (
            (
                "쌍바닥신호",
                "#84cc16",
                "쌍바닥 매수 (double bottom)",
                "저가",
                0.90,
            ),
            (
                "쌍고점신호",
                "#38bdf8",
                "쌍고점 매도 (double top)",
                "고가",
                1.10,
            ),
        ):
            if column not in selected or column not in df.columns:
                continue

            signals = pd.to_numeric(df[column], errors="coerce")
            subset = df.loc[signals.eq(1)].dropna(subset=[price_column])

            if subset.empty:
                continue

            ax.scatter(
                subset.index,
                subset[price_column] * factor,
                marker="o",
                facecolors="none",
                edgecolors=color,
                s=210,
                linewidths=2.0,
                label=label,
                zorder=6,
            )

    @staticmethod
    def plot_price_panel(ax, df, stock_name, signals=None):
        set_panel_style(
            ax,
            f"1. {stock_name} — 주가·목표가·매매 신호"
            f" / 실제 데이터 최근 {DISPLAY_MONTHS}개월",
            "가격 (원)",
        )

        ChartPlotter.plot_candlestick(ax, df)

        if "목표가" in df.columns:
            target = (
                to_numeric(df["목표가"])
                .clip(lower=0)
            )

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

        # 이동평균선(MA5/MA20)을 함께 그려 교차 신호를 눈으로 확인할 수 있게 합니다.
        for column, color, label, style, width in (
            ("MA5", "#f97316", "MA5", "-", 1.8),
            ("MA20", "#7c3aed", "MA20", "-", 1.7),
        ):
            if column not in df.columns:
                continue

            values = to_numeric(df[column])

            if values.notna().sum() == 0:
                continue

            ax.plot(
                df.index,
                values,
                color=color,
                linewidth=width,
                linestyle=style,
                alpha=0.95,
                label=label,
                zorder=4,
            )

        ChartPlotter.plot_signals(ax, df, signals=signals)

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
    # 패널 3: MACD
    # -----------------------------------------------------

    @staticmethod
    def plot_macd_panel(ax, df):
        set_panel_style(
            ax,
            "3. 일간·주간·월간 MACD Oscillator",
            "MACD\nOscillator",
        )

        specs = [
            (
                "Daily_Oscillator",
                "일간",
                "#111827",
                DAILY_MACD_LINEWIDTH,
                2,
            ),
            (
                "Weekly_Oscillator",
                "주간",
                "#dc2626",
                WEEKLY_MACD_LINEWIDTH,
                3,
            ),
            (
                "Monthly_Oscillator",
                "월간",
                "#16a34a",
                MONTHLY_MACD_LINEWIDTH,
                4,
            ),
        ]

        plotted = False
        x_values = df.index.to_numpy(dtype=float)

        for (
            column,
            label,
            color,
            linewidth,
            zorder,
        ) in specs:
            if column not in df.columns:
                continue

            values = (
                to_numeric(df[column])
                .to_numpy(dtype=float)
            )

            valid = np.isfinite(values)

            if not valid.any():
                continue

            ax.plot(
                x_values,
                values,
                color=color,
                linewidth=linewidth,
                label=label,
                zorder=zorder,
            )

            ax.fill_between(
                x_values,
                values,
                0,
                where=valid,
                color=color,
                alpha=0.08,
                zorder=1,
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
    # 패널 4: 외국인·기관·개인 누적 순매수 수량
    # -----------------------------------------------------

    @staticmethod
    def plot_investor_volume_panel(
        ax,
        df,
        cumulative_start,
    ):
        set_panel_style(
            ax,
            "4. 외국인·기관·개인 누적 순매수 수량"
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

            values = to_numeric(df[column])

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
    # 패널 5: 분기 재무
    # -----------------------------------------------------

    @staticmethod
    def plot_financial_panel(
        ax,
        df,
        stock_name,
        excel_file=None,
    ):
        """
        최근 6개 분기와 엑셀의 미래 분기 재무를 표시합니다.

        기준일:
            실제 마지막 가격 데이터 날짜

        표시 위치:
            분기 시작월의 첫 가용 날짜

        미래 분기:
            월~금 기준 표시용 날짜를 추가하여 오른쪽에 배치

        반환값:
            실제 날짜 + 미래 표시용 날짜의 DatetimeIndex

            재무를 표시할 수 없으면 None을 반환합니다.
        """
        set_panel_style(
            ax,
            "5. 분기 재무 — 분기 시작 위치 표시"
            f" / 최근 {FINANCIAL_QUARTERS}개 분기"
            " + 엑셀 미래 전망",
            f"매출액\n({FINANCIAL_UNIT_LABEL})",
        )

        if df is None or df.empty:
            panel_message(
                ax,
                "표시할 가격 데이터가 없습니다.",
            )
            return None

        if (
            not excel_file
            or not Path(excel_file).is_file()
        ):
            panel_message(
                ax,
                "재무 엑셀 파일이 없습니다.",
            )
            return None

        # -------------------------------------------------
        # 1. 재무 엑셀 읽기
        # -------------------------------------------------

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

        missing = required - set(financial.columns)

        if missing:
            raise ValueError(
                "재무 컬럼 누락: "
                + ", ".join(sorted(missing))
            )

        financial = (
            financial
            .dropna(how="all")
            .copy()
        )

        if financial.empty:
            raise ValueError(
                "재무 엑셀에 데이터가 없습니다."
            )

        financial["분기"] = (
            financial["분기"]
            .astype("string")
            .str.strip()
        )

        valid_quarters = (
            financial["분기"]
            .str.fullmatch(
                r"\d{4}\.[1-4]Q",
                na=False,
            )
        )

        if not valid_quarters.all():
            raise ValueError(
                "분기는 YYYY.NQ 형식이어야 합니다."
            )

        if financial["분기"].duplicated().any():
            raise ValueError(
                "재무 엑셀에 중복된 분기가 있습니다."
            )

        value_columns = [
            "매출액",
            "영업이익",
            "순이익",
        ]

        for column in value_columns:
            original = financial[column]
            converted = to_numeric(original)

            supplied = (
                original.notna()
                & original.astype("string")
                .str.strip()
                .ne("")
                .fillna(False)
            )

            if (
                supplied & converted.isna()
            ).any():
                raise ValueError(
                    f"'{column}'에는 "
                    "숫자 또는 빈칸만 입력하세요."
                )

            financial[column] = converted

        period_labels = (
            financial["분기"]
            .str.replace(
                r"^(\d{4})\.([1-4])Q$",
                r"\1Q\2",
                regex=True,
            )
        )

        financial["_period"] = pd.PeriodIndex(
            period_labels.to_numpy(),
            freq="Q-DEC",
        )

        # -------------------------------------------------
        # 2. 실제 가격 날짜와 기준일
        # -------------------------------------------------

        actual_dates = pd.DatetimeIndex(
            pd.to_datetime(
                df["날짜"],
                errors="coerce",
            )
        )

        if actual_dates.tz is not None:
            actual_dates = (
                actual_dates
                .tz_convert("Asia/Seoul")
                .tz_localize(None)
            )

        actual_dates = actual_dates.normalize()

        if (
            actual_dates.isna().any()
            or not actual_dates.is_monotonic_increasing
            or actual_dates.has_duplicates
        ):
            raise ValueError(
                "차트 날짜는 결측·중복 없이 "
                "오름차순이어야 합니다."
            )

        reference_date = actual_dates[-1]

        cutoff = (
            reference_date
            - pd.DateOffset(months=DISPLAY_MONTHS)
        )

        current_period = (
            reference_date.to_period("Q-DEC")
        )

        # 현재 분기를 포함한 최근 6개 분기
        historical_periods = pd.period_range(
            end=current_period,
            periods=FINANCIAL_QUARTERS,
            freq="Q-DEC",
        )

        # 숫자가 하나 이상 입력된 미래 분기만 대상으로 합니다.
        has_financial_values = (
            financial[value_columns]
            .notna()
            .any(axis=1)
        )

        future_periods = pd.PeriodIndex(
            financial.loc[
                (
                    financial["_period"]
                    > current_period
                )
                & has_financial_values,
                "_period",
            ],
            freq="Q-DEC",
        ).sort_values()

        if len(future_periods):
            # 중간 누락 분기도 유지하여
            # 누락 분기를 건너뛰어 선을 잇지 않습니다.
            allowed_periods = pd.period_range(
                start=historical_periods[0],
                end=future_periods[-1],
                freq="Q-DEC",
            )
        else:
            allowed_periods = historical_periods

        financial = (
            financial
            .set_index("_period")
            .reindex(allowed_periods)
            .rename_axis("_period")
            .reset_index()
        )

        periods = pd.PeriodIndex(
            financial["_period"],
            freq="Q-DEC",
        )

        financial["분기"] = [
            f"{period.year}.{period.quarter}Q"
            for period in periods
        ]

        financial["_quarter_start"] = (
            periods.start_time.normalize()
        )

        financial["_quarter_end"] = (
            periods.end_time.normalize()
        )

        # 반드시 실제 마지막 가격 날짜를 기준으로 판단합니다.
        financial["_estimated"] = (
            financial["_quarter_end"]
            > reference_date
        )

        financial = financial.loc[
            financial["_quarter_start"].ge(cutoff)
        ].copy()

        financial = (
            financial
            .sort_values("_quarter_start")
            .reset_index(drop=True)
        )

        if financial.empty:
            panel_message(
                ax,
                "표시할 분기 재무가 없습니다.",
            )
            return None

        # -------------------------------------------------
        # 3. 미래 분기 표시용 날짜 축 확장
        # -------------------------------------------------

        axis_dates = actual_dates

        if len(future_periods):
            last_future_start = (
                future_periods[-1]
                .start_time
                .normalize()
            )

            # 분기 시작일이 주말이면 다음 월~금 날짜에 배치합니다.
            future_display_end = pd.bdate_range(
                start=last_future_start,
                periods=1,
            )[0]

            if future_display_end > reference_date:
                extension_dates = pd.bdate_range(
                    start=(
                        reference_date
                        + pd.Timedelta(days=1)
                    ),
                    end=future_display_end,
                )

                axis_dates = actual_dates.append(
                    extension_dates
                )

        # -------------------------------------------------
        # 4. 분기 시작일을 공통 X축 위치로 변환
        # -------------------------------------------------

        positions = axis_dates.searchsorted(
            pd.DatetimeIndex(
                financial["_quarter_start"]
            ),
            side="left",
        )

        valid_positions = (
            positions < len(axis_dates)
        )

        financial = (
            financial.loc[valid_positions]
            .reset_index(drop=True)
        )

        positions = positions[valid_positions]

        if financial.empty:
            panel_message(
                ax,
                "날짜 축과 겹치는 분기가 없습니다.",
            )
            return None

        mapped_dates = axis_dates.take(
            positions
        )

        mapped_periods = (
            mapped_dates.to_period("Q-DEC")
        )

        financial_periods = pd.PeriodIndex(
            financial["_period"],
            freq="Q-DEC",
        )

        same_quarter = np.asarray(
            mapped_periods == financial_periods
        )

        # 과거 분기 시작월 데이터가 없으면
        # 다른 월로 밀어서 표시하지 않습니다.
        same_start_month = np.asarray(
            mapped_dates.month
            == pd.DatetimeIndex(
                financial["_quarter_start"]
            ).month
        )

        valid_mapping = (
            same_quarter & same_start_month
        )

        financial = (
            financial.loc[valid_mapping]
            .reset_index(drop=True)
        )

        positions = (
            positions[valid_mapping]
            .astype(float)
        )

        if financial.empty:
            panel_message(
                ax,
                "분기 시작월에 대응하는 "
                "날짜 데이터가 없습니다.",
            )
            return None

        revenue = financial[
            "매출액"
        ].to_numpy(dtype=float)

        operating = financial[
            "영업이익"
        ].to_numpy(dtype=float)

        net_income = financial[
            "순이익"
        ].to_numpy(dtype=float)

        estimated = financial[
            "_estimated"
        ].to_numpy(dtype=bool)

        quarter_ordinals = np.array(
            [
                period.ordinal
                for period in financial["_period"]
            ],
            dtype=np.int64,
        )

        all_values = np.concatenate([
            revenue,
            operating,
            net_income,
        ])

        if not np.isfinite(all_values).any():
            panel_message(
                ax,
                "표시할 재무 숫자 데이터가 없습니다.",
            )
            return None

        # -------------------------------------------------
        # 5. 매출액 막대
        # -------------------------------------------------

        if len(positions) >= 2:
            positive_gaps = np.diff(positions)
            positive_gaps = positive_gaps[
                positive_gaps > 0
            ]

            if positive_gaps.size:
                bar_width = min(
                    24.0,
                    max(
                        1.0,
                        float(
                            positive_gaps.min()
                        ) * 0.38,
                    ),
                )
            else:
                bar_width = 12.0
        else:
            bar_width = min(
                18.0,
                max(
                    1.0,
                    len(axis_dates) * 0.05,
                ),
            )

        revenue_specs = (
            (
                False,
                "매출액",
                "#b8b8b8",
                "#737373",
                None,
            ),
            (
                True,
                "매출액(예상)",
                "#eeeeee",
                "#737373",
                "///",
            ),
        )

        for (
            is_estimated,
            label,
            facecolor,
            edgecolor,
            hatch,
        ) in revenue_specs:
            mask = (
                np.isfinite(revenue)
                & (estimated == is_estimated)
            )

            if not mask.any():
                continue

            bars = ax.bar(
                positions[mask],
                revenue[mask],
                width=bar_width,
                color=facecolor,
                edgecolor=edgecolor,
                linewidth=0.9,
                hatch=hatch,
                alpha=0.9,
                label=label,
                zorder=2,
            )

            for bar, value in zip(
                bars,
                revenue[mask],
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
                        4 if value >= 0 else -4,
                    ),
                    textcoords="offset points",
                    ha="center",
                    va=(
                        "bottom"
                        if value >= 0
                        else "top"
                    ),
                    fontsize=7.5,
                    color="#525252",
                )

        # -------------------------------------------------
        # 6. 영업이익·순이익 보조축
        # -------------------------------------------------

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

        line_specs = (
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
        )

        line_legend_handles = []

        for values, label, color, marker in line_specs:
            finite = np.isfinite(values)

            if not finite.any():
                continue

            # 인접한 분기 사이만 연결합니다.
            # 예상이 포함된 연결 구간은 점선으로 표시합니다.
            for index in range(1, len(positions)):
                consecutive = (
                    quarter_ordinals[index]
                    - quarter_ordinals[index - 1]
                    == 1
                )

                if not (
                    consecutive
                    and finite[index - 1]
                    and finite[index]
                ):
                    continue

                forecast_segment = (
                    estimated[index - 1]
                    or estimated[index]
                )

                right_ax.plot(
                    positions[index - 1:index + 1],
                    values[index - 1:index + 1],
                    color=color,
                    linestyle=(
                        "--"
                        if forecast_segment
                        else "-"
                    ),
                    linewidth=1.8,
                    alpha=(
                        0.85
                        if forecast_segment
                        else 1.0
                    ),
                    zorder=4,
                )

            # 종료 분기: 채운 마커
            actual_mask = (
                finite & ~estimated
            )

            if actual_mask.any():
                right_ax.plot(
                    positions[actual_mask],
                    values[actual_mask],
                    linestyle="None",
                    marker=marker,
                    markersize=5,
                    markerfacecolor=color,
                    markeredgecolor=color,
                    color=color,
                    zorder=5,
                )

                line_legend_handles.append(
                    Line2D(
                        [0],
                        [0],
                        color=color,
                        linestyle="-",
                        linewidth=1.8,
                        marker=marker,
                        markersize=5,
                        markerfacecolor=color,
                        markeredgecolor=color,
                        label=label,
                    )
                )

            # 진행 중·미래 분기: 빈 마커
            forecast_mask = (
                finite & estimated
            )

            if forecast_mask.any():
                right_ax.plot(
                    positions[forecast_mask],
                    values[forecast_mask],
                    linestyle="None",
                    marker=marker,
                    markersize=6,
                    markerfacecolor="white",
                    markeredgecolor=color,
                    markeredgewidth=1.5,
                    color=color,
                    zorder=6,
                )

                line_legend_handles.append(
                    Line2D(
                        [0],
                        [0],
                        color=color,
                        linestyle="--",
                        linewidth=1.8,
                        marker=marker,
                        markersize=6,
                        markerfacecolor="white",
                        markeredgecolor=color,
                        markeredgewidth=1.5,
                        label=f"{label}(예상)",
                    )
                )

            # 숫자 표시
            for x_value, value in zip(
                positions,
                values,
            ):
                if not np.isfinite(value):
                    continue

                offset = (
                    7
                    if label == "영업이익"
                    else -11
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

        # -------------------------------------------------
        # 7. 분기명과 예상 표시
        # -------------------------------------------------

        for (
            x_value,
            quarter,
            is_estimated,
        ) in zip(
            positions,
            financial["분기"],
            estimated,
        ):
            quarter_label = (
                f"{quarter}\n(예상)"
                if is_estimated
                else str(quarter)
            )

            ax.text(
                x_value,
                0.02,
                quarter_label,
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="bottom",
                fontsize=7.5,
                fontweight=(
                    "bold"
                    if is_estimated
                    else "normal"
                ),
                color=(
                    "#b45309"
                    if is_estimated
                    else "#525252"
                ),
                bbox={
                    "facecolor": "white",
                    "alpha": 0.9,
                    "edgecolor": "none",
                    "pad": 1,
                },
                zorder=7,
            )

        ax.axhline(
            0,
            color="#94a3b8",
            linewidth=0.6,
        )

        ax.margins(y=0.38)
        right_ax.margins(y=0.45)

        # -------------------------------------------------
        # 8. 범례 및 안내
        # -------------------------------------------------

        bar_handles, bar_labels = (
            ax.get_legend_handles_labels()
        )

        legend_handles = (
            bar_handles + line_legend_handles
        )

        legend_labels = (
            bar_labels
            + [
                handle.get_label()
                for handle in line_legend_handles
            ]
        )

        if legend_handles:
            ax.legend(
                legend_handles,
                legend_labels,
                loc="upper left",
                ncol=3,
                fontsize=7.2,
                framealpha=0.92,
            )

        ax.text(
            0.99,
            0.98,
            "진행·미래 분기: 빗금·점선·빈 마커\n"
            "분기 시작 위치는 공시일을 의미하지 않음",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=7,
            color="#6b7280",
            bbox={
                "facecolor": "white",
                "alpha": 0.9,
                "edgecolor": "none",
                "pad": 1,
            },
        )

        return axis_dates

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
        signals=None,
    ):
        """
        5패널 Figure를 생성합니다.

        실제 데이터:
            마지막 가격 데이터 기준 최근 18개월

        미래 재무:
            엑셀에 입력된 미래 분기까지 오른쪽에 추가

        interactive:
            기존 외부 호출과의 호환용 인자입니다.

        signals:
            차트에 표시할 신호 키 목록(charts.SIGNAL_OPTIONS).
            None이면 전체 신호를 표시합니다.
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
            ChartPlotter.filter_recent_period(
                full_frame
            )
        )

        if plot_df.empty:
            raise ValueError(
                "차트에 표시할 데이터가 없습니다."
            )

        actual_dates = pd.DatetimeIndex(
            pd.to_datetime(plot_df["날짜"])
        )

        actual_count = len(plot_df)
        reference_date = actual_dates[-1]

        # 재무 패널이 미래 날짜를 반환하면 이 축을 확장합니다.
        axis_dates = actual_dates

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
                    1.15,
                    1.2,
                    1.65,
                ],
                hspace=0.20,
            )

            ax1 = fig.add_subplot(grid[0])

            ax2 = fig.add_subplot(
                grid[1],
                sharex=ax1,
            )

            ax3 = fig.add_subplot(
                grid[2],
                sharex=ax1,
            )

            ax4 = fig.add_subplot(
                grid[3],
                sharex=ax1,
            )

            ax5 = fig.add_subplot(
                grid[4],
                sharex=ax1,
            )

            # 실제 시장 데이터만 전달합니다.
            ChartPlotter.plot_price_panel(
                ax1,
                plot_df,
                stock_name,
                signals=signals,
            )

            ChartPlotter.plot_volume_panel(
                ax2,
                plot_df,
            )

            ChartPlotter.plot_macd_panel(
                ax3,
                plot_df,
            )

            ChartPlotter.plot_investor_volume_panel(
                ax4,
                plot_df,
                cumulative_start,
            )

            axes_before_financial = set(fig.axes)

            try:
                financial_dates = (
                    ChartPlotter.plot_financial_panel(
                        ax5,
                        plot_df,
                        stock_name,
                        excel_file,
                    )
                )

                if financial_dates is not None:
                    financial_dates = pd.DatetimeIndex(
                        financial_dates
                    )

                    if len(financial_dates) >= actual_count:
                        axis_dates = financial_dates

            except Exception as error:
                # 재무 패널에서 생성한 보조축만 제거합니다.
                for extra_ax in list(fig.axes):
                    if (
                        extra_ax
                        not in axes_before_financial
                    ):
                        extra_ax.remove()

                ax5.clear()

                set_panel_style(
                    ax5,
                    "5. 분기 재무",
                    FINANCIAL_UNIT_LABEL,
                )

                panel_message(
                    ax5,
                    "재무 데이터를 표시하지 못했습니다.\n"
                    + str(error),
                )

                axis_dates = actual_dates

            # ---------------------------------------------
            # 미래 구간 음영
            # ---------------------------------------------

            has_future = (
                len(axis_dates) > actual_count
            )

            if has_future:
                boundary = actual_count - 0.5

                for panel_ax in (
                    ax1,
                    ax2,
                    ax3,
                    ax4,
                    ax5,
                ):
                    panel_ax.axvspan(
                        boundary,
                        len(axis_dates) - 0.5,
                        color="#f59e0b",
                        alpha=0.06,
                        zorder=0,
                    )

                    panel_ax.axvline(
                        boundary,
                        color="#b45309",
                        linestyle=":",
                        linewidth=1.0,
                        alpha=0.8,
                        zorder=3,
                    )

                ax1.text(
                    0.99,
                    0.97,
                    "오른쪽 음영: 미래 재무 전망 구간\n"
                    f"실제 가격 기준일: "
                    f"{reference_date:%Y-%m-%d}",
                    transform=ax1.transAxes,
                    ha="right",
                    va="top",
                    fontsize=8,
                    color="#92400e",
                    bbox={
                        "facecolor": "white",
                        "alpha": 0.9,
                        "edgecolor": "none",
                        "pad": 3,
                    },
                    zorder=8,
                )

            # ---------------------------------------------
            # 모든 패널의 공통 날짜 눈금
            # ---------------------------------------------

            tick_count = min(
                10,
                len(axis_dates),
            )

            tick_positions = np.unique(
                np.linspace(
                    0,
                    len(axis_dates) - 1,
                    tick_count,
                    dtype=int,
                )
            )

            if has_future:
                # 실제 데이터 종료일을 눈금에 포함합니다.
                tick_positions = np.unique(
                    np.concatenate([
                        tick_positions,
                        np.array(
                            [actual_count - 1],
                            dtype=int,
                        ),
                    ])
                )

                minimum_gap = max(
                    1.0,
                    len(axis_dates) * 0.035,
                )

                # 실제 기준일과 너무 가까운 눈금은 제외합니다.
                tick_positions = np.array(
                    [
                        position
                        for position in tick_positions
                        if (
                            position == actual_count - 1
                            or abs(
                                position
                                - (actual_count - 1)
                            ) >= minimum_gap
                        )
                    ],
                    dtype=int,
                )

            tick_labels = [
                pd.Timestamp(
                    axis_dates[position]
                ).strftime("%Y-%m-%d")
                for position in tick_positions
            ]

            ax5.set_xticks(tick_positions)

            ax5.set_xticklabels(
                tick_labels,
                rotation=25,
                ha="right",
                fontsize=8,
            )

            if has_future:
                xlabel = (
                    "날짜 — 모든 패널 동일 축"
                    " / 미래 구간은 월~금 기준 표시용 날짜"
                )
            else:
                xlabel = (
                    "거래일 — 모든 패널 동일 날짜 기준"
                )

            ax5.set_xlabel(
                xlabel,
                fontsize=9,
                labelpad=3,
            )

            # 막대가 가장자리에서 잘리지 않도록 여백 설정
            edge_padding = min(
                15.0,
                max(
                    1.0,
                    len(axis_dates) * 0.035,
                ),
            )

            ax1.set_xlim(
                -edge_padding,
                len(axis_dates) - 1 + edge_padding,
            )

            for panel_ax in (
                ax1,
                ax2,
                ax3,
                ax4,
            ):
                panel_ax.tick_params(
                    axis="x",
                    labelbottom=False,
                )

            fig.subplots_adjust(
                left=0.095,
                right=0.90,
                top=0.975,
                bottom=0.065,
            )

            # ---------------------------------------------
            # 선택적 파일 저장
            # ---------------------------------------------

            if save_chart:
                output_dir = Path(img_dir)

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
    signals=None,
    dpi=300,
):
    """
    최근 18개월 실제 데이터와 미래 분기 재무를
    포함한 5패널 차트를 PNG bytes로 반환합니다.

    signals: 표시할 신호 키 목록(None이면 전체).
    dpi: 화면 미리보기는 낮은 값을 써서 다시 그리는 시간을 줄입니다.
    """
    with _CHART_LOCK:
        fig = None

        try:
            fig = ChartPlotter.generate_chart(
                df,
                stock_name,
                excel_file=excel_file,
                save_chart=False,
                interactive=False,
                signals=signals,
            )

            with io.BytesIO() as buffer:
                fig.savefig(
                    buffer,
                    format="png",
                    dpi=dpi,
                    bbox_inches="tight",
                    facecolor="white",
                    transparent=False,
                )

                return buffer.getvalue()

        finally:
            if fig is not None:
                plt.close(fig)
