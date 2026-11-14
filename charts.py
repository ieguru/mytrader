# charts.py
"""6패널 종합차트를 고해상도 PNG bytes로 반환합니다."""

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

VOLUME_COLUMNS = {
    "외국인합계_순매수": "외국인_누적순매수수량",
    "기관합계_순매수": "기관_누적순매수수량",
    "개인_순매수": "개인_누적순매수수량",
}


def to_numeric(series):
    """숫자 문자열과 결측값을 정리합니다."""
    if not pd.api.types.is_numeric_dtype(series):
        series = (
            series.astype("string")
            .str.replace(",", "", regex=False)
            .str.strip()
        )

    return pd.to_numeric(
        series,
        errors="coerce",
    ).replace(
        [np.inf, -np.inf],
        np.nan,
    ).astype(float)


def number_formatter(value, position):
    """천 단위 쉼표 표시."""
    return f"{value:,.0f}"


def panel_message(ax, message):
    """패널 중앙에 안내 문구를 표시합니다."""
    ax.text(
        0.5,
        0.5,
        message,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=10,
        color="#6b7280",
    )


class ChartPlotter:
    """가격·거래량·MACD·재무·투자자 수급·공매도 차트"""

    # =====================================================
    # 한글 폰트
    # =====================================================

    @staticmethod
    def setup_korean_font():
        """프로젝트 또는 운영체제의 한글 폰트를 등록합니다."""
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
            if font_name in installed:
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

    # =====================================================
    # 데이터 준비
    # =====================================================

    @staticmethod
    def prepare_chart_frame(df):
        """
        날짜·가격을 정리하고 전체 조회 기간의
        투자자 누적 순매수 수량을 계산합니다.
        """
        if df is None or df.empty:
            raise ValueError("차트 데이터가 없습니다.")

        frame = df.copy()

        if isinstance(frame.index, pd.DatetimeIndex):
            dates = frame.index

        elif "날짜" in frame.columns:
            dates = pd.DatetimeIndex(
                pd.to_datetime(
                    frame["날짜"],
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
        frame = frame.loc[~frame.index.isna()].copy()
        frame = frame.sort_index()

        frame = frame.loc[
            ~frame.index.duplicated(keep="last")
        ].copy()

        if frame.empty:
            raise ValueError("유효한 날짜 데이터가 없습니다.")

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

        if "분기" not in frame.columns:
            frame["분기"] = [
                f"{date.year}.{(date.month - 1) // 3 + 1}Q"
                for date in frame.index
            ]
        else:
            frame["분기"] = (
                frame["분기"]
                .astype(str)
                .str.strip()
            )

        # 최근 2년으로 자르기 전에 전체 조회 기간을 누적합니다.
        for source, target in VOLUME_COLUMNS.items():
            if source not in frame.columns:
                continue

            daily_volume = to_numeric(
                frame[source]
            )

            frame[source] = daily_volume

            # 누락값을 0으로 간주하지 않습니다.
            # 중간 누락 이후의 정확한 누적 수량은 알 수 없으므로
            # 이후 누적값도 NaN으로 유지합니다.
            frame[target] = daily_volume.cumsum(
                skipna=False
            )

        if "공매도수량" in frame.columns:
            short_volume = to_numeric(
                frame["공매도수량"]
            )

            frame["공매도수량"] = short_volume.where(
                short_volume >= 0
            )

            # 표시 구간으로 자르기 전에 계산하여
            # 차트 시작 부분에도 이전 조회 데이터가 반영됩니다.
            frame["공매도수량_5거래일평균"] = (
                frame["공매도수량"]
                .rolling(
                    window=5,
                    min_periods=5,
                )
                .mean()
            )

        return frame

    @staticmethod
    def filter_last_2_years(df):
        """최근 730일을 표시용 데이터로 반환합니다."""
        if df is None or df.empty:
            return pd.DataFrame()

        frame = df.copy()

        frame["날짜"] = pd.to_datetime(
            frame["날짜"],
            errors="coerce",
        )

        frame = frame.dropna(subset=["날짜"])

        if frame.empty:
            return frame

        cutoff = (
            frame["날짜"].max()
            - pd.Timedelta(days=730)
        )

        return (
            frame.loc[frame["날짜"] >= cutoff]
            .reset_index(drop=True)
        )

    # =====================================================
    # 패널 1: 일봉·목표가·신호
    # =====================================================

    @staticmethod
    def plot_candlestick(ax, df, width=0.6):
        """상승 빨간색, 하락 파란색 캔들."""
        required = ["시가", "고가", "저가", "종가"]

        plot_df = df.dropna(
            subset=required
        ).copy()

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
                linewidth=0.8,
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
                linewidth=0.5,
                zorder=3,
            )

            doji = subset.loc[height.eq(0)]

            if not doji.empty:
                ax.hlines(
                    doji["종가"],
                    doji.index - width / 2,
                    doji.index + width / 2,
                    color=color,
                    linewidth=1.0,
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
        """주간 및 주간·월간 결합 신호."""
        specs = [
            (
                "Weekly_Buy",
                "^",
                "#e11d48",
                100,
                "주간 매수",
                "저가",
                0.97,
            ),
            (
                "Weekly_Sell",
                "v",
                "#2563eb",
                100,
                "주간 매도",
                "고가",
                1.03,
            ),
            (
                "Weekly_Monthly_Buy",
                "D",
                "#16a34a",
                130,
                "주간·월간 매수",
                "저가",
                0.94,
            ),
            (
                "Weekly_Monthly_Sell",
                "X",
                "#9740e8",
                130,
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
            if (
                column not in df.columns
                or price_column not in df.columns
            ):
                continue

            subset = df.loc[
                df[column].eq(1)
            ].dropna(subset=[price_column])

            if subset.empty:
                continue

            ax.scatter(
                subset.index,
                subset[price_column] * factor,
                marker=marker,
                color=color,
                s=size,
                edgecolors="black",
                linewidth=0.7,
                label=label,
                zorder=6,
            )

    @staticmethod
    def plot_price_panel(ax, df, stock_name):
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
                    linewidth=2.0,
                    alpha=0.95,
                    label="목표가",
                )

                last_index = valid.index[-1]
                last_value = float(valid.iloc[-1])

                ax.annotate(
                    f" 목표가 {last_value:,.0f}",
                    xy=(last_index, last_value),
                    xytext=(4, 4),
                    textcoords="offset points",
                    color="#ff7f0e",
                    fontsize=9,
                    fontweight="bold",
                    va="bottom",
                    ha="left",
                )

        ChartPlotter.plot_signals(ax, df)

        ax.set_title(
            f"{stock_name} - 일봉차트(최근 2년) "
            "+ 목표가 + 주간 / 주간·월간 매매 신호",
            fontsize=12,
            fontweight="bold",
            pad=10,
        )

        ax.set_ylabel(
            "가격 (원)",
            fontsize=10,
            fontweight="bold",
        )

        ax.yaxis.set_major_formatter(
            FuncFormatter(number_formatter)
        )

        ax.grid(True, alpha=0.3)

        handles, labels = (
            ax.get_legend_handles_labels()
        )

        if handles:
            ax.legend(
                handles,
                labels,
                loc="upper left",
                fontsize=8,
            )

    # =====================================================
    # 패널 2: 거래량
    # =====================================================

    @staticmethod
    def plot_volume_panel(ax, df):
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

        ax.set_ylabel(
            "거래량 (주)",
            fontsize=10,
            fontweight="bold",
        )

        ax.yaxis.set_major_formatter(
            FuncFormatter(number_formatter)
        )

        ax.grid(
            True,
            alpha=0.3,
            axis="y",
        )

    # =====================================================
    # 패널 3: MACD
    # =====================================================

    @staticmethod
    def plot_macd_panel(ax, df):
        specs = [
            ("Daily_Oscillator", "일간", "black"),
            ("Weekly_Oscillator", "주간", "#dc2626"),
            ("Monthly_Oscillator", "월간", "#16a34a"),
        ]

        plotted = False
        x_values = df.index.to_numpy(dtype=float)

        for column, label, color in specs:
            if column not in df.columns:
                continue

            values = to_numeric(
                df[column]
            ).to_numpy(dtype=float)

            if not np.isfinite(values).any():
                continue

            ax.plot(
                x_values,
                values,
                color=color,
                linewidth=1.5,
                label=label,
            )

            ax.fill_between(
                x_values,
                values,
                0,
                where=np.isfinite(values),
                color=color,
                alpha=0.12,
            )

            plotted = True

        ax.axhline(
            0,
            color="black",
            linewidth=0.8,
            linestyle="--",
        )

        ax.set_ylabel(
            "MACD\nOscillator",
            fontsize=10,
            fontweight="bold",
        )

        ax.grid(True, alpha=0.3)

        if plotted:
            ax.legend(
                loc="upper left",
                fontsize=9,
            )
        else:
            panel_message(
                ax,
                "MACD 데이터가 없습니다.",
            )

    # =====================================================
    # 패널 4: 재무
    # =====================================================

    @staticmethod
    def plot_financial_panel(
        ax,
        df,
        stock_name,
        excel_file=None,
    ):
        """영업이익·목표가·분기 최고가."""
        if not excel_file or not Path(excel_file).is_file():
            panel_message(
                ax,
                "재무 엑셀 파일이 없습니다.",
            )
            ax.set_xticks([])
            ax.set_yticks([])
            return

        financial = pd.read_excel(
            excel_file,
            sheet_name=stock_name,
        )

        required = {
            "분기",
            "영업이익",
            "목표가",
        }

        missing = required - set(financial.columns)

        if missing:
            raise ValueError(
                "재무 컬럼 누락: "
                + ", ".join(sorted(missing))
            )

        financial = financial.copy()

        financial["분기"] = (
            financial["분기"]
            .astype(str)
            .str.strip()
        )

        financial = financial.loc[
            financial["분기"].str.fullmatch(
                r"\d{4}\.[1-4]Q"
            )
        ].copy()

        if financial.empty:
            raise ValueError(
                "유효한 분기 데이터가 없습니다."
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

        financial = (
            financial
            .sort_values(["_year", "_quarter"])
            .tail(8)
            .reset_index(drop=True)
        )

        quarters = financial["분기"].tolist()

        income = to_numeric(
            financial["영업이익"]
        ).to_numpy(dtype=float)

        targets = to_numeric(
            financial["목표가"]
        ).clip(lower=0).to_numpy(dtype=float)

        quarter_highs = []

        for quarter in quarters:
            rows = df.loc[
                df["분기"].eq(quarter),
                "고가",
            ]

            value = to_numeric(rows).max()

            quarter_highs.append(
                float(value)
                if pd.notna(value)
                else np.nan
            )

        quarter_highs = np.asarray(
            quarter_highs,
            dtype=float,
        )

        x_values = np.arange(len(quarters))
        valid_income = np.isfinite(income)

        if valid_income.any():
            bars = ax.bar(
                x_values[valid_income],
                income[valid_income],
                width=0.4,
                color="skyblue",
                edgecolor="navy",
                linewidth=1.2,
                label="영업이익",
                alpha=0.8,
            )

            for bar, value in zip(
                bars,
                income[valid_income],
            ):
                ax.annotate(
                    f"{value:,.0f}",
                    xy=(
                        bar.get_x() + bar.get_width() / 2,
                        value,
                    ),
                    xytext=(
                        0,
                        3 if value >= 0 else -3,
                    ),
                    textcoords="offset points",
                    ha="center",
                    va="bottom" if value >= 0 else "top",
                    fontsize=8,
                    color="navy",
                )

            minimum = min(
                0,
                float(np.nanmin(income)),
            )

            maximum = max(
                0,
                float(np.nanmax(income)),
            )

            span = maximum - minimum

            if span == 0:
                span = max(abs(maximum), 1)

            ax.set_ylim(
                minimum - span * 0.2,
                maximum + span * 0.25,
            )

        ax.axhline(
            0,
            color="gray",
            linewidth=0.7,
            alpha=0.6,
        )

        ax.set_ylabel(
            "영업이익 (백만원)",
            fontsize=10,
            fontweight="bold",
            color="navy",
        )

        ax.tick_params(
            axis="y",
            labelcolor="navy",
        )

        right_ax = ax.twinx()

        valid_targets = np.isfinite(targets)

        if valid_targets.any():
            right_ax.plot(
                x_values,
                targets,
                marker="^",
                color="#ff7f0e",
                linewidth=2.0,
                markersize=7,
                linestyle="--",
                label="목표가",
            )

            for x_value, value in zip(
                x_values[valid_targets],
                targets[valid_targets],
            ):
                right_ax.annotate(
                    f"{value:,.0f}",
                    xy=(x_value, value),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    color="#ff7f0e",
                )

        valid_highs = np.isfinite(quarter_highs)

        if valid_highs.any():
            right_ax.plot(
                x_values,
                quarter_highs,
                marker="s",
                color="black",
                linewidth=2.0,
                markersize=6,
                label="분기 최고가(표시 구간)",
            )

            for x_value, value in zip(
                x_values[valid_highs],
                quarter_highs[valid_highs],
            ):
                right_ax.annotate(
                    f"{value:,.0f}",
                    xy=(x_value, value),
                    xytext=(0, -6),
                    textcoords="offset points",
                    ha="center",
                    va="top",
                    fontsize=8,
                    color="black",
                )

        prices = np.concatenate([
            targets[valid_targets],
            quarter_highs[valid_highs],
        ])

        if len(prices):
            minimum = float(prices.min())
            maximum = float(prices.max())
            span = maximum - minimum

            if span == 0:
                span = max(abs(maximum) * 0.1, 1)

            right_ax.set_ylim(
                max(0, minimum - span * 0.2),
                maximum + span * 0.3,
            )

        right_ax.set_ylabel(
            "주가 (원)",
            fontsize=10,
            fontweight="bold",
        )

        right_ax.yaxis.set_major_formatter(
            FuncFormatter(number_formatter)
        )

        ax.set_xticks(x_values)

        ax.set_xticklabels(
            quarters,
            rotation=30,
            ha="right",
            fontsize=8,
        )

        ax.set_xlim(
            -0.5,
            len(quarters) - 0.5,
        )

        handles_left, labels_left = (
            ax.get_legend_handles_labels()
        )

        handles_right, labels_right = (
            right_ax.get_legend_handles_labels()
        )

        if handles_left or handles_right:
            ax.legend(
                handles_left + handles_right,
                labels_left + labels_right,
                loc="upper left",
                fontsize=8,
                framealpha=0.95,
            )

        ax.set_title(
            "영업이익(좌축) / 목표가·분기 최고가(우축)",
            fontsize=11,
            fontweight="bold",
            pad=10,
        )

        ax.grid(True, alpha=0.3)

    # =====================================================
    # 패널 5: 외국인·기관·개인 누적 순매수 수량
    # =====================================================

    @staticmethod
    def plot_investor_volume_panel(
        ax,
        df,
        cumulative_start,
    ):
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
                linewidth=2.2,
                label=f"{label} {last_value:,.0f}주",
            )

            plotted = True

        ax.axhline(
            0,
            color="#6b7280",
            linewidth=0.9,
            linestyle="--",
        )

        ax.set_title(
            "외국인·기관·개인 누적 순매수 수량 "
            f"(누적 시작: {cumulative_start})",
            fontsize=11,
            fontweight="bold",
            pad=10,
        )

        ax.set_ylabel(
            "누적 순매수 (주)",
            fontsize=10,
            fontweight="bold",
        )

        ax.yaxis.set_major_formatter(
            FuncFormatter(number_formatter)
        )

        ax.grid(True, alpha=0.3)

        if plotted:
            ax.legend(
                loc="best",
                fontsize=9,
                framealpha=0.95,
            )

            valid_through = min(
                valid_dates
            ).strftime("%Y-%m-%d")

            note = (
                f"누적수량 유효 기준일: {valid_through}"
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
                fontsize=8,
                color="#6b7280",
            )

        else:
            panel_message(
                ax,
                "누적 순매수 수량을 표시할 수 없습니다.\n"
                "투자자 데이터 포함 여부, KRX 인증 또는 "
                "데이터 누락을 확인하세요.",
            )

    # =====================================================
    # 패널 6: 일별 공매도 거래 수량
    # =====================================================

    @staticmethod
    def plot_shorting_volume_panel(ax, df):
        """공매도 잔고가 아닌 일별 거래 수량을 표시합니다."""
        ax.set_title(
            "일별 공매도 거래 수량",
            fontsize=11,
            fontweight="bold",
            pad=10,
        )

        ax.set_ylabel(
            "공매도 수량 (주)",
            fontsize=10,
            fontweight="bold",
        )

        ax.set_xlabel(
            "거래일",
            fontsize=10,
            fontweight="bold",
        )

        ax.yaxis.set_major_formatter(
            FuncFormatter(number_formatter)
        )

        ax.grid(
            True,
            alpha=0.3,
            axis="y",
        )

        if "공매도수량" not in df.columns:
            panel_message(
                ax,
                "공매도 거래 수량 데이터가 없습니다.\n"
                "KRX 인증 및 데이터 조회 상태를 확인하세요.",
            )
            return

        values = to_numeric(
            df["공매도수량"]
        )

        values = values.where(values >= 0)
        valid = values.dropna()

        if valid.empty:
            panel_message(
                ax,
                "유효한 공매도 거래 수량이 없습니다.",
            )
            return

        ax.bar(
            valid.index,
            valid,
            width=0.75,
            color="#7c3aed",
            alpha=0.75,
            label="공매도 거래 수량",
        )

        if "공매도수량_5거래일평균" in df.columns:
            average = to_numeric(
                df["공매도수량_5거래일평균"]
            )
        else:
            average = values.rolling(
                window=5,
                min_periods=5,
            ).mean()

        if average.notna().any():
            ax.plot(
                df.index,
                average,
                color="#f97316",
                linewidth=1.8,
                label="5거래일 평균",
            )

        ax.set_ylim(bottom=0)

        ax.legend(
            loc="upper left",
            fontsize=9,
            framealpha=0.95,
        )

        last_index = valid.index[-1]
        last_value = float(valid.iloc[-1])

        last_date = pd.Timestamp(
            df.loc[last_index, "날짜"]
        ).strftime("%Y-%m-%d")

        note = (
            f"최근 유효 데이터: {last_date}"
            f" / {last_value:,.0f}주"
        )

        if values.isna().any():
            note += " · 일부 날짜 미조회"

        ax.text(
            0.99,
            0.95,
            note,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8,
            color="#6b7280",
        )

    # =====================================================
    # 종합차트 생성
    # =====================================================

    @staticmethod
    def generate_chart(
        df,
        stock_name,
        excel_file=None,
        save_chart=False,
        interactive=False,
        img_dir="img",
    ):
        """6패널 Figure를 생성합니다."""
        ChartPlotter.setup_korean_font()

        full_frame = ChartPlotter.prepare_chart_frame(df)

        cumulative_start = full_frame.index[0].strftime(
            "%Y-%m-%d"
        )

        plot_df = ChartPlotter.filter_last_2_years(
            full_frame
        )

        if plot_df.empty:
            raise ValueError(
                "차트에 표시할 데이터가 없습니다."
            )

        if interactive:
            plt.ion()

        fig = plt.figure(figsize=(18, 19))

        try:
            grid = fig.add_gridspec(
                6,
                1,
                height_ratios=[
                    2.5,
                    0.8,
                    1.2,
                    1.1,
                    1.2,
                    1.0,
                ],
                hspace=0.55,
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

            # 재무 패널은 분기 X축을 사용합니다.
            ax4 = fig.add_subplot(grid[3])

            ax5 = fig.add_subplot(
                grid[4],
                sharex=ax1,
            )

            ax6 = fig.add_subplot(
                grid[5],
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

            ChartPlotter.plot_macd_panel(
                ax3,
                plot_df,
            )

            # 재무 오류가 다른 패널 생성을 막지 않게 합니다.
            try:
                ChartPlotter.plot_financial_panel(
                    ax4,
                    plot_df,
                    stock_name,
                    excel_file,
                )
            except Exception:
                panel_message(
                    ax4,
                    "재무 데이터를 표시하지 못했습니다.\n"
                    "엑셀 시트명과 분기·목표가·영업이익을 확인하세요.",
                )

            ChartPlotter.plot_investor_volume_panel(
                ax5,
                plot_df,
                cumulative_start,
            )

            ChartPlotter.plot_shorting_volume_panel(
                ax6,
                plot_df,
            )

            # 거래일 X축 설정
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

            ax6.set_xticks(tick_positions)

            ax6.set_xticklabels(
                tick_labels,
                rotation=35,
                ha="right",
                fontsize=8,
            )

            ax1.set_xlim(
                -1,
                len(plot_df),
            )

            # 공통 날짜는 맨 아래 패널에 표시합니다.
            for ax in (ax1, ax2, ax3, ax5):
                ax.tick_params(
                    axis="x",
                    labelbottom=False,
                )

            fig.subplots_adjust(
                left=0.10,
                right=0.90,
                top=0.97,
                bottom=0.06,
            )

            # app.py에서는 False로 호출하므로 자동 저장하지 않습니다.
            if save_chart:
                output_dir = Path(img_dir)

                output_dir.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                save_date = pd.Timestamp.now(
                    tz="Asia/Seoul"
                ).strftime("%Y%m%d")

                filename = (
                    output_dir
                    / f"{stock_name}_{save_date}.png"
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
    """6패널 종합차트를 고해상도 PNG bytes로 반환합니다."""
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

            if fig is None:
                raise ValueError(
                    "차트에 표시할 데이터가 없습니다."
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
