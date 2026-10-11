"""기존 5패널 차트를 PNG bytes로 반환합니다."""
import io
import os
import platform
import threading
from datetime import datetime, timedelta

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
import mplfinance as mpf
import numpy as np
import pandas as pd
from PIL import Image

_CHART_LOCK = threading.RLock()

class ChartPlotter:
    """KOSPI 종목 종합 차트 생성 클래스"""

    def setup_korean_font():
        """Windows / Linux / macOS에서 한글 폰트를 등록합니다."""
        from pathlib import Path
        import os
        import platform
        import matplotlib.pyplot as plt
        from matplotlib import font_manager

        base_dir = Path(__file__).resolve().parent
        system = platform.system()

        candidates = []

        # 프로젝트에 포함한 폰트가 있으면 우선 사용
        candidates.extend([
            base_dir / "fonts" / "NanumGothic.ttf",
            base_dir / "fonts" / "NotoSansKR-Regular.ttf",
        ])

        if system == "Windows":
            windows_dir = Path(os.environ.get("WINDIR", r"C:\Windows"))
            candidates.extend([
                windows_dir / "Fonts" / "malgun.ttf",
                windows_dir / "Fonts" / "malgunbd.ttf",
            ])

        elif system == "Darwin":
            candidates.extend([
                Path("/System/Library/Fonts/Supplemental/AppleGothic.ttf"),
                Path("/Library/Fonts/AppleGothic.ttf"),
            ])

        else:
            # Streamlit Community Cloud / GitHub Actions
            candidates.extend([
                Path(
                    "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"
                ),
                Path(
                    "/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf"
                ),
            ])

        for font_path in candidates:
            if not font_path.is_file():
                continue

            # 운영체제의 폰트 검색 캐시에 의존하지 않고 직접 등록
            font_manager.fontManager.addfont(str(font_path))

            font_name = font_manager.FontProperties(
                fname=str(font_path)
            ).get_name()

            plt.rcParams["font.family"] = [font_name]
            plt.rcParams["font.sans-serif"] = [font_name]
            plt.rcParams["axes.unicode_minus"] = False

            return font_name

        raise RuntimeError(
            "한글 폰트를 찾지 못했습니다. "
            "Community Cloud에서는 packages.txt에 fonts-nanum을 추가하거나, "
            "프로젝트의 fonts 폴더에 NanumGothic.ttf를 넣으세요."
        )

    @staticmethod
    def plot_candlestick(ax, df, width=0.6):
        """
        일봉차트 그리기 (상승: 빨강, 하락: 파랑)
    
        수정 이유:
        오늘 장중 행에 일부 결측이 섞이면 캔들이 누락될 수 있으므로
        차트 직전에 OHLC 숫자형 정리 + 결측 제거를 수행함.
        """
        plot_df = df.copy()
        required_cols = ['시가', '고가', '저가', '종가']
        for col in required_cols:
            plot_df[col] = pd.to_numeric(plot_df[col], errors='coerce')
        plot_df = plot_df.dropna(subset=required_cols)
        if len(plot_df) == 0:
            ax.text(0.5, 0.5, '표시할 캔들 데이터가 없습니다.', ha='center', va='center')
            return
        up = plot_df[plot_df['종가'] >= plot_df['시가']]
        down = plot_df[plot_df['종가'] < plot_df['시가']]
        ax.bar(up.index, up['종가'] - up['시가'], width, bottom=up['시가'], color='red', edgecolor='darkred', linewidth=0.5)
        for i in up.index:
            ax.plot([i, i], [up.loc[i, '저가'], up.loc[i, '고가']], color='red', linewidth=0.8)
        ax.bar(down.index, down['시가'] - down['종가'], width, bottom=down['종가'], color='blue', edgecolor='darkblue', linewidth=0.5)
        for i in down.index:
            ax.plot([i, i], [down.loc[i, '저가'], down.loc[i, '고가']], color='blue', linewidth=0.8)
        try:
            last_idx = plot_df.index.max()
            ax.axvspan(last_idx - 0.5, last_idx + 0.5, color='gold', alpha=0.08)
        except:
            pass

    @staticmethod
    def plot_signals(ax, df):
        """매매신호 표시 (등급 구분 없이 매수/매도만)

        - Weekly 매수/매도        : Weekly_Trend 전환(-1<->1) 신호
        - Weekly+Monthly 매수/매도: Weekly_Trend 방향 + Monthly_Trend 전환 결합 신호
        - 강력 매수 / 적극 매수 / 매수 / 적극 매도 / 매도 등급 마커는 삭제됨
        """
        specs = [('Weekly_Buy', '^', '#e11d48', 150, 'Weekly 매수', '저가', 0.97), ('Weekly_Sell', 'v', '#2563eb', 150, 'Weekly 매도', '고가', 1.03), ('Weekly_Monthly_Buy', 'D', '#16a34a', 130, 'Weekly+Monthly 매수', '저가', 0.94), ('Weekly_Monthly_Sell', 'X', '#7c3aed', 130, 'Weekly+Monthly 매도', '고가', 1.06)]
        for col, marker, color, size, label, price_col, factor in specs:
            if col not in df.columns or price_col not in df.columns:
                continue
            subset = df[df[col] == 1]
            if len(subset) == 0:
                continue
            ax.scatter(subset.index, subset[price_col] * factor, marker=marker, color=color, s=size, edgecolors='black', linewidth=0.8, label=label, zorder=6)

    @staticmethod
    def filter_last_2_years(df):
        """최근 2년 데이터만 필터링 (730일)"""
        if len(df) == 0:
            return df
        df = df.copy()
        df['날짜'] = pd.to_datetime(df['날짜'], errors='coerce')
        df = df.dropna(subset=['날짜'])
        if len(df) == 0:
            print('❌ 유효한 날짜 데이터가 없습니다.')
            return df
        cutoff_date = df['날짜'].max() - timedelta(days=730)
        df_filtered = df[df['날짜'] >= cutoff_date].reset_index(drop=True)
        if len(df_filtered) > 0:
            start_date = df_filtered['날짜'].min().strftime('%Y-%m-%d')
            end_date = df_filtered['날짜'].max().strftime('%Y-%m-%d')
        return df_filtered

    @staticmethod
    def generate_chart(df, stock_name, excel_file=None, save_chart=True, interactive=True, img_dir='img'):
        """5패널 종합 차트 생성 (일봉 + 거래량 + MACD + 재무 + 매매주체 순매수)"""
        ChartPlotter.setup_korean_font()
        df_2year = ChartPlotter.filter_last_2_years(df)
        if len(df_2year) == 0:
            print('❌ 표시할 데이터가 없습니다.')
            return None
        df_2year = df_2year.reset_index(drop=True)
        if interactive:
            plt.ion()
        fig = plt.figure(figsize=(18, 16))
        gs = fig.add_gridspec(5, 1, height_ratios=[2.5, 0.8, 1.2, 0.8, 1.0], hspace=0.35)
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1], sharex=ax1)
        ax3 = fig.add_subplot(gs[2], sharex=ax1)
        ax4 = fig.add_subplot(gs[3])
        ax5 = fig.add_subplot(gs[4], sharex=ax1)
        ChartPlotter.plot_candlestick(ax1, df_2year)
        if '목표가' in df_2year.columns:
            target_series = (pd.to_numeric(df_2year["목표가"], errors="coerce").clip(lower=0))
            valid_target = target_series.notna()
            if valid_target.any():
                ax1.plot(df_2year.index[valid_target], target_series[valid_target], color='#ff7f0e', linestyle='--', linewidth=2.0, alpha=0.95, label='목표가')
                last_target_idx = target_series[valid_target].index[-1]
                last_target_val = float(target_series.loc[last_target_idx])
                ax1.text(last_target_idx, last_target_val, f' 목표가 {int(last_target_val):,}', color='#ff7f0e', fontsize=9, fontweight='bold', va='bottom', ha='left', bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='#ffb366', alpha=0.75))
        ChartPlotter.plot_signals(ax1, df_2year)
        ax1.set_ylabel('가격 (원)', fontsize=10, fontweight='bold')
        ax1.set_title(f'{stock_name} - 일봉차트 (최근 2년) + 목표가 + Weekly / Weekly+Monthly 매매신호', fontsize=12, fontweight='bold', pad=10)
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper left', fontsize=9)
        colors = ['red' if df_2year['종가'].iloc[i] >= df_2year['시가'].iloc[i] else 'blue' for i in range(len(df_2year))]
        ax2.bar(df_2year.index, df_2year['거래량'], color=colors, alpha=0.7)
        ax2.set_ylabel('거래량', fontsize=10, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        for label, col, style in [('Daily_Oscillator', 'black', '-'), ('Weekly_Oscillator', 'red', '-'), ('Monthly_Oscillator', 'green', '-')]:
            ax3.plot(df_2year.index, df_2year[label], label=label.split('_')[0], color=col, linewidth=1.5, linestyle=style)
            ax3.fill_between(df_2year.index, df_2year[label], 0, where=df_2year[label] >= 0, color=col, alpha=0.15)
            ax3.fill_between(df_2year.index, df_2year[label], 0, where=df_2year[label] < 0, color=col, alpha=0.15)
        ax3.axhline(0, color='black', linewidth=0.8, linestyle='--')
        ax3.set_ylabel('MACD Oscillator', fontsize=10, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.legend(loc='upper left', fontsize=9)
        try:
            if excel_file and os.path.exists(excel_file):
                df_financial = pd.read_excel(excel_file, sheet_name=stock_name)

                def quarter_to_sort_key(q):
                    try:
                        year, quarter = q.split('.')
                        return (int(year), int(quarter.replace('Q', '')))
                    except:
                        return (0, 0)
                df_financial['sort_key'] = df_financial['분기'].apply(quarter_to_sort_key)
                df_financial = df_financial.sort_values('sort_key')
                df_financial = df_financial.tail(8).reset_index(drop=True)
                quarters = df_financial['분기'].tolist()
                operating_income = df_financial['영업이익'].values.astype(float)
                target_prices = (pd.to_numeric(df_financial["목표가"], errors="coerce").clip(lower=0).to_numpy(dtype=float))
                current_prices_by_quarter = []
                for _, row in df_financial.iterrows():
                    quarter = row['분기']
                    quarter_data = df_2year[df_2year['분기'] == quarter]
                    if len(quarter_data) > 0:
                        max_price = quarter_data['고가'].max()
                    else:
                        max_price = df_2year['종가'].iloc[-1]
                    current_prices_by_quarter.append(max_price)
                x_pos = np.arange(len(quarters))
                width = 0.4
                bars = ax4.bar(x_pos, operating_income, width=width, color='skyblue', edgecolor='navy', linewidth=1.5, label='영업이익', alpha=0.8)
                ax4.set_ylabel('영업이익 (백만원)', fontsize=10, fontweight='bold', color='navy')
                ax4.tick_params(axis='y', labelcolor='navy')
                for i, (bar, val) in enumerate(zip(bars, operating_income)):
                    if pd.notna(val):
                        ax4.text(bar.get_x() + bar.get_width() / 2, val + operating_income.max() * 0.02, f'{int(val):,}', ha='center', va='bottom', fontsize=8, fontweight='bold', color='navy')
                ax4.set_ylim(0, operating_income.max() * 1.15)
                ax4_right = ax4.twinx()
                valid_target_mask = pd.notna(target_prices)
                if valid_target_mask.any():
                    ax4_right.plot(x_pos[valid_target_mask], target_prices[valid_target_mask], marker='^', color='red', linewidth=2.5, markersize=8, label='목표가', linestyle='--', alpha=0.9)
                    for i, price in enumerate(target_prices):
                        if pd.notna(price):
                            ax4_right.text(i, price + max(current_prices_by_quarter) * 0.02, f'{int(price):,}', ha='center', va='bottom', fontsize=8, fontweight='bold', color='red')
                ax4_right.plot(x_pos, current_prices_by_quarter, marker='s', color='black', linewidth=2.5, markersize=8, label='현재가(분기최고)', linestyle='-', alpha=0.9)
                for i, price in enumerate(current_prices_by_quarter):
                    ax4_right.text(i, price - max(current_prices_by_quarter) * 0.02, f'{int(price):,}', ha='center', va='top', fontsize=8, fontweight='bold', color='black')
                ax4_right.set_ylabel('주가 (원)', fontsize=10, fontweight='bold', color='black')
                ax4_right.tick_params(axis='y', labelcolor='black')
                all_prices = list(target_prices[valid_target_mask]) + current_prices_by_quarter
                min_price = min(all_prices)
                max_price = max(all_prices)
                price_range = max_price - min_price
                ax4_right.set_ylim(min_price - price_range * 0.1, max_price + price_range * 0.15)
                ax4.set_xlabel('분기 (과거 → 현재)', fontsize=10, fontweight='bold')
                ax4.set_xticks(x_pos)
                ax4.set_xticklabels(quarters, rotation=45, ha='right', fontsize=8)
                ax4.set_xlim(-0.5, len(quarters) - 0.5)
                lines1, labels1 = ax4.get_legend_handles_labels()
                lines2, labels2 = ax4_right.get_legend_handles_labels()
                ax4.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=9, framealpha=0.95)
                for i in range(0, len(quarters), 2):
                    ax4.axvspan(i - 0.5, i + 0.5, alpha=0.05, color='gray')
                ax4.set_title('영업이익(좌선) / 목표가(주황선△)·현재가(빨강점선■)', fontsize=11, fontweight='bold', pad=10)
                ax4.grid(True, alpha=0.3)
            else:
                ax4.text(0.5, 0.5, '엑셀 파일을 찾을 수 없습니다.', ha='center', va='center', fontsize=10)
                ax4.set_xticks([])
                ax4.set_yticks([])
        except Exception as e:
            ax4.text(0.5, 0.5, f'재무데이터 오류: {str(e)}', ha='center', va='center', fontsize=10)
            ax4.set_xticks([])
            ax4.set_yticks([])
        try:
            investor_cols_net = [col for col in df_2year.columns if '순매수' in col]
            colors_investor = {'개인': 'blue', '기관합계': 'black', '외국인합계': 'red', '기타법인': 'green'}
            if len(investor_cols_net) > 0:
                for col in investor_cols_net:
                    investor_name = col.replace('_순매수', '')
                    cumsum_data = df_2year[col].fillna(0).cumsum()
                    color = colors_investor.get(investor_name, '#888888')
                    ax5.plot(df_2year.index, cumsum_data, label=f'{investor_name}', linewidth=2.5, color=color, alpha=0.85, marker='', markersize=2)
                ax5.axhline(0, color='black', linewidth=1, linestyle='-', alpha=0.5)
                ax5.set_ylabel('누적 순매수 (주)', fontsize=10, fontweight='bold')
                ax5.set_xlabel('거래일', fontsize=10, fontweight='bold')
                ax5.set_title('매매주체별 누적 순매수 수량 (2년 기준)', fontsize=11, fontweight='bold', pad=10)
                ax5.grid(True, alpha=0.3)
                ax5.legend(loc='best', fontsize=9, framealpha=0.95)
                y_lim = ax5.get_ylim()
                y_max = max(abs(y_lim[0]), abs(y_lim[1]))
                if y_max > 1000000000.0:
                    ax5.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x / 1000000000.0)}B'))
                elif y_max > 1000000.0:
                    ax5.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x / 1000000.0)}M'))
                elif y_max > 1000.0:
                    ax5.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x / 1000.0)}K'))
            else:
                ax5.text(0.5, 0.5, '순매수 데이터를 찾을 수 없습니다.', ha='center', va='center', fontsize=10)
                ax5.set_xticks([])
                ax5.set_yticks([])
        except Exception as e:
            ax5.text(0.5, 0.5, f'매매주체 데이터 오류: {str(e)}', ha='center', va='center', fontsize=10)
            ax5.set_xticks([])
            ax5.set_yticks([])
        tick_interval = max(1, len(df_2year) // 10)
        ax1.set_xticks(np.arange(0, len(df_2year), tick_interval))
        date_labels = [df_2year['날짜'].iloc[i].strftime('%Y-%m-%d') if i < len(df_2year) and pd.notna(df_2year['날짜'].iloc[i]) else '' for i in np.arange(0, len(df_2year), tick_interval)]
        ax1.set_xticklabels(date_labels, rotation=45, ha='right', fontsize=8)
        plt.tight_layout()
        chart_filename = None
        if save_chart:
            try:
                os.makedirs(img_dir, exist_ok=True)
                save_date = datetime.now().strftime('%Y%m%d')
                chart_filename = os.path.join(img_dir, f'{stock_name}_{save_date}.gif')
                fig.savefig(chart_filename, dpi=150, bbox_inches='tight', format='jpg')
                print(f'✅ 차트 저장 완료: {chart_filename}')
            except Exception as e:
                chart_filename = None
                print(f'⚠️ 차트 저장 오류: {e}')
        return fig

def make_chart_png(df, stock_name, excel_file=None):
    """종합차트를 고해상도 PNG 데이터로 생성합니다."""
    with _CHART_LOCK:
        try:
            fig = ChartPlotter.generate_chart(
                df,
                stock_name,
                excel_file=excel_file,
                save_chart=False,
                interactive=False,
            )

            if fig is None:
                raise ValueError("차트에 표시할 데이터가 없습니다.")

            # 기존 종합차트 크기 유지
            fig.set_size_inches(18, 16)

            buffer = io.BytesIO()

            fig.savefig(
                buffer,
                format="png",
                dpi=300,                 # 기존 120 → 300
                bbox_inches="tight",
                facecolor="white",
                transparent=False,
            )

            return buffer.getvalue()

        finally:
            plt.close("all")

