"""정규화된 피처. 모두 t일 장 마감 시점에 알 수 있는 값만 씁니다."""
import numpy as np
import pandas as pd

SIGNAL_MAP = {
    "Weekly_Buy": "sig_w_buy",
    "Weekly_Sell": "sig_w_sell",
    "Weekly_Monthly_Buy": "sig_wm_buy",
    "Weekly_Monthly_Sell": "sig_wm_sell",
    "이동평균매수": "sig_ma_buy",
    "이동평균매도": "sig_ma_sell",
    "유목민신호": "sig_nomad_buy",
    "유목민매도신호": "sig_nomad_sell",
    "쌍바닥신호": "sig_dbl_bottom",
    "쌍고점신호": "sig_dbl_top",
}

INVESTOR_MAP = {
    "외국인합계_순매수": "net_foreign",
    "기관합계_순매수": "net_inst",
    "개인_순매수": "net_indiv",
}

MARKET_COLS = ["kospi_ret", "kosdaq_ret", "kospi_vol20"]

CONT_COLS = [
    "ret1", "vol_chg",
    "daily_osc", "weekly_osc", "monthly_osc", "daily_osc_chg",
    "rsi30", "disp20", "disp5",
    *INVESTOR_MAP.values(),
    *MARKET_COLS,
]
BIN_COLS = ["weekly_trend", "monthly_trend", *SIGNAL_MAP.values()]
FEATURE_COLS = CONT_COLS + BIN_COLS


def build_features(data, market=None):
    f = pd.DataFrame(index=data.index)
    close = pd.to_numeric(data["종가"], errors="coerce")
    volume = pd.to_numeric(data["거래량"], errors="coerce").replace(0, np.nan)

    f["ret1"] = np.log(close).diff()
    f["vol_chg"] = np.log(volume).diff().clip(-3, 3)

    # MACD를 가격 대비 %로 바꿔 종목 간 크기를 맞춥니다.
    for prefix in ("Daily", "Weekly", "Monthly"):
        f[f"{prefix.lower()}_osc"] = data[f"{prefix}_Oscillator"] / close * 100

    # 주간·월간은 '전일 대비 변화' 대신 직전 완료 봉 기준 추세(+1/-1/0)를 씁니다.
    f["daily_osc_chg"] = f["daily_osc"].diff()
    f["weekly_trend"] = data["Weekly_Trend"]
    f["monthly_trend"] = data["Monthly_Trend"]

    f["rsi30"] = data["RSI30"] / 100 - 0.5
    f["disp20"] = data["이격도20"] / 100 - 1
    f["disp5"] = data["이격도5"] / 100 - 1

    # 순매수 '증감률'은 0이나 음수일 때 깨지므로 거래량 대비 비율로 씁니다.
    for src, dst in INVESTOR_MAP.items():
        f[dst] = (
            (pd.to_numeric(data[src], errors="coerce") / volume).clip(-1, 1)
            if src in data
            else np.nan
        )

    m = market.reindex(f.index) if market is not None else pd.DataFrame(index=f.index)
    for column in MARKET_COLS:
        f[column] = m[column] if column in m else np.nan

    for src, dst in SIGNAL_MAP.items():
        f[dst] = (
            pd.to_numeric(data[src], errors="coerce").fillna(0) if src in data else 0.0
        )

    return f[FEATURE_COLS].astype(float)


class Scaler:
    """연속형 피처만 z-score로 바꿉니다. 통계는 학습 구간에서만 계산합니다."""

    def fit(self, frames):
        stacked = pd.concat([f[CONT_COLS] for f in frames])
        self.mean = stacked.mean()
        self.std = stacked.std().replace(0, 1).fillna(1)
        return self

    def transform(self, f):
        out = f.copy()
        out[CONT_COLS] = ((f[CONT_COLS] - self.mean) / self.std).clip(-5, 5)
        return out.fillna(0.0)   # 장 마감 전 미집계 수급 등 결측은 0(평균)
