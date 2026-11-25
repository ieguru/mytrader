"""추가 피처: 3 수급 누적 · 4 신호 강도 · 5 밸류·규모 · 6 공매도·외국인 보유
모두 t일 장 마감까지 공개된 값만 사용. 공매도 잔고는 공시가 2거래일 늦어 SHORT_LAG만큼 밀어서 사용.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from rl.config import DATA_DIR

EXTRA_DIR = Path(DATA_DIR) / "extra"
SHORT_LAG = 2
NEW_PREFIX = ("flow_", "sig_", "val_", "size_", "short_", "frgn_")
RANK_BASE = ["sig_wosc_slope", "val_ey", "val_bp", "val_div", "size_logcap",
             "short_ratio", "short_chg20", "frgn_chg20"]


def extra_path(code: str) -> Path:
    return EXTRA_DIR / f"{code}.csv"


def load_extra(code: str, index=None):
    p = extra_path(code)
    if not p.exists():
        return None
    ex = pd.read_csv(p, index_col=0, parse_dates=True, encoding="utf-8-sig")
    ex = ex[~ex.index.duplicated(keep="last")].sort_index().apply(pd.to_numeric, errors="coerce")
    if index is not None:
        ex = ex.reindex(index).ffill(limit=5)          # 과거 값으로만 채움 (최대 5거래일)
    return ex


def _col(ex, name):
    return ex[name] if ex is not None and name in ex.columns else None


def stock_features(df: pd.DataFrame, ex=None) -> pd.DataFrame:
    c, v = df["종가"], df["거래량"].replace(0, np.nan)
    f = pd.DataFrame(index=df.index)

    # 3. 외국인·기관·개인 누적 순매수 (거래량 대비)
    for col in [x for x in df.columns if "순매수" in str(x)]:
        nb = pd.to_numeric(df[col], errors="coerce")
        name = str(col).replace("_순매수", "").replace("순매수", "").strip("_") or "수급"
        for n in (20, 60):
            f[f"flow_{name}_{n}"] = (nb.rolling(n, min_periods=n // 2).sum()
                                     / v.rolling(n, min_periods=n // 2).sum()).clip(-1, 1)
        f[f"flow_{name}_accel"] = f[f"flow_{name}_20"] - f[f"flow_{name}_60"]

    # 4. 신호 강도
    if "Weekly_Oscillator" in df.columns:
        wo = pd.to_numeric(df["Weekly_Oscillator"], errors="coerce") / c * 100
        f["sig_wosc_slope"] = wo - wo.shift(5)
    f["sig_week_vol"] = np.log(v.rolling(5).mean() / v.rolling(60).mean()).clip(-3, 3)
    f["sig_week_ret"] = np.log(c / c.shift(5))
    f["sig_week_range"] = df["고가"].rolling(5).max() / df["저가"].rolling(5).min() - 1

    if ex is not None:
        ex = ex.reindex(df.index)
        # 5. 밸류·규모
        per, pbr, div, cap = (_col(ex, k) for k in ("PER", "PBR", "DIV", "시가총액"))
        if per is not None:
            f["val_ey"] = (1 / per).where(per > 0, 0.0).where(per.notna())
            f["val_loss"] = (per <= 0).astype(float).where(per.notna())
        if pbr is not None:
            f["val_bp"] = (1 / pbr).where(pbr > 0)
        if div is not None:
            f["val_div"] = div / 100
        if cap is not None:
            f["size_logcap"] = np.log(cap.where(cap > 0))
        # 6. 공매도·외국인 보유
        sh, nsh = _col(ex, "공매도잔고"), _col(ex, "상장주식수")
        if sh is not None and nsh is not None:
            r = (sh / nsh.where(nsh > 0)).shift(SHORT_LAG)
            f["short_ratio"] = r * 100
            f["short_chg20"] = (r - r.shift(20)) * 100
        fo = _col(ex, "외국인지분율")
        if fo is not None:
            f["frgn_own"] = fo / 100
            f["frgn_chg20"] = fo - fo.shift(20)
    return f.replace([np.inf, -np.inf], np.nan)


def add_extra_features(stocks: dict) -> int:
    """각 종목 s['feat']에 추가 피처와 종목 간 순위(_rank)를 붙인다. 추가 데이터가 있는 종목 수를 반환."""
    n_ex = 0
    for code, s in stocks.items():
        ex = load_extra(code, s["dates"])
        n_ex += ex is not None
        s["feat"] = s["feat"].join(stock_features(s["df"], ex), rsuffix="_dup")
    cols = sorted({c for s in stocks.values() for c in s["feat"].columns
                   if c in RANK_BASE or (c.startswith("flow_") and c.endswith("_60"))})
    for col in cols:
        wide = pd.DataFrame({k: s["feat"][col] for k, s in stocks.items() if col in s["feat"].columns})
        if wide.shape[1] < 5:
            continue
        rank = wide.rank(axis=1, pct=True)
        for k, s in stocks.items():
            if k in rank.columns:
                s["feat"][f"{col}_rank"] = rank[k].reindex(s["dates"])
    return n_ex
