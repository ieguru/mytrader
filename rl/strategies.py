# -*- coding: utf-8 -*-
"""rl/strategies.py : 매매 신호 전략 공통 모듈 (run_filter_seq.py 인터페이스)

0. 기본 원칙
  - 매수/매도는 번갈아 발생(쌍). 보유 중 매수신호, 현금 중 매도신호는 무시
  - t일 신호 -> t+1일 시초가 체결
  - 보유 중이면 구간 마지막 거래일 종가로 평가
  - 매수 시 수수료 0.3%(BUY_FEE)
1 단순 보유 / 2 이동평균(단순) / 3 이동평균(RSI: 이격도20 조건) / 주간 신호
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from rl.config import PROCESSED_DIR

BUY_FEE = 0.003
TRADING_DAYS = 252

BASE_KEYS = ("hold", "ma", "ma_disp", "weekly")
NAMES = {
    "hold": "1. 단순 보유",
    "ma": "2. 이동평균(단순)",
    "ma_disp": "3. 이동평균(RSI)",
    "weekly": "주간 신호(전부 따름)",
    "lgbm": "4. LightGBM 필터",
    "ppo": "5. PPO 필터",
    "random": "무작위 필터",
}
ORDER = [NAMES[k] for k in ("hold", "ma", "ma_disp", "weekly", "lgbm", "ppo", "random")]


# ------------------------------------------------------------ 파일
def select_files(universe=None, tickers=None, max_stocks=None, *_, **__):
    """processed/ 종목 CSV 경로 목록(종목코드 순)."""
    files = sorted(p for p in Path(PROCESSED_DIR).glob("*.csv") if len(p.stem) == 6)
    codes = None
    if universe:
        try:
            from rl.storage import read_universe
            u = read_universe(universe)
            col = next((c for c in u.columns if "코드" in str(c) or "code" in str(c).lower()),
                       u.columns[0])
            codes = {str(c).zfill(6) for c in u[col]}
        except Exception as e:
            print(f"[경고] 유니버스 '{universe}' 읽기 실패({e}) - 전체 종목 사용")
    if tickers:
        t = {str(c).zfill(6) for c in tickers}
        codes = t if codes is None else codes & t
    if codes is not None:
        files = [p for p in files if p.stem in codes]
    if max_stocks:
        files = files[: int(max_stocks)]
    return files


def load_processed(path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig", index_col=0, parse_dates=True)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    for c in ("시가", "고가", "저가", "종가", "거래량"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def index_range(dates, start, end):
    """[start, end] 구간의 (첫 위치 lo, 마지막 위치 hi). hi는 포함(유효 인덱스)."""
    d = pd.DatetimeIndex(dates)
    lo = int(d.searchsorted(pd.Timestamp(start), side="left"))
    hi = int(d.searchsorted(pd.Timestamp(end), side="right")) - 1
    return lo, hi


def new_state() -> dict:
    return dict(pos=0, sig=-1, entry=-1, entry_px=np.nan, skips=0,
                prev_ret=0.0, last_exit=-10**9)


# ------------------------------------------------------------ 신호 (모두 t일까지 데이터만 사용)
def _ma(df):
    c = df["종가"]
    return c, c.rolling(5).mean(), c.rolling(20).mean()


def ma_signals(df):
    """2. 매수: MA5가 MA20 상향 돌파 / 매도: 음봉 AND 종가 <= MA5"""
    c, ma5, ma20 = _ma(df)
    buy = (ma5 > ma20) & (ma5.shift(1) <= ma20.shift(1))
    sell = (c < df["시가"]) & (c <= ma5)
    return buy.fillna(False).to_numpy(bool), sell.fillna(False).to_numpy(bool)


def ma_disp_signals(df):
    """3. 매수: 골든크로스 AND 이격도20>=100 / 매도: 음봉 AND 종가<=MA5 AND 이격도20>=100"""
    c, ma5, ma20 = _ma(df)
    disp = c / ma20 * 100
    buy = (ma5 > ma20) & (ma5.shift(1) <= ma20.shift(1)) & (disp >= 100)
    sell = (c < df["시가"]) & (c <= ma5) & (disp >= 100)
    return buy.fillna(False).to_numpy(bool), sell.fillna(False).to_numpy(bool)


def weekly_events(df):
    """(매수 bool 배열, 매도 bool 배열, 출처). 엔진 열 우선, 없으면 주봉 MACD(5,13,5) 직접 계산."""
    if {"Weekly_Buy", "Weekly_Sell"} <= set(df.columns):
        b = pd.to_numeric(df["Weekly_Buy"], errors="coerce").fillna(0) > 0
        s = pd.to_numeric(df["Weekly_Sell"], errors="coerce").fillna(0) > 0
        return b.to_numpy(bool), s.to_numpy(bool), "Weekly_Buy/Weekly_Sell"
    if "Weekly_Trend" in df.columns:
        tr = pd.to_numeric(df["Weekly_Trend"], errors="coerce")
        src = "Weekly_Trend"
    else:
        per = df.index.to_period("W-FRI")
        wk = df["종가"].groupby(per).last()
        macd = wk.ewm(span=5, adjust=False).mean() - wk.ewm(span=13, adjust=False).mean()
        osc = macd - macd.ewm(span=5, adjust=False).mean()
        trw = np.sign(osc).shift(1)              # 이번 주는 지난주(완료된 주)까지 값 사용
        tr = pd.Series(trw.reindex(per).to_numpy(), index=df.index)
        src = "종가로 직접 계산"
    prev = tr.shift(1)
    return (((prev == -1) & (tr == 1)).to_numpy(bool),
            ((prev == 1) & (tr == -1)).to_numpy(bool), src)


def _state_from(buy, sell):
    pos = np.zeros(len(buy), int)
    h = 0
    for i in range(len(buy)):
        if h == 0 and buy[i]:
            h = 1
        elif h == 1 and sell[i]:
            h = 0
        pos[i] = h
    return pos


def prepare_stock(df: pd.DataFrame, code: str) -> dict:
    df = df[(df["시가"] > 0) & (df["종가"] > 0)].copy()
    if len(df) < 60:
        raise ValueError("데이터 부족")
    wb, ws, src = weekly_events(df)
    sig = {"weekly": {"buy": wb, "sell": ws,
                      "buy_idx": np.flatnonzero(wb), "sell_idx": np.flatnonzero(ws),
                      "ev_idx": np.flatnonzero(wb | ws)}}
    for key, fn in (("ma", ma_signals), ("ma_disp", ma_disp_signals)):
        b, s = fn(df)
        sig[key] = {"buy": b, "sell": s, "buy_idx": np.flatnonzero(b), "sell_idx": np.flatnonzero(s)}
    return dict(code=code, df=df, dates=df.index, open=df["시가"].to_numpy(float),
                close=df["종가"].to_numpy(float), sig=sig, src=src,
                state_w=_state_from(wb, ws))


# ------------------------------------------------------------ 시뮬레이션
def _simulate(s, lo, hi, key, fee, decider=None, max_skips=None):
    o, c, d = s["open"], s["close"], s["dates"]
    n = hi - lo + 1
    ret, held = np.zeros(n), np.zeros(n, bool)
    cnt = dict(nb=0, nbf=0, ns=0, nsf=0)
    trades, st = [], new_state()
    if key == "hold":
        buy = np.zeros(len(o), bool)
        sell = buy
    else:
        buy, sell = s["sig"][key]["buy"], s["sig"][key]["sell"]
    pend = 1 if key == "hold" else 0                # 1=다음 시가 매수, -1=다음 시가 매도
    for k, t in enumerate(range(lo, hi + 1)):
        if pend == 1:                                # 시가 매수
            st.update(pos=1, entry=t, entry_px=o[t], skips=0)
            ret[k] = (1 - fee) * c[t] / o[t] - 1
            held[k] = True
        elif pend == -1:                             # 시가 매도
            ret[k] = o[t] / c[t - 1] - 1
            r = (1 - fee) * o[t] / st["entry_px"] - 1
            trades.append(dict(code=s["code"], 매수일=d[st["entry"]], 매수가=st["entry_px"],
                               매도일=d[t], 매도가=o[t], 상태="청산", 수익률=r,
                               보유일=t - st["entry"]))
            st.update(pos=0, sig=-1, entry=-1, entry_px=np.nan, skips=0,
                      prev_ret=r if np.isfinite(r) else st["prev_ret"], last_exit=t - 1)
        elif st["pos"]:
            ret[k] = c[t] / c[t - 1] - 1
            held[k] = True
        pend = 0
        if t >= hi:                                  # 마지막 날은 새 주문 없음
            break
        if st["pos"] == 0 and buy[t]:
            cnt["nb"] += 1
            if decider is None or decider(s, t, st):
                cnt["nbf"] += 1
                st["sig"] = t
                pend = 1
        elif st["pos"] == 1 and sell[t]:
            cnt["ns"] += 1
            forced = max_skips is not None and st["skips"] >= max_skips
            if forced or decider is None or decider(s, t, st):
                cnt["nsf"] += 1
                pend = -1
            else:
                st["skips"] += 1
    if st["pos"]:                                    # 미청산: 마지막 종가 평가
        trades.append(dict(code=s["code"], 매수일=d[st["entry"]], 매수가=st["entry_px"],
                           매도일=d[hi], 매도가=c[hi], 상태="보유중",
                           수익률=(1 - fee) * c[hi] / st["entry_px"] - 1,
                           보유일=hi - st["entry"]))
    idx = d[lo:hi + 1]
    return pd.Series(np.nan_to_num(ret), index=idx), pd.Series(held, index=idx), trades, cnt


def _metrics(port, held, T, cnt):
    eq = (1 + port).cumprod()
    sd = port.std()
    m = {"누적수익률(%)": (eq.iloc[-1] - 1) * 100,
         "연환산(%)": (eq.iloc[-1] ** (TRADING_DAYS / max(len(eq), 1)) - 1) * 100,
         "샤프": float(port.mean() / sd * np.sqrt(TRADING_DAYS)) if sd > 0 else 0.0,
         "MDD(%)": (eq / eq.cummax() - 1).min() * 100,
         "노출(%)": float(held.mean()) * 100,
         "거래수": float(len(T))}
    if len(T):
        r = T["수익률"]
        loss = -r[r < 0].sum()
        m.update({"승률(%)": (r > 0).mean() * 100,
                  "Profit Factor": r[r > 0].sum() / loss if loss > 0 else np.nan,
                  "평균보유일": T["보유일"].mean()})
    else:
        m.update({"승률(%)": np.nan, "Profit Factor": np.nan, "평균보유일": np.nan})
    m["매수 따름(%)"] = cnt["nbf"] / cnt["nb"] * 100 if cnt["nb"] else np.nan
    m["매도 따름(%)"] = cnt["nsf"] / cnt["ns"] * 100 if cnt["ns"] else np.nan
    return m


def evaluate(stocks, start, end, key, fee=BUY_FEE, decider=None, max_skips=None):
    """종목별 1/N 동일 비중. 반환: (지표 dict 또는 None, 매매기록 DataFrame)"""
    rets, helds, trades = {}, {}, []
    cnt = dict(nb=0, nbf=0, ns=0, nsf=0)
    for code, s in stocks.items():
        lo, hi = index_range(s["dates"], start, end)
        if hi - lo < 1:
            continue
        r, h, t, c = _simulate(s, lo, hi, key, fee, decider, max_skips)
        rets[code], helds[code] = r, h
        trades += t
        for k in cnt:
            cnt[k] += c[k]
    if not rets:
        return None, pd.DataFrame()
    port = pd.DataFrame(rets).mean(axis=1, skipna=True).fillna(0)
    held = pd.DataFrame(helds).astype(float).mean(axis=1, skipna=True)
    T = pd.DataFrame(trades)
    return _metrics(port, held, T, cnt), T
