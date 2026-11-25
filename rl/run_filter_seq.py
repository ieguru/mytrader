"""매매 전략 비교 + 주간 신호 필터(LightGBM · PPO)

공통 원칙과 전략 1~3, 주간 신호 정의는 rl/strategies.py
추가 피처(수급 누적·신호 강도·밸류·공매도·외국인 보유)는 rl/features_extra.py
 4 지도학습(LightGBM)   : 주간 매수 신호를 따를지 결정, 매도는 주간 매도 신호를 항상 따름 (쌍 유지)
 5 강화학습 필터(PPO)  : 현금이면 주간 매수 신호, 보유 중이면 주간 매도 신호를 따를지 결정 (쌍 유지)

실행:
  python -m rl.run_filter_seq --signals-only     # 1~3 + 주간 신호 백테스트 (학습 없음)
  python -m rl.run_filter_seq --check            # 미래 정보 점검
  python -m rl.run_filter_seq --diag             # 피처별 예측력 진단 (학습 없음)
  python -m rl.run_filter_seq --debug            # 5종목 · 2024년 · 짧은 학습
  python -m rl.run_filter_seq                    # 전진 분석 (추가 피처 포함)
  python -m rl.run_filter_seq --no-extra --tag _base   # 추가 피처 없이 (비교용)
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import time
import warnings
from pathlib import Path

import gymnasium as gym
import numpy as np
import pandas as pd

from rl.config import OUT_DIR
from rl.features_extra import NEW_PREFIX, add_extra_features, load_extra, stock_features
from rl.strategies import (BASE_KEYS, BUY_FEE, NAMES, ORDER, evaluate, index_range,
                           load_processed, ma_disp_signals, ma_signals, new_state,
                           prepare_stock, select_files, weekly_events)

log = logging.getLogger("rl.filter_seq")

PIT_SIGNALS = ["유목민신호", "유목민매도신호", "쌍바닥신호", "쌍고점신호"]
STATE_COLS = ["pos", "hold_days", "unreal", "skips", "prev_ret", "gap_days"]
REWARDS = ["log", "excess", "risk", "excess_risk"]


# ============================================================ 기본 피처 (과거 데이터만 사용)
def _rsi(close: pd.Series, n: int = 30) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    c, v = df["종가"], df["거래량"].replace(0, np.nan)
    f = pd.DataFrame(index=df.index)
    lr = np.log(c).diff()
    f["ret1"] = lr
    for n in (5, 20, 60):
        f[f"ret{n}"] = np.log(c / c.shift(n))
    f["vol20"] = lr.rolling(20).std() * np.sqrt(252)
    f["vol_chg"] = np.log(v / v.shift(1)).clip(-3, 3)
    f["vol_ratio20"] = np.log(v / v.rolling(20).mean()).clip(-3, 3)
    if "RSI30" in df.columns:
        f["rsi30"] = (pd.to_numeric(df["RSI30"], errors="coerce") - 50) / 50
    else:
        f["rsi30"] = (_rsi(c, 30) - 50) / 50
    for n in (5, 20, 60):
        f[f"disp{n}"] = c / c.rolling(n).mean() - 1
    macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    f["macd_osc"] = (macd - macd.ewm(span=9, adjust=False).mean()) / c * 100
    f["macd_osc_chg"] = f["macd_osc"].diff()
    for col in ("Weekly_Oscillator", "Monthly_Oscillator"):
        if col in df.columns:
            f[col] = pd.to_numeric(df[col], errors="coerce") / c * 100
    f["hi52_gap"] = c / c.rolling(250, min_periods=60).max() - 1
    for col in [x for x in df.columns if "순매수" in str(x)]:
        nb = pd.to_numeric(df[col], errors="coerce")
        f[f"{col}_r1"] = (nb / v).clip(-1, 1)
        f[f"{col}_r5"] = (nb.rolling(5).sum() / v.rolling(5).sum()).clip(-1, 1)
    for col in ("Weekly_Trend", "Monthly_Trend"):
        if col in df.columns:
            f[col] = pd.to_numeric(df[col], errors="coerce")
    for col in PIT_SIGNALS:
        if col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce").fillna(0)
            f[f"{col}_5d"] = s.rolling(5, min_periods=1).max()
    return f.replace([np.inf, -np.inf], np.nan)


def market_features(stocks: dict) -> pd.DataFrame:
    ret1 = pd.DataFrame({c: s["feat"]["ret1"] for c, s in stocks.items()})
    pos = pd.DataFrame({c: pd.Series(s["state_w"], index=s["dates"]) for c, s in stocks.items()})
    m = ret1.mean(axis=1)
    mi = m.fillna(0).cumsum()
    out = pd.DataFrame(index=ret1.index)
    out["mkt_ret5"] = m.rolling(5).sum()
    out["mkt_ret20"] = m.rolling(20).sum()
    out["mkt_vol20"] = m.rolling(20).std() * np.sqrt(252)
    out["mkt_vs_ma120"] = mi - mi.rolling(120).mean()
    out["breadth"] = pos.mean(axis=1)                    # 주간 신호 보유 상태인 종목 비율
    out["breadth_chg5"] = out["breadth"].diff(5)
    return out


class Scaler:
    def fit(self, X):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            self.mu = np.nan_to_num(np.nanmean(X, axis=0))
            sd = np.nanstd(X, axis=0)
        sd[~(sd > 1e-12)] = 1.0
        self.sd = sd
        return self

    def transform(self, X):
        return np.nan_to_num(np.clip((X - self.mu) / self.sd, -5, 5)).astype(np.float32)


def make_obs(s, t, st) -> np.ndarray:
    """t일 장 마감 시점의 관측값: 피처 + 포지션 상태(평가손익은 t일 종가 기준)."""
    if st["pos"]:
        e, px = st["entry"], st["entry_px"]
        hold = min(max(t - e, 0) / 60, 5.0)
        unreal = 0.0
        if t >= e and np.isfinite(px) and px > 0 and np.isfinite(s["close"][t]):
            unreal = float(np.clip(np.log(s["close"][t] / px) * 10, -5, 5))
        extra = [1.0, hold, unreal, min(st["skips"], 5) / 3]
    else:
        extra = [0.0, 0.0, 0.0, 0.0]
    extra += [float(np.clip(st["prev_ret"] * 10, -5, 5)), min(t - st["last_exit"], 250) / 250]
    return np.concatenate([s["X"][t], np.asarray(extra, np.float32)]).astype(np.float32)


# ============================================================ 강화학습 환경 (주간 신호, 쌍 유지)
class SeqFilterEnv(gym.Env):
    """한 스텝 = 판단 시점 하나 (현금이면 주간 매수 신호, 보유 중이면 주간 매도 신호)."""

    def __init__(self, items, fee, ep_len=32, max_skips=None, scale=10.0, reward="log"):
        super().__init__()
        self.items, self.fee, self.ep_len = items, fee, ep_len
        self.max_skips, self.scale, self.reward = max_skips, scale, reward
        nb = np.array([len(self._buys(it)) for it in items], float)
        self.w = nb / nb.sum()
        dim = items[0][0]["X"].shape[1] + len(STATE_COLS)
        self.observation_space = gym.spaces.Box(-10.0, 10.0, (dim,), np.float32)
        self.action_space = gym.spaces.Discrete(2)

    @staticmethod
    def _buys(it):
        s, lo, hi = it
        b = s["sig"]["weekly"]["buy_idx"]
        return b[(b >= lo) & (b < hi)]

    def _next(self, arr, t):
        k = np.searchsorted(arr, t, side="right")
        return int(arr[k]) if k < len(arr) and arr[k] < self.hi else -1

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        it = self.items[self.np_random.choice(len(self.items), p=self.w)]
        self.s, self.lo, self.hi = it
        b = self._buys(it)
        self.t = int(b[self.np_random.integers(len(b))])
        self.st, self.k = new_state(), 0
        return make_obs(self.s, self.t, self.st), {}

    def _leg(self, t, nxt):
        s, o = self.s, self.s["open"]
        e = nxt if nxt >= 0 else self.hi
        end_px = o[nxt + 1] if nxt >= 0 else s["close"][self.hi]
        r = np.log(end_px / o[t + 1])
        if self.reward in ("excess", "excess_risk"):
            r -= s["mcum"][e] - s["mcum"][t]
        if self.reward in ("risk", "excess_risk"):
            v = s["vol"][t]
            v = v if np.isfinite(v) else 0.3
            r *= 0.3 / float(np.clip(v, 0.1, 1.0))
        return r

    def step(self, action):
        s, t, st, o = self.s, self.t, self.st, self.s["open"]
        w = s["sig"]["weekly"]
        if st["pos"] == 0:
            if int(action) == 1:
                st.update(pos=1, sig=t, entry=t + 1, entry_px=o[t + 1], skips=0)
                nxt = self._next(w["sell_idx"], t)
                reward = np.log(1 - self.fee) + self._leg(t, nxt)
            else:
                nxt = self._next(w["buy_idx"], t)
                reward = 0.0
        else:
            forced = self.max_skips is not None and st["skips"] >= self.max_skips
            if forced or int(action) == 1:
                r = (1 - self.fee) * o[t + 1] / st["entry_px"] - 1
                st.update(pos=0, sig=-1, entry=-1, entry_px=np.nan, skips=0,
                          prev_ret=r if np.isfinite(r) else st["prev_ret"], last_exit=t)
                nxt = self._next(w["buy_idx"], t)
                reward = 0.0
            else:
                st["skips"] += 1
                nxt = self._next(w["sell_idx"], t)
                reward = self._leg(t, nxt)
        reward = float(np.nan_to_num(reward)) * self.scale
        self.k += 1
        terminated = nxt < 0
        truncated = (not terminated) and self.k >= self.ep_len
        if not terminated:
            self.t = nxt
        return make_obs(s, self.t, st), reward, terminated, truncated, {}


def ppo_decider(model):
    def decide(s, t, st):
        a, _ = model.predict(make_obs(s, t, st), deterministic=True)
        return bool(int(a))
    return decide


def train_ppo(items, stocks, va_range, seed, args, save_path, fold_name):
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    env = DummyVecEnv([lambda: SeqFilterEnv(items, args.fee, args.ep_len, args.max_skips,
                                            reward=args.reward)])
    model = PPO("MlpPolicy", env, gamma=args.gamma, gae_lambda=0.95, n_steps=1024, batch_size=256,
                n_epochs=10, learning_rate=args.lr, ent_coef=args.ent_coef,
                policy_kwargs=dict(net_arch=[64, 64]), seed=seed, verbose=0, device="cpu")
    best, best_state, bad, hist = -np.inf, None, 0, []
    while model.num_timesteps < args.timesteps:
        model.learn(args.eval_every, reset_num_timesteps=False, progress_bar=False)
        m, _ = evaluate(stocks, *va_range, "weekly", args.fee, ppo_decider(model), args.max_skips)
        sh = m["샤프"] if m else -np.inf
        star = ""
        if sh > best + 1e-9:
            best, best_state, bad, star = sh, copy.deepcopy(model.policy.state_dict()), 0, " ★ 최고 갱신"
        else:
            bad += 1
        bf = m.get("매수 따름(%)", np.nan) if m else np.nan
        sf = m.get("매도 따름(%)", np.nan) if m else np.nan
        hist.append({"step": model.num_timesteps, "val_sharpe": sh, "buy_follow": bf, "sell_follow": sf})
        log.info(f"[검증] {fold_name} seed{seed} step={model.num_timesteps} 샤프={sh:.2f} "
                 f"매수따름={bf:.0f}% 매도따름={sf:.0f}%{star}")
        if bad >= args.patience:
            log.info(f"  {args.patience}회 연속 개선 없음 → 조기 종료")
            break
    if best_state is not None:
        model.policy.load_state_dict(best_state)
    model.save(str(save_path))
    return model, pd.DataFrame(hist)


# ============================================================ 4. 지도학습(LightGBM)
def lgbm_buy_filter(stocks, tr_range, va_range, fee):
    try:
        from lightgbm import LGBMClassifier
    except ImportError:
        log.warning("lightgbm이 없어 건너뜁니다")
        return None, None
    X, y = [], []
    for s in stocks.values():
        w, o = s["sig"]["weekly"], s["open"]
        lo, hi = index_range(s["dates"], *tr_range)
        for t in w["buy_idx"][(w["buy_idx"] >= lo) & (w["buy_idx"] < hi)]:
            k = np.searchsorted(w["sell_idx"], t, side="right")
            if k >= len(w["sell_idx"]) or w["sell_idx"][k] + 1 > hi:
                continue
            ns = w["sell_idx"][k]
            r = (1 - fee) * o[ns + 1] / o[t + 1] - 1
            if np.isfinite(r):
                X.append(s["X"][t])
                y.append(r)
    if len(y) < 50:
        return None, None
    clf = LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=30,
                         subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                         random_state=0, verbose=-1)
    clf.fit(np.vstack(X), (np.asarray(y) > 0).astype(int))
    for s in stocks.values():
        b = s["sig"]["weekly"]["buy_idx"]
        s["lgb"] = dict(zip(b.tolist(), clf.predict_proba(s["X"][b])[:, 1])) if len(b) else {}

    def make(th):
        return lambda s, t, st: st["pos"] == 1 or s["lgb"].get(int(t), 0.0) >= th

    best_th, best_sh = 0.0, -np.inf
    for th in [0.0] + list(np.arange(0.40, 0.66, 0.05)):
        m, _ = evaluate(stocks, *va_range, "weekly", fee, make(th))
        if m and m["샤프"] > best_sh + 1e-9:
            best_th, best_sh = float(th), m["샤프"]
    return make(best_th), best_th


def random_decider(pb, ps, seed):
    rng = np.random.default_rng(seed)
    return lambda s, t, st: rng.random() < (ps if st["pos"] else pb)


# ============================================================ 전진 분석 1구간
def _save(T: pd.DataFrame, path: Path):
    if len(T):
        T.to_csv(path, index=False, encoding="utf-8-sig")


def run_fold(Y, stocks, args, out_dir):
    name = f"F{Y}"
    tr_range = (args.train_start, f"{Y - 2}-12-31")
    va_range = (f"{Y - 1}-01-01", f"{Y - 1}-12-31")
    te_range = (f"{Y}-01-01", f"{Y}-12-31")

    rows_fit = []
    for s in stocks.values():
        lo, hi = index_range(s["dates"], *tr_range)
        ev = s["sig"]["weekly"]["ev_idx"]
        ev = ev[(ev >= lo) & (ev < hi)]
        if len(ev):
            rows_fit.append(s["F"].to_numpy(float)[ev])
    if not rows_fit:
        log.warning(f"{name}: 학습 구간 신호가 없어 건너뜁니다")
        return []
    sc = Scaler().fit(np.vstack(rows_fit))
    for s in stocks.values():
        s["X"] = sc.transform(s["F"].to_numpy(float))

    items = []
    for s in stocks.values():
        lo, hi = index_range(s["dates"], *tr_range)
        b = s["sig"]["weekly"]["buy_idx"]
        if ((b >= lo) & (b < hi)).sum() >= 2:
            items.append((s, lo, hi))
    n_buy = sum(len(SeqFilterEnv._buys(it)) for it in items)
    log.info(f"=== {name}: 학습 {tr_range[0]}~{tr_range[1]} (주간 매수 신호 {n_buy}건, {len(items)}종목) / "
             f"검증 {Y - 1} / 테스트 {Y} / 보상 {args.reward}")
    if n_buy < 50:
        log.warning(f"{name}: 학습 신호가 부족해 건너뜁니다")
        return []

    fold_dir = out_dir / name
    fold_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    def add(key, m, seed=0):
        if m:
            rows.append({"fold": name, "전략": NAMES[key], "seed": seed, **m})

    for key in BASE_KEYS:
        m, T = evaluate(stocks, *te_range, key, args.fee)
        add(key, m)
        _save(T, fold_dir / f"trades_{key}.csv")

    pbs, pss = [], []
    for seed in range(args.seeds):
        t0 = time.time()
        model, hist = train_ppo(items, stocks, va_range, seed, args,
                                fold_dir / f"ppo_seq_seed{seed}.zip", name)
        hist.to_csv(fold_dir / f"val_curve_seed{seed}.csv", index=False, encoding="utf-8-sig")
        m, T = evaluate(stocks, *te_range, "weekly", args.fee, ppo_decider(model), args.max_skips)
        add("ppo", m, seed)
        _save(T, fold_dir / f"trades_ppo_seed{seed}.csv")
        if m:
            pbs.append(m.get("매수 따름(%)", 50.0) / 100)
            pss.append(m.get("매도 따름(%)", 50.0) / 100)
            log.info(f"  PPO seed{seed} 완료 ({time.time() - t0:.0f}초) 테스트 샤프 {m['샤프']:.2f}, "
                     f"매수따름 {pbs[-1]:.0%}, 매도따름 {pss[-1]:.0%}")

    decide, th = lgbm_buy_filter(stocks, tr_range, va_range, args.fee)
    if decide is not None:
        m, T = evaluate(stocks, *te_range, "weekly", args.fee, decide)
        add("lgbm", m)
        _save(T, fold_dir / "trades_lgbm.csv")
        log.info(f"  LightGBM 임계값 {th:.2f} (0이면 전부 따름)")

    pb = float(np.nanmean(pbs)) if pbs else 0.5
    ps = float(np.nanmean(pss)) if pss else 0.5
    draws = [evaluate(stocks, *te_range, "weekly", args.fee, random_decider(pb, ps, k), args.max_skips)[0]
             for k in range(args.random_draws)]
    draws = [d for d in draws if d]
    if draws:
        add("random", pd.DataFrame(draws).mean().to_dict())
    return rows


# ============================================================ 학습 없이 전략 1~3 + 주간 신호
def run_signals_only(stocks, args):
    out_dir = Path(OUT_DIR) / f"signals{args.tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    start, end = pd.Timestamp(args.start), pd.Timestamp(args.end)
    rows = []
    for key in BASE_KEYS:
        m, T = evaluate(stocks, start, end, key, args.fee)
        if m:
            rows.append({"전략": NAMES[key], **m})
        _save(T, out_dir / f"trades_{key}.csv")
    summary = pd.DataFrame(rows).set_index("전략")
    yearly = {}
    for Y in range(start.year, end.year + 1):
        ys, ye = max(start, pd.Timestamp(f"{Y}-01-01")), min(end, pd.Timestamp(f"{Y}-12-31"))
        for key in BASE_KEYS:
            m, _ = evaluate(stocks, ys, ye, key, args.fee)
            if m:
                yearly.setdefault(NAMES[key], {})[Y] = m["샤프"]
    by_year = pd.DataFrame(yearly).T
    summary.to_csv(out_dir / "summary.csv", encoding="utf-8-sig")
    by_year.to_csv(out_dir / "sharpe_by_year.csv", encoding="utf-8-sig")
    with pd.option_context("display.width", 260, "display.max_columns", 20,
                           "display.float_format", "{:.2f}".format):
        print(f"\n=== 전략 비교 {start.date()} ~ {end.date()} ===")
        print(summary)
        print("\n=== 연도별 샤프 ===")
        print(by_year)
    log.info(f"결과 저장: {out_dir}")
    return 0


# ============================================================ 피처별 예측력 진단
def run_diag(stocks, feat_cols, args):
    """주간 매수 신호일의 각 피처와 그 쌍(매수→다음 매도)의 수익률의 연도별 순위상관."""
    recs = []
    for s in stocks.values():
        w, o, F = s["sig"]["weekly"], s["open"], s["F"].to_numpy(float)
        for t in w["buy_idx"]:
            k = np.searchsorted(w["sell_idx"], t, side="right")
            if k >= len(w["sell_idx"]) or w["sell_idx"][k] + 1 >= len(o):
                continue
            r = (1 - args.fee) * o[w["sell_idx"][k] + 1] / o[t + 1] - 1
            y = s["dates"][t].year
            if np.isfinite(r) and y >= args.diag_start:
                recs.append(np.concatenate([[y, r], F[t]]))
    if not recs:
        log.error("진단할 매수 신호가 없습니다")
        return 1
    ev = pd.DataFrame(recs, columns=["year", "ret"] + feat_cols)
    rows = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for col in feat_cols:
            g = ev[["year", "ret", col]].dropna()
            ics = [x[col].corr(x["ret"], method="spearman") for _, x in g.groupby("year")
                   if len(x) >= 30 and x[col].nunique() > 5]
            ics = np.array([i for i in ics if np.isfinite(i)])
            if len(ics) < 3:
                continue
            q = pd.qcut(g[col].rank(method="first"), 5, labels=False)
            spread = (g["ret"][q == 4].mean() - g["ret"][q == 0].mean()) * 100
            sign = np.sign(ics.mean())
            rows.append({"피처": col, "신규": col.startswith(NEW_PREFIX), "평균IC": ics.mean(),
                         "t값": ics.mean() / (ics.std(ddof=1) + 1e-12) * np.sqrt(len(ics)),
                         "같은방향연도": f"{int((np.sign(ics) == sign).sum())}/{len(ics)}",
                         "상위-하위(%p)": spread, "표본": len(g)})
    res = pd.DataFrame(rows).sort_values("t값", key=np.abs, ascending=False).reset_index(drop=True)
    path = Path(OUT_DIR) / f"diag_features{args.tag}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(path, index=False, encoding="utf-8-sig")
    with pd.option_context("display.width", 200, "display.max_rows", 120,
                           "display.float_format", "{:.3f}".format):
        print(f"\n=== 피처별 예측력: 주간 매수 신호 {len(ev)}건, {args.diag_start}년~ ===")
        print(res)
    print("\n판단 기준: |평균IC| ≥ 0.05, |t값| ≥ 2, 같은방향연도가 대부분인 피처만 의미 있음.")
    log.info(f"결과 저장: {path}")
    return 0


# ============================================================ 미래 정보 점검
def _engine_check(code, points) -> str:
    try:
        from analysis_engine import analyze_frame
        from rl import storage
    except Exception as e:
        return f"건너뜀 (import 실패: {e})"
    try:
        try:
            raw = storage.read_raw(code)
        except Exception:
            raw = storage.read_raw(storage.raw_path(code))
    except Exception as e:
        return f"건너뜀 (원본 CSV 읽기 실패: {e})"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fdf = analyze_frame(raw, code, excel_file=None)["data"]
        fb, fs, _ = weekly_events(fdf)
        fb, fs = pd.Series(fb, index=fdf.index), pd.Series(fs, index=fdf.index)
        diff = 0
        for cut in points:
            if cut >= len(raw):
                continue
            sub = analyze_frame(raw.iloc[: cut + 1], code, excel_file=None)["data"]
            b, s, _ = weekly_events(sub)
            d = raw.index[cut]
            if d in fb.index:
                diff += int(bool(b[-1]) != bool(fb.loc[d])) + int(bool(s[-1]) != bool(fs.loc[d]))
    return "통과" if diff == 0 else f"{diff}곳에서 값이 달라짐 (의심)"


def _cut_check(fn, df, points):
    full, bad = fn(df), set()
    for cut in points:
        f = fn(df.iloc[: cut + 1])
        for col in full.columns:
            if not np.isclose(full[col].iloc[cut], f[col].iloc[-1], rtol=1e-6, atol=1e-9, equal_nan=True):
                bad.add(col)
    return bad


def check_lookahead(files, cuts=8):
    path = next((p for p in files if p.stem == "000660"), files[0])
    df = load_processed(path)
    points = np.linspace(300, len(df) - 2, cuts).astype(int)
    ok = True
    print(f"점검 종목: {path.stem}, 자른 지점 {len(points)}곳, 주간 신호 출처: {weekly_events(df)[2]}")

    bad = _cut_check(make_features, df, points)
    print("기본 피처: " + ("통과" if not bad else f"미래 정보 의심 {sorted(bad)}"))
    ok &= not bad

    ex = load_extra(path.stem, df.index)
    bad = _cut_check(lambda d: stock_features(d, None if ex is None else ex.loc[d.index]), df, points)
    print(f"추가 피처 (추가 데이터 {'있음' if ex is not None else '없음'}): "
          + ("통과" if not bad else f"미래 정보 의심 {sorted(bad)}"))
    ok &= not bad

    for key, fn in (("ma", ma_signals), ("ma_disp", ma_disp_signals)):
        fb, fs = fn(df)
        diff = 0
        for cut in points:
            b, s = fn(df.iloc[: cut + 1])
            diff += int(b[-1] != fb[cut]) + int(s[-1] != fs[cut])
        print(f"{NAMES[key]} 신호: " + ("통과" if diff == 0 else f"{diff}곳에서 값이 달라짐 (의심)"))
        ok &= diff == 0

    print("주간 매수·매도 신호 (엔진 재계산, 수 분 소요)...", flush=True)
    res = _engine_check(path.stem, points)
    print("주간 매수·매도 신호: " + res)
    ok &= not res.endswith("(의심)")
    return 0 if ok else 1


# ============================================================ 실행
def parse_args(argv=None):
    p = argparse.ArgumentParser(description="매매 전략 비교 + 주간 신호 필터(LightGBM·PPO)")
    p.add_argument("--signals-only", action="store_true")
    p.add_argument("--start", default="2016-01-01")
    p.add_argument("--end", default=pd.Timestamp.today().strftime("%Y-%m-%d"))
    p.add_argument("--check", action="store_true")
    p.add_argument("--diag", action="store_true", help="피처별 예측력 진단 (학습 없음)")
    p.add_argument("--diag-start", type=int, default=2017)
    p.add_argument("--no-extra", action="store_true", help="추가 피처(features_extra) 끄기")
    p.add_argument("--debug", action="store_true")
    p.add_argument("--universe", default=None, choices=["stocks", "kospi200"])
    p.add_argument("--tickers", nargs="*", default=None)
    p.add_argument("--max-stocks", type=int, default=None)
    p.add_argument("--fee", type=float, default=BUY_FEE)
    p.add_argument("--test-years", nargs="*", type=int, default=[2024, 2025, 2026])
    p.add_argument("--train-start", default="2016-01-01")
    p.add_argument("--timesteps", type=int, default=100_000)
    p.add_argument("--eval-every", type=int, default=5_000)
    p.add_argument("--patience", type=int, default=6)
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--gamma", type=float, default=0.9)
    p.add_argument("--ent-coef", type=float, default=0.001)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--ep-len", type=int, default=32)
    p.add_argument("--max-skips", type=int, default=None)
    p.add_argument("--reward", default="log", choices=REWARDS)
    p.add_argument("--random-draws", type=int, default=10)
    p.add_argument("--tag", default="")
    args = p.parse_args(argv)
    if args.debug:
        args.max_stocks = args.max_stocks or 5
        args.test_years, args.timesteps, args.eval_every = [2024], 4_000, 2_000
        args.seeds, args.random_draws = 1, 2
    return args


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args(argv)
    files = select_files(args.universe, args.tickers, args.max_stocks)
    if not files:
        log.error("processed 폴더에 사용할 CSV가 없습니다. STEP 1을 먼저 실행하세요.")
        return 1
    if args.check:
        return check_lookahead(files)

    t0 = time.time()
    stocks = {}
    for path in files:
        try:
            stocks[path.stem] = prepare_stock(load_processed(path), path.stem)
        except Exception as e:
            log.warning(f"{path.stem} 건너뜀: {e}")
    if not stocks:
        log.error("읽을 수 있는 종목이 없습니다")
        return 1
    sources = sorted({s["src"] for s in stocks.values()})
    log.info(f"{len(stocks)}종목 / 주간 신호 출처 {sources} / 매수 수수료 {args.fee:.2%}")

    if args.signals_only:
        return run_signals_only(stocks, args)

    for s in stocks.values():
        s["feat"] = make_features(s["df"])
    if not args.no_extra:
        n_ex = add_extra_features(stocks)
        log.info(f"추가 피처 적용: 추가 데이터(extra) 있는 종목 {n_ex} / {len(stocks)}")
        if n_ex == 0:
            log.warning("data/rl/extra 가 비어 있습니다. 밸류·공매도·외국인보유 피처는 빠집니다 (STEP 19 실행).")
    mkt = market_features(stocks)
    mret = pd.DataFrame({c: s["feat"]["ret1"] for c, s in stocks.items()}).mean(axis=1)
    mcum = mret.fillna(0).cumsum()
    feat_cols = []
    for s in stocks.values():
        s["mcum"] = mcum.reindex(s["dates"]).ffill().fillna(0).to_numpy()
        s["vol"] = s["feat"]["vol20"].to_numpy(float)
        s["F"] = s["feat"].join(mkt, how="left")
        feat_cols += [c for c in s["F"].columns if c not in feat_cols]
    for s in stocks.values():
        s["F"] = s["F"].reindex(columns=feat_cols)
    n_new = sum(c.startswith(NEW_PREFIX) for c in feat_cols)

    if args.diag:
        return run_diag(stocks, feat_cols, args)

    n_buy = sum(len(s["sig"]["weekly"]["buy_idx"]) for s in stocks.values())
    log.info(f"실험 시작: 주간 매수 신호 {n_buy}건 / 피처 {len(feat_cols)}개(신규 {n_new})+상태 {len(STATE_COLS)} / "
             f"테스트 {args.test_years} / 시드 {args.seeds} / 보상 {args.reward}")

    out_dir = Path(OUT_DIR) / f"filter_seq{'_debug' if args.debug else ''}{args.tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(vars(args), ensure_ascii=False, indent=2), encoding="utf-8")
    pd.Series(feat_cols, name="feature").to_csv(out_dir / "features.csv", index=False, encoding="utf-8-sig")

    rows = []
    for Y in args.test_years:
        rows += run_fold(Y, stocks, args, out_dir)
    if not rows:
        log.error("결과가 없습니다")
        return 1

    detail = pd.DataFrame(rows)
    present = [o for o in ORDER if o in set(detail["전략"])]
    summary = detail.drop(columns="seed").groupby("전략").mean(numeric_only=True).reindex(present)
    by_fold = detail.pivot_table(index="전략", columns="fold", values="샤프", aggfunc="mean").reindex(present)
    detail.to_csv(out_dir / "detail_all.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(out_dir / "summary.csv", encoding="utf-8-sig")
    by_fold.to_csv(out_dir / "sharpe_by_fold.csv", encoding="utf-8-sig")
    with pd.option_context("display.width", 260, "display.max_columns", 20,
                           "display.float_format", "{:.2f}".format):
        print(f"\n=== 전체 평균 (구간·시드) / 보상 {args.reward} / 추가 피처 {'끔' if args.no_extra else '켬'} ===")
        print(summary)
        print("\n=== 구간별 샤프 ===")
        print(by_fold)
    log.info(f"결과 저장: {out_dir} (총 {(time.time() - t0) / 60:.1f}분)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
