"""주간 신호 + 강화학습 필터

기본 매매는 analysis_engine의 '주간 신호'(Weekly_Buy / Weekly_Sell)가 정하고,
주간 매수 신호가 나온 날에만 강화학습(PPO)이 '따른다(1) / 건너뛴다(0)'를 고른다.
매도는 주간 매도 신호를 항상 따른다.

실행:
  python -m rl.run_filter --check     # 미래 정보 점검
  python -m rl.run_filter --debug     # 5종목 · 2024년 · 짧은 학습
  python -m rl.run_filter             # 본 실험 (테스트 2024~2026, 시드 5)
  python -m rl.run_filter --test-years 2020 2021 2022 2023 2024 2025 2026
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

from rl.config import OUT_DIR, PROCESSED_DIR, RLConfig

log = logging.getLogger("rl.filter")

META_COLS = ["code", "date", "i", "j", "exit_exec", "ret", "hold_days"]
PIT_SIGNALS = ["유목민신호", "유목민매도신호", "쌍바닥신호", "쌍고점신호"]
ORDER = ["단순보유", "주간 신호(전부 따름)", "강화학습 필터(PPO)",
         "지도학습 필터(LightGBM)", "무작위 필터(채택률 일치)"]


# ============================================================ 데이터
def load_processed(path: Path) -> pd.DataFrame:
    try:
        from rl.storage import read_csv
        df = read_csv(path)
    except Exception:
        df = pd.read_csv(path, encoding="utf-8-sig")
    if not isinstance(df.index, pd.DatetimeIndex):
        for c in ("날짜", "Date", "date", "일자", "Unnamed: 0"):
            if c in df.columns:
                df = df.set_index(c)
                break
        df.index = pd.to_datetime(df.index)
    if df.index.min().year < 1990:
        raise ValueError("날짜 열을 찾지 못했습니다")
    df = df[~df.index.duplicated(keep="last")].sort_index()
    for c in ("시가", "고가", "저가", "종가", "거래량"):
        if c not in df.columns:
            raise KeyError(f"'{c}' 열이 없습니다")
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _weekly_trend_from_close(df: pd.DataFrame) -> pd.Series:
    """엔진 열이 없을 때의 예비 계산: 주봉(W-FRI) MACD(5,13,5) 오실레이터,
    직전 완료 주봉 기준(shift 1)의 증감 방향 (+1/-1/0)."""
    w = df["종가"].resample("W-FRI").last().dropna()
    macd = w.ewm(span=5, adjust=False).mean() - w.ewm(span=13, adjust=False).mean()
    osc = (macd - macd.ewm(span=5, adjust=False).mean()).shift(1)
    tr = np.sign(osc.diff()).fillna(0)
    tr.index = tr.index.to_period("W-FRI")
    periods = df.index.to_period("W-FRI")
    return pd.Series(periods.map(tr), index=df.index).astype(float).fillna(0)


def weekly_position(df: pd.DataFrame) -> pd.Series:
    """주간 신호 보유 상태(1=보유, 0=현금). t일 종가 기준 결정, t+1 시가 체결.
    매수: 주간 추세 -1→+1 (Weekly_Buy), 매도: +1→-1 (Weekly_Sell)."""
    cols = set(df.columns)
    if {"Weekly_Buy", "Weekly_Sell"} <= cols:
        buy = pd.to_numeric(df["Weekly_Buy"], errors="coerce").fillna(0).to_numpy() > 0
        sell = pd.to_numeric(df["Weekly_Sell"], errors="coerce").fillna(0).to_numpy() > 0
        source = "Weekly_Buy/Weekly_Sell"
    else:
        if "Weekly_Trend" in cols:
            tr = pd.to_numeric(df["Weekly_Trend"], errors="coerce").fillna(0)
            source = "Weekly_Trend"
        else:
            tr = _weekly_trend_from_close(df)
            source = "종가로 직접 계산"
        prev = tr.shift(1)
        buy = ((prev == -1) & (tr == 1)).to_numpy()
        sell = ((prev == 1) & (tr == -1)).to_numpy()
    pos = np.zeros(len(df), dtype=int)
    state = 0
    for k in range(len(df)):
        if state == 0 and buy[k]:
            state = 1
        elif state == 1 and sell[k]:
            state = 0
        pos[k] = state
    out = pd.Series(pos, index=df.index)
    out.attrs["source"] = source
    return out


# ============================================================ 피처 (모두 과거 데이터만 사용)
def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
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
    pos = pd.DataFrame({c: s["pos"] for c, s in stocks.items()})
    m = ret1.mean(axis=1)
    out = pd.DataFrame(index=ret1.index)
    out["mkt_ret5"] = m.rolling(5).sum()
    out["mkt_ret20"] = m.rolling(20).sum()
    out["mkt_vol20"] = m.rolling(20).std() * np.sqrt(252)
    out["breadth"] = pos.mean(axis=1)          # 주간 신호가 켜진 종목 비율
    out["breadth_chg5"] = out["breadth"].diff(5)
    return out


# ============================================================ 매수 신호 이벤트
def extract_events(code: str, df: pd.DataFrame, pos: pd.Series, cost: float) -> pd.DataFrame:
    o = df["시가"].to_numpy(float)
    p = pos.to_numpy()
    n, dates = len(p), df.index
    rows, prev_ret, last_exit, starts = [], 0.0, -10**9, []
    for i in range(1, n):
        if p[i] == 1 and p[i - 1] == 0:
            j = i + 1
            while j < n and p[j] == 1:
                j += 1
            ei, xi = i + 1, j + 1                      # 매수 체결일, 매도 체결일
            ok = xi < n and o[ei] > 0 and o[xi] > 0
            ret = np.log(o[xi] / o[ei]) - 2 * cost if ok else np.nan
            starts = [k for k in starts if i - k <= 120]
            rows.append(dict(
                code=code, date=dates[i], i=i, j=j,
                exit_exec=dates[xi] if xi < n else pd.NaT,
                ret=ret, hold_days=j - i,                       # ← 결과(미래) 정보, 피처 아님
                prev_ret=prev_ret,                              # 직전 매매 결과 (이미 확정)
                gap_days=min(i - last_exit, 250) / 250,
                n_sig120=len(starts),
            ))
            starts.append(i)
            if not np.isnan(ret):
                prev_ret = ret
            last_exit = j
    return pd.DataFrame(rows)


def build_event_table(stocks: dict, mkt: pd.DataFrame, cost: float) -> pd.DataFrame:
    parts = []
    for code, s in stocks.items():
        ev = extract_events(code, s["df"], s["pos"], cost)
        if ev.empty:
            continue
        feat = s["feat"].join(mkt, how="left")
        X = feat.iloc[ev["i"].to_numpy()].reset_index(drop=True)
        parts.append(pd.concat([ev.reset_index(drop=True), X], axis=1))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


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


# ============================================================ 백테스트 (모든 전략 동일 회계)
class Backtester:
    """t일 종가 결정 → t+1 시가 체결 → t+2 시가까지 손익. 포지션 변경마다 편도 비용.
    종목별 1/N, 매일 리밸런싱. 시작일에는 현금에서 출발."""

    def __init__(self, stocks: dict, cost: float):
        self.opens = {c: s["df"]["시가"].astype(float).where(lambda x: x > 0)
                      for c, s in stocks.items()}
        self.cost = cost

    def positions(self, ev, accept):
        pos = {c: np.zeros(len(o)) for c, o in self.opens.items()}
        if len(ev):
            for code, i, j in zip(ev["code"].to_numpy()[accept],
                                  ev["i"].to_numpy()[accept], ev["j"].to_numpy()[accept]):
                pos[code][i:j] = 1.0
        return pos

    def run(self, ev, accept, start, end, hold_all=False):
        start, end = pd.Timestamp(start), pd.Timestamp(end)
        if hold_all or accept is None or not len(ev):
            accept = np.zeros(len(ev), bool) if accept is None else np.asarray(accept, bool)
        else:
            accept = np.asarray(accept, dtype=bool)
        pos = None if hold_all else self.positions(ev, accept)
        R, D, T = {}, {}, {}
        for c, o in self.opens.items():
            fwd = (o.shift(-2) / o.shift(-1) - 1).to_numpy()
            m = (o.index >= start) & (o.index <= end) & np.isfinite(fwd)
            if not m.any():
                continue
            d = np.ones(m.sum()) if hold_all else pos[c][m]
            turn = np.abs(np.diff(d, prepend=0.0))
            idx = o.index[m]
            R[c] = pd.Series(d * fwd[m] - self.cost * turn, idx)
            D[c] = pd.Series(d, idx)
            T[c] = pd.Series(turn, idx)
        if not R:
            return None
        R, D, T = pd.DataFrame(R), pd.DataFrame(D), pd.DataFrame(T)
        port = R.mean(axis=1).fillna(0.0)
        n = len(port)
        eq = (1 + port).cumprod()
        cum = eq.iloc[-1] - 1
        sd = port.std()
        if hold_all:
            n_trades, rets, rate = len(R.columns), np.array([]), 1.0
        else:
            n_trades = int(accept.sum())
            rets = ev["ret"].to_numpy()[accept] if len(ev) else np.array([])
            rets = rets[np.isfinite(rets)]
            rate = accept.mean() if len(accept) else 0.0
        return {
            "누적수익률(%)": cum * 100,
            "연환산(%)": ((1 + cum) ** (252 / n) - 1) * 100 if cum > -1 else np.nan,
            "샤프": port.mean() / sd * np.sqrt(252) if sd > 0 else 0.0,
            "MDD(%)": (eq / eq.cummax() - 1).min() * 100,
            "회전율(연)": float(np.nanmean(T.sum(axis=0))) * 252 / n,
            "노출(%)": float(np.nanmean(D.to_numpy())) * 100,
            "거래수": n_trades,
            "승률(%)": (rets > 0).mean() * 100 if len(rets) else np.nan,
            "채택률(%)": rate * 100,
        }


# ============================================================ 강화학습 환경 (bandit)
class FilterEnv(gym.Env):
    """한 스텝 = 매수 신호 한 건. 행동 1=따름, 0=건너뜀. 보상 = 행동 × 매매 순수익률."""

    def __init__(self, X, y, ep_len=64, loss_penalty=0.0, scale=10.0):
        super().__init__()
        self.X, self.y = X, y
        self.ep_len, self.lp, self.scale = ep_len, loss_penalty, scale
        self.observation_space = gym.spaces.Box(-5.0, 5.0, (X.shape[1],), np.float32)
        self.action_space = gym.spaces.Discrete(2)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.idx = self.np_random.integers(0, len(self.X), size=self.ep_len)
        self.k = 0
        return self.X[self.idx[0]], {}

    def step(self, action):
        y = float(self.y[self.idx[self.k]])
        if y < 0:
            y *= 1.0 + self.lp
        reward = float(action) * y * self.scale
        self.k += 1
        done = self.k >= self.ep_len
        obs = self.X[self.idx[min(self.k, self.ep_len - 1)]]
        return obs, reward, done, False, {}


def train_ppo(Xtr, ytr, va, Xva, bt, va_range, seed, args, save_path, fold_name):
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    env = DummyVecEnv([lambda: FilterEnv(Xtr, ytr, args.ep_len, args.loss_penalty)])
    model = PPO("MlpPolicy", env, gamma=0.0, gae_lambda=1.0, n_steps=512, batch_size=128,
                n_epochs=10, learning_rate=args.lr, ent_coef=args.ent_coef,
                policy_kwargs=dict(net_arch=[64, 64]), seed=seed, verbose=0, device="cpu")
    best, best_state, bad, steps, hist = -np.inf, None, 0, 0, []
    while steps < args.timesteps:
        model.learn(args.eval_every, reset_num_timesteps=False, progress_bar=False)
        steps = model.num_timesteps
        acc = model.predict(Xva, deterministic=True)[0].astype(bool)
        m = bt.run(va, acc, *va_range)
        sh = m["샤프"] if m else -np.inf
        star = ""
        if sh > best + 1e-9:
            best, best_state, bad, star = sh, copy.deepcopy(model.policy.state_dict()), 0, " ★ 최고 갱신"
        else:
            bad += 1
        hist.append({"step": steps, "val_sharpe": sh, "accept_rate": acc.mean()})
        log.info(f"[검증] {fold_name} seed{seed} step={steps} 샤프={sh:.2f} "
                 f"채택률={acc.mean():.0%}{star}")
        if bad >= args.patience:
            log.info(f"  {args.patience}회 연속 개선 없음 → 조기 종료")
            break
    if best_state is not None:
        model.policy.load_state_dict(best_state)
    model.save(str(save_path))
    return model, pd.DataFrame(hist)


def lgbm_filter(Xtr, ytr, Xva, Xte, va, bt, va_range):
    try:
        from lightgbm import LGBMClassifier
    except ImportError:
        log.warning("lightgbm이 없어 지도학습 필터를 건너뜁니다")
        return None, None, None
    clf = LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15,
                         min_child_samples=30, subsample=0.8, subsample_freq=1,
                         colsample_bytree=0.8, random_state=0, verbose=-1)
    clf.fit(Xtr, (ytr > 0).astype(int))
    pva, pte = clf.predict_proba(Xva)[:, 1], clf.predict_proba(Xte)[:, 1]
    best_th = 0.0                                  # 0.0 = 전부 따름 (필터 없음도 후보)
    base = bt.run(va, np.ones(len(va), bool), *va_range)
    best_sh = base["샤프"] if base else -np.inf
    for th in np.arange(0.40, 0.66, 0.05):
        m = bt.run(va, pva >= th, *va_range)
        if m and m["샤프"] > best_sh:
            best_th, best_sh = float(th), m["샤프"]
    return pte >= best_th, best_th, pte


# ============================================================ 전진 분석 1구간
def run_fold(Y, ev, feat_cols, bt, args, out_dir):
    name = f"F{Y}"
    tr_end = pd.Timestamp(f"{Y - 2}-12-31")
    va_range = (f"{Y - 1}-01-01", f"{Y - 1}-12-31")
    te_range = (f"{Y}-01-01", f"{Y}-12-31")
    tr = ev[(ev["date"] >= pd.Timestamp(args.train_start)) & (ev["date"] <= tr_end)
            & (ev["exit_exec"] <= tr_end) & ev["ret"].notna()]        # 경계 넘는 매매는 제외(purge)
    va = ev[(ev["date"] >= va_range[0]) & (ev["date"] <= va_range[1])].reset_index(drop=True)
    te = ev[(ev["date"] >= te_range[0]) & (ev["date"] <= te_range[1])].reset_index(drop=True)
    if len(tr):
        log.info(f"=== {name}: 학습 {args.train_start}~{tr_end.date()} ({len(tr)}건, "
                 f"평균 {tr['ret'].mean() * 100:.2f}%, 승률 {(tr['ret'] > 0).mean():.0%}) / "
                 f"검증 {Y - 1} ({len(va)}건) / 테스트 {Y} ({len(te)}건)")
    if len(tr) < 50 or va.empty or te.empty:
        log.warning(f"{name}: 이벤트가 부족해 건너뜁니다 (학습 {len(tr)}, 검증 {len(va)}, 테스트 {len(te)})")
        return []

    sc = Scaler().fit(tr[feat_cols].to_numpy(float))
    Xtr, ytr = sc.transform(tr[feat_cols].to_numpy(float)), tr["ret"].to_numpy(float)
    Xva, Xte = sc.transform(va[feat_cols].to_numpy(float)), sc.transform(te[feat_cols].to_numpy(float))
    fold_dir = out_dir / name
    fold_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    def add(strat, m, seed=0):
        if m:
            rows.append({"fold": name, "전략": strat, "seed": seed, **m})

    add("단순보유", bt.run(te, None, *te_range, hold_all=True))
    add("주간 신호(전부 따름)", bt.run(te, np.ones(len(te), bool), *te_range))

    out_ev = te[META_COLS].copy()
    rates = []
    for seed in range(args.seeds):
        t0 = time.time()
        model, hist = train_ppo(Xtr, ytr, va, Xva, bt, va_range, seed, args,
                                fold_dir / f"ppo_filter_seed{seed}.zip", name)
        hist.to_csv(fold_dir / f"val_curve_seed{seed}.csv", index=False, encoding="utf-8-sig")
        acc = model.predict(Xte, deterministic=True)[0].astype(bool)
        out_ev[f"ppo_seed{seed}"] = acc.astype(int)
        rates.append(acc.mean())
        add("강화학습 필터(PPO)", bt.run(te, acc, *te_range), seed)
        log.info(f"  PPO seed{seed} 완료 ({time.time() - t0:.0f}초), 테스트 채택률 {acc.mean():.0%}")

    acc, th, prob = lgbm_filter(Xtr, ytr, Xva, Xte, va, bt, va_range)
    if acc is not None:
        out_ev["lgbm_prob"] = prob
        add("지도학습 필터(LightGBM)", bt.run(te, acc, *te_range))
        log.info(f"  LightGBM 임계값 {th:.2f} (0이면 전부 따름), 채택률 {acc.mean():.0%}")

    rate = float(np.mean(rates)) if rates else 0.5
    rng = np.random.default_rng(0)
    draws = [bt.run(te, rng.random(len(te)) < rate, *te_range) for _ in range(args.random_draws)]
    draws = [d for d in draws if d]
    if draws:
        add("무작위 필터(채택률 일치)", pd.DataFrame(draws).mean().to_dict())

    out_ev.to_csv(fold_dir / "events_test.csv", index=False, encoding="utf-8-sig")
    return rows


# ============================================================ 미래 정보 점검
def _engine_weekly_check(code: str, points) -> str:
    """원본을 잘라 analysis_engine을 다시 돌려 주간 신호 포지션이 같은지 확인."""
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
        full = weekly_position(analyze_frame(raw, code, excel_file=None)["data"])
        diff = 0
        for cut in points:
            if cut >= len(raw):
                continue
            sub = analyze_frame(raw.iloc[: cut + 1], code, excel_file=None)["data"]
            p = weekly_position(sub)
            d = raw.index[cut]
            if d in p.index and d in full.index:
                diff += int(p.loc[d] != full.loc[d])
    return "통과" if diff == 0 else f"{diff}곳에서 값이 달라짐 (의심)"


def check_lookahead(files, cuts=8):
    path = next((p for p in files if p.stem == "000660"), files[0])
    df = load_processed(path)
    full_f = make_features(df)
    bad = set()
    points = np.linspace(300, len(df) - 2, cuts).astype(int)
    for cut in points:
        f = make_features(df.iloc[: cut + 1])
        for col in full_f.columns:
            if not np.isclose(full_f[col].iloc[cut], f[col].iloc[-1],
                              rtol=1e-6, atol=1e-9, equal_nan=True):
                bad.add(col)
    src = weekly_position(df).attrs.get("source", "?")
    print(f"점검 종목: {path.stem}, 자른 지점 {len(points)}곳, 주간 신호 출처: {src}")
    print("피처: " + ("통과" if not bad else f"미래 정보 의심 {sorted(bad)}"))
    print("주간 신호 포지션 (엔진 재계산, 수 분 소요)...", flush=True)
    res = _engine_weekly_check(path.stem, points)
    print("주간 신호 포지션: " + res)
    return 0 if not bad and not res.endswith("(의심)") else 1


# ============================================================ 실행
def parse_args(argv=None):
    p = argparse.ArgumentParser(description="주간 신호 + 강화학습 필터")
    p.add_argument("--check", action="store_true", help="미래 정보 점검만 실행")
    p.add_argument("--debug", action="store_true", help="5종목·2024년·짧은 학습")
    p.add_argument("--tickers", nargs="*", default=None)
    p.add_argument("--max-stocks", type=int, default=None)
    p.add_argument("--test-years", nargs="*", type=int, default=[2024, 2025, 2026])
    p.add_argument("--train-start", default="2016-01-01")
    p.add_argument("--timesteps", type=int, default=60_000)
    p.add_argument("--eval-every", type=int, default=3_000)
    p.add_argument("--patience", type=int, default=6)
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--ent-coef", type=float, default=0.001)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--ep-len", type=int, default=64)
    p.add_argument("--loss-penalty", type=float, default=0.0, help="손실 매매 보상 가중(0.5면 손실 1.5배)")
    p.add_argument("--random-draws", type=int, default=20)
    p.add_argument("--tag", default="", help="결과 폴더 이름 뒤에 붙일 문자")
    args = p.parse_args(argv)
    if args.debug:
        args.max_stocks = args.max_stocks or 5
        args.test_years, args.timesteps, args.eval_every = [2024], 3_000, 1_000
        args.seeds, args.random_draws = 1, 2
    return args


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args(argv)
    files = sorted(p for p in Path(PROCESSED_DIR).glob("*.csv") if p.stem.isdigit())
    if args.tickers:
        files = [p for p in files if p.stem in set(args.tickers)]
    if args.max_stocks:
        files = files[: args.max_stocks]
    if not files:
        log.error(f"{PROCESSED_DIR}에 처리된 CSV가 없습니다. STEP 1(--prepare-only)을 먼저 실행하세요.")
        return 1
    if args.check:
        return check_lookahead(files)

    t0 = time.time()
    cost = RLConfig().round_trip_cost / 2
    stocks, sources = {}, set()
    for path in files:
        try:
            df = load_processed(path)
            pos = weekly_position(df)
            sources.add(pos.attrs.get("source", "?"))
            stocks[path.stem] = dict(df=df, pos=pos, feat=make_features(df))
        except Exception as e:
            log.warning(f"{path.stem} 건너뜀: {e}")
    if not stocks:
        log.error("읽을 수 있는 종목이 없습니다")
        return 1
    ev = build_event_table(stocks, market_features(stocks), cost)
    if ev.empty:
        log.error("주간 매수 신호가 한 건도 없습니다")
        return 1
    feat_cols = [c for c in ev.columns if c not in META_COLS]
    log.info(f"실험 시작: {len(stocks)}종목 / 주간 신호 출처 {sorted(sources)} / 매수 신호 {len(ev)}건 / "
             f"피처 {len(feat_cols)}개 / 테스트 연도 {args.test_years} / 시드 {args.seeds}")

    out_dir = Path(OUT_DIR) / f"filter_weekly{'_debug' if args.debug else ''}{args.tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(
        json.dumps(vars(args), ensure_ascii=False, indent=2), encoding="utf-8")
    pd.Series(feat_cols, name="feature").to_csv(out_dir / "features.csv", index=False, encoding="utf-8-sig")

    bt = Backtester(stocks, cost)
    rows = []
    for Y in args.test_years:
        rows += run_fold(Y, ev, feat_cols, bt, args, out_dir)
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
    with pd.option_context("display.width", 220, "display.max_columns", 20,
                           "display.float_format", "{:.2f}".format):
        print("\n=== 전체 평균 (구간·시드) ===")
        print(summary)
        print("\n=== 구간별 샤프 ===")
        print(by_fold)
    log.info(f"결과 저장: {out_dir} (총 {(time.time() - t0) / 60:.1f}분)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
