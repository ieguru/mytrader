# %% 설정
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import date
from stable_baselines3 import PPO

from charts import ChartPlotter
from rl.baselines import buy_hold
from rl.config import FOLDS, OUT_DIR, RLConfig
from rl.dataset import load_market, load_stock, load_universe
from rl.env import TradingEnv
from rl.evaluate import evaluate_positions, position_returns
from rl.features import build_features
from rl.run import prepare_fold
from rl.train import policy_positions

ChartPlotter.setup_korean_font()          # 그래프 한글 표시

RUN, FOLD, SEED = OUT_DIR / "stocks_dd", "F2024", 0
saved = json.loads((RUN / "config.json").read_text())
cfg = RLConfig(**{k: tuple(v) if isinstance(v, list) else v for k, v in saved.items()})

# %% 데이터·모델 불러오기 (캐시 사용)
universe = load_universe("stocks", cfg.data_start)
names = {t: n for n, t in universe}
data = {}
for name, ticker in universe:
    try:
        _, data[ticker] = load_stock(name, ticker, cfg.data_start, date.today().isoformat())
    except Exception as e:
        print("건너뜀", name, e)

market = load_market(cfg.data_start, date.today().isoformat(), data)
feats = {t: build_features(d, market) for t, d in data.items()}
fold = next(f for f in FOLDS if f.name == FOLD)
train, val, test = prepare_fold(fold, data, feats, cfg)
model = PPO.load(RUN / FOLD / f"ppo_seed{SEED}.zip", device="cpu")

# %% 검증 곡선
curve = pd.read_csv(RUN / FOLD / f"val_curve_seed{SEED}.csv")
curve.plot(x="timesteps", y="val_sharpe", marker="o", title="검증 샤프")

# %% 종목별 테스트 성과 (RL vs 단순보유)
rows = []
for seg in test:
    pos = policy_positions(model, seg, cfg)
    rl = evaluate_positions([seg], [pos], cfg)
    bh = evaluate_positions([seg], [buy_hold(seg)], cfg)
    rows.append({"종목": names.get(seg.ticker, seg.ticker),
                 "RL 연환산": rl["연환산(%)"], "보유 연환산": bh["연환산(%)"],
                 "RL 샤프": rl["샤프"], "노출(%)": rl["노출(%)"], "거래수": rl["거래수"]})
table = pd.DataFrame(rows).round(2).sort_values("RL 샤프", ascending=False)
table

# %% 한 종목 매매 시점 + 정책의 보유 확률
TICKER = "000660"
seg = next(s for s in test if s.ticker == TICKER)

env = TradingEnv([seg], cfg, train=False)
obs, _ = env.reset()
pos, prob = [], []
while True:
    tensor, _ = model.policy.obs_to_tensor(obs)
    p = model.policy.get_distribution(tensor).distribution.probs[0].detach().numpy()
    prob.append(p[-1])                                   # 가장 높은 포지션(보유)의 확률
    obs, _, term, trunc, info = env.step(int(p.argmax()))
    pos.append(info["pos"])
    if term or trunc:
        break
pos, prob = np.array(pos), np.array(prob)

d = seg.first + np.arange(len(pos))
x, close = seg.dates[d], seg.close[d]
prev = np.concatenate([[0.0], pos[:-1]])
buy, sell = pos > prev, pos < prev

fig, (a1, a2, a3) = plt.subplots(3, 1, figsize=(13, 9), sharex=True,
                                 gridspec_kw={"height_ratios": [3, 1, 2]})
a1.plot(x, close, color="black", lw=1)
a1.scatter(x[buy], close[buy], marker="^", color="red", s=70, label="매수 결정")
a1.scatter(x[sell], close[sell], marker="v", color="blue", s=70, label="매도 결정")
a1.set_title(f"{names.get(TICKER, TICKER)} — 결정일 종가 기준 (체결은 다음날 시가)")
a1.legend()
a2.plot(x, prob, color="tab:purple")
a2.axhline(0.5, ls=":", color="gray")
a2.set_ylabel("보유 확률")
r = position_returns(pos, seg, cfg.round_trip_cost)
bh = position_returns(buy_hold(seg), seg, cfg.round_trip_cost)
a3.plot(r.index, (1 + r).cumprod(), label="PPO")
a3.plot(bh.index, (1 + bh).cumprod(), ls="--", label="단순보유")
a3.legend()
fig.tight_layout()
