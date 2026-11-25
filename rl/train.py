"""PPO 학습. TensorBoard에 학습·검증 지표와 매매 그림을 기록합니다."""
import logging

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.logger import Figure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from .env import TradingEnv
from .evaluate import evaluate_positions, position_returns

logger = logging.getLogger("rl")


def policy_positions(model, seg, cfg):
    env = TradingEnv([seg], cfg, train=False)
    obs, _ = env.reset()
    out = []

    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(int(action))
        out.append(info["pos"])
        if terminated or truncated:
            break

    return np.array(out)


class RolloutStats(BaseCallback):
    """학습 중 정책의 행동 습관: 노출 비율, 거래 빈도, 일평균 손익."""

    def _on_rollout_start(self):
        self.pos, self.turn, self.pnl = [], [], []

    def _on_step(self):
        for info in self.locals["infos"]:
            self.pos.append(info["pos"])
            self.turn.append(info["turn"])
            self.pnl.append(info["pnl"])
        return True

    def _on_rollout_end(self):
        if not self.pos:
            return
        self.logger.record("train/exposure", float(np.mean(self.pos)))
        self.logger.record("train/trades_per_100d", float(np.mean(self.turn) * 100))
        self.logger.record("train/daily_pnl_bp", float(np.mean(self.pnl) * 1e4))


class ValidationStop(BaseCallback):
    """검증 샤프로 최적 모델을 저장하고 조기 종료합니다. 지표와 그림을 TensorBoard에 남깁니다."""

    def __init__(self, val_segs, cfg, path):
        super().__init__()
        self.val_segs, self.cfg, self.path = val_segs, cfg, path
        self.best, self.bad, self.last = -np.inf, 0, 0
        self.history = []

    def _figure(self, seg, pos):
        d = seg.first + np.arange(len(pos))
        x, close = seg.dates[d], seg.close[d]
        r = position_returns(pos, seg, self.cfg.round_trip_cost)
        bh = position_returns(np.ones(len(pos)), seg, self.cfg.round_trip_cost)

        fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
        a1.plot(x, close, color="black", lw=1)
        a1.fill_between(x, close.min(), close.max(), where=pos > 0,
                        color="tab:red", alpha=0.15, step="post")
        a1.set_title(f"{seg.ticker}  close (shaded = holding)  step={self.num_timesteps}")
        a2.plot(r.index, (1 + r).cumprod(), label="PPO")
        a2.plot(bh.index, (1 + bh).cumprod(), ls="--", label="Buy&Hold")
        a2.legend(loc="upper left")
        fig.tight_layout()
        return fig

    def _on_step(self):
        if self.num_timesteps - self.last < self.cfg.eval_freq:
            return True

        self.last = self.num_timesteps
        positions = [policy_positions(self.model, s, self.cfg) for s in self.val_segs]
        m = evaluate_positions(self.val_segs, positions, self.cfg)
        score = float(m["샤프"]) if np.isfinite(m["샤프"]) else 0.0
        self.history.append((self.num_timesteps, score))

        for key, name in (("샤프", "sharpe"), ("연환산(%)", "annual_pct"),
                          ("MDD(%)", "mdd_pct"), ("노출(%)", "exposure_pct"),
                          ("거래수", "trades")):
            value = m[key]
            if value is not None and np.isfinite(value):
                self.logger.record(f"val/{name}", float(value))

        self.logger.record(
            "val/positions",
            Figure(self._figure(self.val_segs[0], positions[0]), close=True),
            exclude=("stdout", "log", "json", "csv"),
        )

        if score > self.best + 1e-3:
            self.best, self.bad = score, 0
            self.model.save(self.path)
            mark = "★ 최고 갱신"
        else:
            self.bad += 1
            mark = f"개선 없음 {self.bad}/{self.cfg.patience}"

        self.logger.record("val/best_sharpe", self.best)
        self.logger.dump(self.num_timesteps)
        logger.info("[검증] step=%d 샤프=%.2f 노출=%.0f%% 거래=%d  %s",
                    self.num_timesteps, score, m["노출(%)"], m["거래수"], mark)

        return self.bad < self.cfg.patience


def train_ppo(train_segs, val_segs, cfg, seed, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.unlink(missing_ok=True)

    # Monitor로 감싸야 rollout/ep_rew_mean(에피소드 보상)이 기록됩니다.
    env = DummyVecEnv([
        (lambda i=i: Monitor(TradingEnv(train_segs, cfg, train=True, seed=seed * 100 + i)))
        for i in range(cfg.n_envs)
    ])

    run_dir = path.parent.parent                       # output/rl/<universe>_<reward>
    model = PPO(
        "MlpPolicy", env, seed=seed, device="cpu", verbose=cfg.verbose,
        gamma=cfg.gamma, learning_rate=cfg.learning_rate,
        n_steps=1024, batch_size=256, ent_coef=cfg.ent_coef,
        policy_kwargs={"net_arch": list(cfg.net_arch)},
        tensorboard_log=(str(run_dir / "tb") if cfg.tensorboard else None),
    )

    stopper = ValidationStop(val_segs, cfg, path)
    model.learn(
        cfg.total_timesteps,
        callback=CallbackList([RolloutStats(), stopper]),
        tb_log_name=f"{path.parent.name}_seed{seed}",   # 예: F2024_seed0
        progress_bar=cfg.progress_bar,
    )

    if path.is_file():
        model = PPO.load(path, device="cpu")

    return model, stopper.history
