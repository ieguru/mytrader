"""
목표 포지션 환경.
시점: t일 종가 후 결정 -> t+1 시가 체결 -> t+2 시가까지의 손익이 t의 보상.
"""
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from .features import FEATURE_COLS

REWARD_SCALE = 100.0


@dataclass
class Segment:
    ticker: str
    X: np.ndarray
    open: np.ndarray
    close: np.ndarray
    dates: pd.DatetimeIndex
    first: int
    raw: pd.DataFrame

    @property
    def n_decisions(self):
        return len(self.open) - 2 - self.first


def make_segment(ticker, scaled, data, start, end, cfg):
    idx = data.index
    dec = np.flatnonzero((idx >= pd.Timestamp(start)) & (idx <= pd.Timestamp(end)))
    dec = dec[dec >= max(cfg.warmup_days, cfg.window - 1)]

    if len(dec) < 30:
        return None

    lo, hi = dec[0] - cfg.window + 1, dec[-1] + 1
    raw = data.iloc[lo:hi]
    close = raw["종가"].to_numpy(float)
    open_ = raw["시가"].to_numpy(float)
    open_ = np.where(open_ > 0, open_, close)   # 거래정지일 시가 0 -> 종가

    return Segment(
        ticker,
        scaled.iloc[lo:hi][FEATURE_COLS].to_numpy(np.float32),
        open_, close, raw.index, cfg.window - 1, raw,
    )


class TradingEnv(gym.Env):
    def __init__(self, segments, cfg, train=True, seed=None):
        super().__init__()
        self.segments = segments
        self.cfg = cfg
        self.train = train
        self.levels = np.asarray(cfg.action_levels, float)

        n_feat = segments[0].X.shape[1]
        self.action_space = spaces.Discrete(len(self.levels))
        self.observation_space = spaces.Box(
            -np.inf, np.inf, (cfg.window * n_feat + 3,), np.float32
        )

        lengths = np.array([s.n_decisions for s in segments], float)
        self.p = lengths / lengths.sum()
        self.rng = np.random.default_rng(seed)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if seed is not None:
            self.rng = np.random.default_rng(seed)

        if self.train:
            self.seg = self.segments[self.rng.choice(len(self.segments), p=self.p)]
            last_start = len(self.seg.open) - 2 - self.cfg.episode_len

            if last_start > self.seg.first:
                self.t = int(self.rng.integers(self.seg.first, last_start))
                self.limit = self.cfg.episode_len
            else:
                self.t, self.limit = self.seg.first, None
        else:
            self.seg = self.segments[(options or {}).get("segment", 0)]
            self.t, self.limit = self.seg.first, None

        self.pos, self.hold, self.entry, self.steps = 0.0, 0, None, 0
        self.eq = self.peak = self.dd = 0.0
        self.A = self.B = 0.0

        return self._obs(), {}

    def _obs(self):
        w = self.cfg.window
        hist = self.seg.X[self.t - w + 1: self.t + 1].ravel()
        # 평가손익은 t일 '종가' 기준(t+1 시가는 아직 모름)
        unrealized = np.log(self.seg.close[self.t] / self.entry) if self.pos > 0 else 0.0
        state = np.array([self.pos, min(self.hold, 60) / 20, unrealized], np.float32)
        return np.concatenate([hist, state]).astype(np.float32)

    def _shape(self, pnl):
        mode = self.cfg.reward_mode

        if mode == "log":
            return pnl * REWARD_SCALE

        if mode == "dd":
            self.eq += pnl
            self.peak = max(self.peak, self.eq)
            dd = self.peak - self.eq
            penalty = max(0.0, dd - self.dd)
            self.dd = dd
            return (pnl - self.cfg.dd_penalty * penalty) * REWARD_SCALE

        # 차등 샤프비율
        eta = self.cfg.dsr_eta
        dA, dB = pnl - self.A, pnl * pnl - self.B
        var = self.B - self.A ** 2

        if self.steps > 20 and var > 1e-12:
            reward = (self.B * dA - 0.5 * self.A * dB) / var ** 1.5
        else:
            reward = pnl * REWARD_SCALE

        self.A += eta * dA
        self.B += eta * dB
        return float(np.clip(reward, -10, 10))

    def step(self, action):
        new = float(self.levels[int(action)])
        o1, o2 = self.seg.open[self.t + 1], self.seg.open[self.t + 2]
        turn = abs(new - self.pos)

        pnl = new * np.log(o2 / o1) - self.cfg.round_trip_cost / 2 * turn

        if new > 0 and self.pos == 0:
            self.entry = o1
        if new == 0:
            self.entry = None

        self.hold = self.hold + 1 if new > 0 else 0
        self.pos = new
        reward = self._shape(pnl)

        self.t += 1
        self.steps += 1

        terminated = self.t + 2 >= len(self.seg.open)
        truncated = self.limit is not None and self.steps >= self.limit

        info = {"pos": new, "pnl": pnl, "turn": turn, "reward": reward}
        return self._obs(), float(reward), terminated, truncated, info
