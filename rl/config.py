"""강화학습 실험 설정."""
from dataclasses import dataclass
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "rl"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MARKET_CSV = DATA_DIR / "market.csv"
MANIFEST_CSV = DATA_DIR / "manifest.csv"
OUT_DIR = BASE_DIR / "output" / "rl"


def universe_csv(kind):
    return DATA_DIR / f"universe_{kind}.csv"


@dataclass(frozen=True)
class Fold:
    name: str
    train_end: str
    val_start: str
    val_end: str
    test_start: str
    test_end: str


# 전진 분석: 학습 구간은 앞으로 늘어나고, 검증·테스트는 1년씩 밉니다.
FOLDS = (
    Fold("F2024", "2022-12-31", "2023-01-01", "2023-12-31", "2024-01-01", "2024-12-31"),
    Fold("F2025", "2023-12-31", "2024-01-01", "2024-12-31", "2025-01-01", "2025-12-31"),
    Fold("F2026", "2024-12-31", "2025-01-01", "2025-12-31", "2026-01-01", "2026-12-31"),
)


@dataclass
class RLConfig:
    data_start: str = "2015-01-01"    # 앞 1년은 지표 안정화(warmup)
    warmup_days: int = 250
    window: int = 20                  # 관측 시계열 길이(거래일)
    round_trip_cost: float = 0.003    # 왕복 비용
    action_levels: tuple = (0.0, 1.0) # 목표 포지션
    reward_mode: str = "dd"           # "log" | "dd" | "dsr"
    dd_penalty: float = 0.5
    dsr_eta: float = 0.01
    gamma: float = 0.98
    episode_len: int = 252
    n_envs: int = 4
    total_timesteps: int = 400_000
    eval_freq: int = 20_000
    patience: int = 5
    seeds: tuple = (0, 1, 2, 3, 4)
    net_arch: tuple = (64, 64)
    learning_rate: float = 3e-4
    ent_coef: float = 0.01
    sup_horizon: int = 10
    random_draws: int = 20
    verbose: int = 0                  # 1이면 터미널에 SB3 학습 표 출력
    progress_bar: bool = True         # 학습 진행 막대
    tensorboard: bool = True
    workers: int = 4                  # 지표 계산 병렬 프로세스 수
