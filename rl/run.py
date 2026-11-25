# rl/run.py
"""
전진 분석 강화학습 실험 (오프라인: data/rl/ 의 CSV만 사용)

준비
    python -m rl.download --universe stocks          # KRX -> data/rl/raw/*.csv (1회)
    python -m rl.run --prepare-only                  # 지표·PIT 신호 -> data/rl/processed/*.csv

실행
    python -m rl.run --debug                         # 초소형(중단점 디버깅용)
    python -m rl.run --folds F2024 --seeds 1 --timesteps 60000
    python -m rl.run --seeds 5 --reward dd           # 본 실험
    python -m rl.run --seeds 5 --reward dsr
    python -m rl.run --seeds 5 --universe kospi200   # 선정 편향 점검

결과
    output/rl/<universe>_<reward>/
        config.json, summary.csv, summary_by_fold.csv
        <fold>/detail.csv, ppo_seed*.zip, val_curve_seed*.csv
        tb/   (TensorBoard 로그)
"""
import argparse
import json
import logging
import os
import time
import dataclasses
from dataclasses import asdict

import numpy as np
import pandas as pd

from .baselines import SIGNAL_STRATEGIES, buy_hold, random_like, signal_positions
from .config import FOLDS, OUT_DIR, RLConfig
from .dataset import load_all, load_market
from .env import make_segment
from .evaluate import evaluate_positions
from .features import Scaler, build_features
from .storage import read_universe

logger = logging.getLogger("rl")

NUMERIC_COLUMNS = [
    "누적수익률(%)", "연환산(%)", "샤프", "MDD(%)",
    "회전율(연)", "노출(%)", "거래수", "승률(%)", "Profit Factor",
]


# =========================================================
# 구간 준비
# =========================================================

def segments(data, scaled, start, end, cfg):
    """종목별 [start, end] 결정 구간 세그먼트. 결정일이 30일 미만인 종목은 제외합니다."""
    out = [make_segment(t, scaled[t], data[t], start, end, cfg) for t in data]
    return [s for s in out if s is not None]


def prepare_fold(fold, data, feats, cfg):
    """
    학습 구간 통계로만 정규화한 (학습, 검증, 테스트) 세그먼트를 돌려줍니다.
    inspect_run.py에서도 같은 함수로 구간을 재현합니다.
    """
    train_frames = [
        f.loc[:fold.train_end].iloc[cfg.warmup_days:] for f in feats.values()
    ]
    train_frames = [f for f in train_frames if len(f)]

    if not train_frames:
        raise ValueError(f"[{fold.name}] 학습 구간 데이터가 없습니다.")

    scaler = Scaler().fit(train_frames)
    scaled = {t: scaler.transform(f) for t, f in feats.items()}

    return (
        segments(data, scaled, cfg.data_start, fold.train_end, cfg),
        segments(data, scaled, fold.val_start, fold.val_end, cfg),
        segments(data, scaled, fold.test_start, fold.test_end, cfg),
    )


# =========================================================
# 구간 1개 실행
# =========================================================

def run_fold(fold, data, feats, cfg, out_dir, skip_supervised=False):
    # 무거운 모듈(torch, lightgbm)은 실제 학습할 때만 import합니다.
    from .train import policy_positions, train_ppo

    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()

    train, val, test = prepare_fold(fold, data, feats, cfg)
    logger.info(
        "[%s] 학습 %d / 검증 %d / 테스트 %d 종목",
        fold.name, len(train), len(val), len(test),
    )

    if not (train and val and test):
        logger.warning("[%s] 비어 있는 구간이 있어 건너뜁니다.", fold.name)
        return None

    rows = []

    def add(name, positions, seed=None, extra=None):
        metrics = evaluate_positions(test, positions, cfg)
        metrics.update({"전략": name, "seed": seed, **(extra or {})})
        rows.append(metrics)

    # ---------------- 비교 기준 ----------------
    add("단순보유", [buy_hold(s) for s in test])

    for name, (buy_col, sell_col) in SIGNAL_STRATEGIES.items():
        add(name, [signal_positions(seg, buy_col, sell_col) for seg in test])

    if not skip_supervised:
        from .supervised import run_supervised

        logger.info("[%s] 지도학습(LightGBM) 학습", fold.name)
        sup_pos, thresholds = run_supervised(train, val, test, cfg)
        add("지도학습(LightGBM)", sup_pos,
            extra={"설정": json.dumps(thresholds, ensure_ascii=False)})

    # ---------------- 강화학습 ----------------
    rng = np.random.default_rng(0)

    for seed in cfg.seeds:
        logger.info("[%s] PPO seed %d 학습 시작", fold.name, seed)
        model, history = train_ppo(train, val, cfg, seed, out_dir / f"ppo_seed{seed}.zip")

        pd.DataFrame(history, columns=["timesteps", "val_sharpe"]).to_csv(
            out_dir / f"val_curve_seed{seed}.csv", index=False
        )

        rl_pos = [policy_positions(model, s, cfg) for s in test]
        add("강화학습(PPO)", rl_pos, seed)

        # 같은 거래 수·보유 기간 분포의 무작위 매매(진입 타이밍 효과 확인용)
        draws = [
            evaluate_positions(test, [random_like(p, rng) for p in rl_pos], cfg)
            for _ in range(cfg.random_draws)
        ]
        random_row = pd.DataFrame(draws).mean(numeric_only=True).to_dict()
        random_row.update({"전략": "무작위(거래 수 일치)", "seed": seed})
        rows.append(random_row)

    detail = pd.DataFrame(rows)
    detail.to_csv(out_dir / "detail.csv", index=False, encoding="utf-8-sig")

    logger.info("[%s] 완료 (%.1f분)", fold.name, (time.time() - started) / 60)
    return detail


# =========================================================
# 결과 요약
# =========================================================

def summarize(details, run_dir):
    all_detail = pd.concat(details, ignore_index=True)
    all_detail.to_csv(run_dir / "detail_all.csv", index=False, encoding="utf-8-sig")

    columns = [c for c in NUMERIC_COLUMNS if c in all_detail]

    by_fold = (
        all_detail.groupby(["fold", "전략"])[columns]
        .agg(["mean", "std"])
        .round(2)
    )
    by_fold.to_csv(run_dir / "summary_by_fold.csv", encoding="utf-8-sig")

    overall = (
        all_detail.groupby("전략")[columns]
        .mean()
        .round(2)
        .sort_values("샤프", ascending=False)
    )
    overall.to_csv(run_dir / "summary.csv", encoding="utf-8-sig")

    # 구간별 샤프 비교표: RL이 '모든 구간'에서 이기는지 한눈에 봅니다.
    sharpe = all_detail.pivot_table(index="전략", columns="fold", values="샤프", aggfunc="mean")
    sharpe.round(2).to_csv(run_dir / "sharpe_by_fold.csv", encoding="utf-8-sig")

    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print("\n=== 전체 평균 (구간·시드) ===")
        print(overall.to_string())
        print("\n=== 구간별 샤프 ===")
        print(sharpe.round(2).to_string())

    return overall


# =========================================================
# 진입점
# =========================================================

def parse_args(argv):
    p = argparse.ArgumentParser(description="전진 분석 강화학습 실험 (CSV 오프라인)")
    p.add_argument("--universe", choices=["stocks", "kospi200"], default="stocks")
    p.add_argument("--folds", default="all", help="예: F2024,F2025")
    p.add_argument("--seeds", type=int, default=5, help="시드 개수(0부터)")
    p.add_argument("--timesteps", type=int, help="PPO 총 학습 스텝")
    p.add_argument("--reward", choices=["log", "dd", "dsr"], default="dd")
    p.add_argument("--levels", default="0,1", help="목표 포지션 단계. 예: 0,0.5,1")
    p.add_argument("--max-stocks", type=int, help="앞에서부터 N종목만 사용")
    p.add_argument("--workers", type=int, help="지표 계산 프로세스 수")
    p.add_argument("--refresh", action="store_true", help="processed/ 캐시 무시하고 재계산")
    p.add_argument("--prepare-only", action="store_true", help="processed/ 생성까지만")
    p.add_argument("--skip-supervised", action="store_true", help="LightGBM 기준 모델 생략")
    p.add_argument("--verbose", action="store_true", help="SB3 학습 표 출력")
    p.add_argument("--debug", action="store_true", help="디버깅용 초소형 실행")
    p.add_argument("--no-tb", action="store_true", help="TensorBoard 기록 끄기(장시간 실험용)")
    return p.parse_args(argv)


def build_config(args):
    cfg = RLConfig(
        reward_mode=args.reward,
        seeds=tuple(range(args.seeds)),
        action_levels=tuple(float(x) for x in args.levels.split(",")),
    )

    if args.timesteps:
        cfg.total_timesteps = args.timesteps
    if args.verbose:
        cfg.verbose = 1

    if args.workers is None:
        args.workers = getattr(cfg, "workers", None) or max(1, (os.cpu_count() or 2) // 2)

    if args.debug:
        cfg.n_envs, cfg.total_timesteps, cfg.eval_freq = 1, 6000, 2000
        cfg.seeds, cfg.random_draws, cfg.progress_bar = (0,), 2, False
        args.folds = "F2024" if args.folds == "all" else args.folds
        args.max_stocks = args.max_stocks or 5
        args.workers = 1   # 단일 프로세스여야 중단점이 걸립니다

    if len(cfg.action_levels) < 2 or cfg.action_levels[0] != 0.0:
        raise SystemExit("--levels 는 0으로 시작하는 2개 이상의 값이어야 합니다. 예: 0,1")

    if args.no_tb:
        cfg = dataclasses.replace(cfg, tensorboard=False)

    return cfg


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = build_config(args)

    # ---------------- 데이터 (CSV) ----------------
    universe = read_universe(args.universe)
    if args.max_stocks:
        universe = universe[: args.max_stocks]

    data = load_all(universe, refresh=args.refresh, workers=args.workers)

    if not data:
        logger.error("사용할 수 있는 종목이 없습니다. rl.download 결과(manifest.csv)를 확인하세요.")
        return 1

    market = load_market(data)

    if args.prepare_only:
        logger.info("처리 완료: %d / %d 종목", len(data), len(universe))
        return 0

    feats = {t: build_features(d, market) for t, d in data.items()}

    # ---------------- 구간 선택 ----------------
    wanted = None if args.folds == "all" else set(args.folds.split(","))
    folds = [f for f in FOLDS if wanted is None or f.name in wanted]

    if not folds:
        logger.error("구간 이름이 올바르지 않습니다: %s (가능: %s)",
                     args.folds, ", ".join(f.name for f in FOLDS))
        return 1

    suffix = "_debug" if args.debug else ""
    run_dir = OUT_DIR / f"{args.universe}_{args.reward}{suffix}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(
        json.dumps(asdict(cfg), ensure_ascii=False, indent=2), encoding="utf-8"
    )

    logger.info("실험 시작: %d종목 / 구간 %s / 시드 %s / 보상 %s",
                len(data), [f.name for f in folds], list(cfg.seeds), cfg.reward_mode)

    # ---------------- 실행 ----------------
    details = []
    for fold in folds:
        detail = run_fold(fold, data, feats, cfg, run_dir / fold.name,
                          skip_supervised=args.skip_supervised)
        if detail is not None:
            details.append(detail.assign(fold=fold.name))

    if not details:
        logger.error("실행된 구간이 없습니다.")
        return 1

    summarize(details, run_dir)
    logger.info("결과 저장: %s", run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
