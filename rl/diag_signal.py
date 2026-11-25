"""매수 신호 예측력 진단
주간 매수 신호가 나온 날의 피처만으로 그 매매의 성패(수익>0)를 미리 구분할 수 있는지 연도별로 검사한다.
실행: python -m rl.diag_signal [--universe kospi200]
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from rl.config import PROCESSED_DIR, RLConfig
from rl.run_filter import (META_COLS, build_event_table, load_processed,
                           make_features, market_features, weekly_position)

log = logging.getLogger("rl.diag")


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--universe", default=None, choices=["stocks", "kospi200"])
    p.add_argument("--test-years", nargs="*", type=int,
                   default=[2020, 2021, 2022, 2023, 2024, 2025, 2026])
    p.add_argument("--train-start", default="2016-01-01")
    args = p.parse_args(argv)

    files = sorted(f for f in Path(PROCESSED_DIR).glob("*.csv") if f.stem.isdigit())
    if args.universe:
        from rl.config import universe_csv
        u = pd.read_csv(universe_csv(args.universe), dtype=str, encoding="utf-8-sig")
        col = next((c for c in u.columns if c in ("종목코드", "code", "ticker", "티커")), u.columns[0])
        codes = set(u[col].str.strip().str.zfill(6))
        files = [f for f in files if f.stem in codes]

    stocks = {}
    for f in files:
        try:
            df = load_processed(f)
            stocks[f.stem] = dict(df=df, pos=weekly_position(df), feat=make_features(df))
        except Exception as e:
            log.warning(f"{f.stem} 건너뜀: {e}")
    cost = RLConfig().round_trip_cost / 2
    ev = build_event_table(stocks, market_features(stocks), cost)
    ev = ev[ev["ret"].notna()].reset_index(drop=True)
    feat_cols = [c for c in ev.columns if c not in META_COLS]
    log.info(f"{len(stocks)}종목, 매수 신호(결과 확정) {len(ev)}건, 피처 {len(feat_cols)}개")

    from lightgbm import LGBMClassifier
    from sklearn.metrics import roc_auc_score

    rows = []
    for Y in args.test_years:
        tr_end = pd.Timestamp(f"{Y - 1}-12-31")
        tr = ev[(ev["date"] >= pd.Timestamp(args.train_start)) & (ev["exit_exec"] <= tr_end)]
        te = ev[(ev["date"] >= pd.Timestamp(f"{Y}-01-01")) & (ev["date"] <= pd.Timestamp(f"{Y}-12-31"))]
        if len(tr) < 100 or len(te) < 30:
            log.warning(f"{Y}: 표본 부족 (학습 {len(tr)}, 테스트 {len(te)})")
            continue
        clf = LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15,
                             min_child_samples=30, subsample=0.8, subsample_freq=1,
                             colsample_bytree=0.8, random_state=0, verbose=-1)
        clf.fit(tr[feat_cols].to_numpy(float), (tr["ret"] > 0).astype(int).to_numpy())
        prob = clf.predict_proba(te[feat_cols].to_numpy(float))[:, 1]
        ret = te["ret"].to_numpy(float)
        y = (ret > 0).astype(int)
        auc = roc_auc_score(y, prob) if 0 < y.sum() < len(y) else np.nan
        ic = pd.Series(prob).corr(pd.Series(ret), method="spearman")
        q = pd.qcut(pd.Series(prob).rank(method="first"), 5, labels=False).to_numpy()
        rows.append({
            "테스트": Y, "학습건수": len(tr), "테스트건수": len(te),
            "승률(%)": y.mean() * 100, "AUC": auc, "순위상관": ic,
            "전체평균(%)": ret.mean() * 100,
            "상위20%평균(%)": ret[q == 4].mean() * 100,
            "하위20%평균(%)": ret[q == 0].mean() * 100,
        })

    res = pd.DataFrame(rows).set_index("테스트")
    res["상위-하위(%p)"] = res["상위20%평균(%)"] - res["하위20%평균(%)"]
    with pd.option_context("display.width", 200, "display.float_format", "{:.3f}".format):
        print("\n=== 매수 신호 예측력 (학습: 테스트 전년도까지) ===")
        print(res)
        print(f"\n평균 AUC {res['AUC'].mean():.3f} / 평균 순위상관 {res['순위상관'].mean():.3f} / "
              f"상위-하위 양수 연도 {(res['상위-하위(%p)'] > 0).sum()}/{len(res)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
