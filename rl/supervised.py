"""지도학습 기준 모델: 10거래일 뒤 수익이 비용을 넘는지 분류합니다."""
import lightgbm as lgb
import numpy as np

from .evaluate import evaluate_positions

LAGS = 5


def _rows(seg, ts):
    return np.concatenate([seg.X[ts - k] for k in range(LAGS)], axis=1)


def _table(segs, h, cost):
    xs, ys = [], []

    for seg in segs:
        # 라벨에 쓰는 t+h가 구간 끝을 넘지 않게 해서 다음 구간 정보 누수를 막습니다.
        ts = np.arange(seg.first, len(seg.open) - h)
        if len(ts) == 0:
            continue
        xs.append(_rows(seg, ts))
        ys.append((np.log(seg.close[ts + h] / seg.open[ts + 1]) > np.log1p(cost)).astype(int))

    return np.vstack(xs), np.concatenate(ys)


def _proba(model, seg):
    ts = np.arange(seg.first, seg.first + seg.n_decisions)
    return model.predict_proba(_rows(seg, ts))[:, 1]


def _hysteresis(p, enter, exit_):
    pos, cur = np.zeros(len(p)), 0.0

    for i, x in enumerate(p):
        if cur == 0 and x > enter:
            cur = 1.0
        elif cur == 1 and x < exit_:
            cur = 0.0
        pos[i] = cur

    return pos


def run_supervised(train, val, test, cfg):
    x_tr, y_tr = _table(train, cfg.sup_horizon, cfg.round_trip_cost)
    x_va, y_va = _table(val, cfg.sup_horizon, cfg.round_trip_cost)

    model = lgb.LGBMClassifier(
        n_estimators=1000, learning_rate=0.03, num_leaves=15,
        min_child_samples=200, subsample=0.8, subsample_freq=1,
        colsample_bytree=0.8, reg_lambda=1.0, random_state=0, verbose=-1,
    )
    model.fit(
        x_tr, y_tr,
        eval_set=[(x_va, y_va)],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )

    # 진입·청산 임계값은 검증 구간에서만 고릅니다.
    p_val = [_proba(model, s) for s in val]
    best = (-np.inf, 0.55, 0.45)

    for enter in np.arange(0.50, 0.76, 0.025):
        for exit_ in np.arange(0.30, enter + 1e-9, 0.05):
            m = evaluate_positions(val, [_hysteresis(p, enter, exit_) for p in p_val], cfg)
            if m["거래수"] >= len(val) and m["샤프"] > best[0]:
                best = (m["샤프"], enter, exit_)

    _, enter, exit_ = best
    positions = [_hysteresis(_proba(model, s), enter, exit_) for s in test]

    return positions, {"enter": round(enter, 3), "exit": round(exit_, 3)}
