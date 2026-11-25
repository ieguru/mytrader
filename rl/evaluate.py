"""모든 전략에 똑같이 적용하는 회계와 성과 지표."""
import numpy as np
import pandas as pd


def position_returns(pos, seg, cost):
    """pos[i]: 결정일 d=first+i의 목표 포지션. 반환: 보유일(d+1) 기준 단순수익률."""
    d = seg.first + np.arange(len(pos))
    gross = seg.open[d + 2] / seg.open[d + 1] - 1
    prev = np.concatenate([[0.0], pos[:-1]])
    r = pos * gross - cost / 2 * np.abs(pos - prev)
    return pd.Series(r, index=seg.dates[d + 1])


def trade_returns(pos, r):
    """진입부터 청산일(청산 비용 포함)까지 복리로 계산한 거래별 수익률."""
    out, cur = [], None

    for p, x in zip(pos, r):
        if p > 0:
            cur = (1 + x) if cur is None else cur * (1 + x)
        elif cur is not None:
            out.append(cur * (1 + x) - 1)
            cur = None

    if cur is not None:
        out.append(cur - 1)   # 미청산 포지션은 마지막 날 평가

    return out


def portfolio_stats(r):
    r = r.dropna()

    if r.empty:
        return {"누적수익률(%)": np.nan, "연환산(%)": np.nan, "샤프": np.nan, "MDD(%)": np.nan}

    eq = (1 + r).cumprod()
    years = len(r) / 252
    std = r.std()

    return {
        "누적수익률(%)": (eq.iloc[-1] - 1) * 100,
        "연환산(%)": (eq.iloc[-1] ** (1 / years) - 1) * 100,
        "샤프": r.mean() / std * np.sqrt(252) if std > 0 else 0.0,
        "MDD(%)": (eq / eq.cummax() - 1).min() * 100,
    }


def evaluate_positions(segs, positions, cfg):
    """종목별 동일 비중 포트폴리오(매일 리밸런싱)로 집계합니다."""
    rets, all_pos, trades, turnover = {}, [], [], []

    for seg, pos in zip(segs, positions):
        pos = np.asarray(pos, float)
        r = position_returns(pos, seg, cfg.round_trip_cost)

        rets[seg.ticker] = r
        all_pos.append(pos)
        trades += trade_returns(pos, r.to_numpy())

        years = max(len(pos) / 252, 1 / 252)
        turnover.append(np.abs(np.diff(np.concatenate([[0.0], pos]))).sum() / years)

    port = pd.DataFrame(rets).sort_index().mean(axis=1)
    trades = np.asarray(trades)
    gains, losses = trades[trades > 0].sum(), -trades[trades < 0].sum()

    return {
        **portfolio_stats(port),
        "회전율(연)": float(np.mean(turnover)),
        "노출(%)": float(np.concatenate(all_pos).mean() * 100),
        "거래수": int(len(trades)),
        "승률(%)": float((trades > 0).mean() * 100) if len(trades) else np.nan,
        "Profit Factor": float(gains / losses) if losses > 0 else np.nan,
    }
