"""비교 기준: 단순 보유, 기존 신호 전략, 거래 수를 맞춘 무작위 매매."""
import numpy as np

SIGNAL_STRATEGIES = {
    "주간 신호": ("Weekly_Buy", "Weekly_Sell"),
    "주·월간 신호": ("Weekly_Monthly_Buy", "Weekly_Monthly_Sell"),
    "이동평균 신호": ("이동평균매수", "이동평균매도"),
}


def buy_hold(seg):
    return np.ones(seg.n_decisions)


def signal_positions(seg, buy_col, sell_col):
    """기존 신호를 같은 회계(다음날 시가 체결, 같은 비용)로 평가합니다."""
    buy = seg.raw[buy_col].fillna(0).to_numpy()
    sell = seg.raw[sell_col].fillna(0).to_numpy()
    cur, out = 0.0, []

    for t in range(seg.first + seg.n_decisions):   # 관측 창 이력부터 상태 이어받기
        if buy[t] == 1:
            cur = 1.0
        elif sell[t] == 1:
            cur = 0.0
        if t >= seg.first:
            out.append(cur)

    return np.array(out)


def random_like(pos, rng):
    """RL과 거래 횟수·보유 기간 분포가 같은 무작위 진입."""
    pos = np.asarray(pos) > 0
    n = len(pos)
    runs, length = [], 0

    for p in pos:
        if p:
            length += 1
        elif length:
            runs.append(length)
            length = 0

    if length:
        runs.append(length)

    out = np.zeros(n)

    if not runs:
        return out

    gaps = rng.multinomial(n - sum(runs), np.ones(len(runs) + 1) / (len(runs) + 1))
    i = gaps[0]

    for run, gap in zip(rng.permutation(runs), gaps[1:]):
        out[i: i + run] = 1.0
        i += run + gap

    return out
