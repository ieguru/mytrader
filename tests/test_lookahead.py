import numpy as np
import pandas as pd
import pytest

from analysis_engine import analyze_frame
from rl.config import PROCESSED_DIR
from rl.dataset import PIT_COLUMNS, patterns_at
from rl.features import FEATURE_COLS, build_features
from rl.storage import raw_path, read_csv, read_raw

TICKER, NAME, CUTS = "000660", "SK하이닉스", 12


def analyzed_until(raw, cut):
    data = analyze_frame(raw.loc[:cut], NAME, excel_file=None)["data"]
    for column, value in zip(PIT_COLUMNS, patterns_at(data, len(data) - 1)):
        data.loc[data.index[-1], column] = value
    return data


def test_no_lookahead():
    processed = PROCESSED_DIR / f"{TICKER}.csv"
    if not (raw_path(TICKER).is_file() and processed.is_file()):
        pytest.skip("rl.download 와 rl.run --prepare-only 를 먼저 실행하세요.")

    raw = read_raw(TICKER)
    full = build_features(read_csv(processed))
    cuts = np.random.default_rng(0).choice(full.index[300:-5], CUTS, replace=False)
    bad = {}

    for cut in sorted(cuts):
        part = build_features(analyzed_until(raw, cut)).loc[cut].to_numpy()
        diff = ~np.isclose(part, full.loc[cut].to_numpy(), rtol=1e-6, atol=1e-9, equal_nan=True)
        for column in np.array(FEATURE_COLS)[diff]:
            bad.setdefault(column, []).append(str(pd.Timestamp(cut).date()))

    assert not bad, f"미래 정보가 의심되는 피처: {bad}"
