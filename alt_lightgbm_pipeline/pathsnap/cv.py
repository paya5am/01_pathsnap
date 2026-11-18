"""Trip-grouped cross-validation (team rule 4, decision D12)."""
from __future__ import annotations

import numpy as np
from sklearn.model_selection import GroupKFold

SEED = 42
MERGE = {16: 10}   # trips 10 and 16 share ~74% of their geometry -> same fold


def cv_groups(trip_ids: np.ndarray) -> np.ndarray:
    g = np.asarray(trip_ids).copy()
    for a, b in MERGE.items():
        g[g == a] = b
    return g


def trip_folds(trip_ids: np.ndarray, n_splits: int = 5):
    """Yield (train_idx, valid_idx) over rows; no trip is ever split across folds.
    GroupKFold is deterministic; groups are shuffled once with SEED for balance."""
    g = cv_groups(trip_ids)
    uniq = np.unique(g)
    perm = np.random.RandomState(SEED).permutation(len(uniq))
    remap = dict(zip(uniq, perm))
    g2 = np.vectorize(remap.get)(g)
    yield from GroupKFold(n_splits=n_splits).split(np.zeros(len(g2)), groups=g2)
