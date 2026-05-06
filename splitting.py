"""
splitting.py — Train / validation / test split utilities (student-implementable).

``split_data`` receives the label array ``y`` and, optionally, the full
DataFrame ``df`` (for group-aware splits).  It must return a list of
``(idx_train, idx_val, idx_test)`` tuples of integer index arrays.

Strategy
--------
This dataset has 689 labelled samples with class ratio ~70/30 (483 hall, 206
truth) and shared SQuAD passages: 538 unique contexts, up to 5 samples sharing
a passage.  A naive stratified split would leak passage-level features across
folds, so we group by passage.

We carve out an honest holdout (``HOLDOUT_FRAC``) once and use the remaining
data for ``N_FOLDS``-fold CV.  In every fold the holdout is the test split, so
``evaluate.run_evaluation`` reports test metrics on the same held-out passages
each time — averaging across folds reduces variance from the train/val split
without ever touching the holdout for tuning.

Group assignment falls back to per-row groups when ``df`` is missing or the
passage cannot be parsed.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    GroupKFold,
    StratifiedGroupKFold,
    StratifiedKFold,
    train_test_split,
)


N_FOLDS = 5
HOLDOUT_FRAC = 0.15
RANDOM_STATE = 42


_CTX_RE = re.compile(r"complete sentence\.\s*\n\s*(.*?)\s*\n\s*Note that", re.DOTALL)


def _extract_groups(df: pd.DataFrame | None, n: int) -> np.ndarray:
    """Return integer group ids (one per row) keyed by SQuAD passage.

    If ``df`` is missing or parsing fails, every row gets a unique group, which
    degenerates the group-aware splitter into a regular stratified one.
    """
    if df is None or "prompt" not in df.columns:
        return np.arange(n)

    groups: list[str] = []
    for prompt in df["prompt"].astype(str).tolist():
        m = _CTX_RE.search(prompt)
        groups.append(m.group(1).strip() if m else f"__row_{len(groups)}")

    seen: dict[str, int] = {}
    out = np.empty(len(groups), dtype=np.int64)
    for i, g in enumerate(groups):
        if g not in seen:
            seen[g] = len(seen)
        out[i] = seen[g]
    return out


def _holdout_split(
    y: np.ndarray,
    groups: np.ndarray,
    holdout_frac: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Split indices into (dev, holdout), keeping groups intact and balancing classes."""
    n = len(y)
    n_groups = len(np.unique(groups))

    if n_groups <= 1:
        idx_dev, idx_hold = train_test_split(
            np.arange(n),
            test_size=holdout_frac,
            random_state=seed,
            stratify=y,
        )
        return idx_dev, idx_hold

    n_splits_for_holdout = max(2, round(1.0 / holdout_frac))
    sgkf = StratifiedGroupKFold(
        n_splits=n_splits_for_holdout,
        shuffle=True,
        random_state=seed,
    )
    idx_dev, idx_hold = next(sgkf.split(np.arange(n), y, groups))
    return idx_dev, idx_hold


def _kfold_splits(
    y: np.ndarray,
    groups: np.ndarray,
    idx_dev: np.ndarray,
    idx_hold: np.ndarray,
    n_folds: int,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray | None, np.ndarray]]:
    """Build (train, val, test=holdout) tuples, one per CV fold on idx_dev."""
    y_dev = y[idx_dev]
    g_dev = groups[idx_dev]
    n_unique_dev_groups = len(np.unique(g_dev))

    if n_unique_dev_groups <= 1:
        splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        iterator = splitter.split(np.arange(len(idx_dev)), y_dev)
    elif n_unique_dev_groups < n_folds:
        splitter = GroupKFold(n_splits=max(2, n_unique_dev_groups))
        iterator = splitter.split(np.arange(len(idx_dev)), y_dev, g_dev)
    else:
        splitter = StratifiedGroupKFold(
            n_splits=n_folds,
            shuffle=True,
            random_state=seed,
        )
        iterator = splitter.split(np.arange(len(idx_dev)), y_dev, g_dev)

    splits: list[tuple[np.ndarray, np.ndarray | None, np.ndarray]] = []
    for tr_local, va_local in iterator:
        idx_train = idx_dev[tr_local]
        idx_val = idx_dev[va_local]
        splits.append((idx_train, idx_val, idx_hold.copy()))
    return splits


def split_data(
    y: np.ndarray,
    df: pd.DataFrame | None = None,
    test_size: float = HOLDOUT_FRAC,
    val_size: float = 0.0,
    random_state: int = RANDOM_STATE,
) -> list[tuple[np.ndarray, np.ndarray | None, np.ndarray]]:
    """Build a list of (idx_train, idx_val, idx_test) tuples.

    Holdout (``test_size`` fraction) is selected once with passage-aware
    stratification.  The remaining indices are split into ``N_FOLDS`` train/val
    folds, also passage-aware.  In every fold the test split is the same
    holdout, so ``evaluate.run_evaluation`` averages CV variance into a single
    honest test metric.
    """
    del val_size
    y = np.asarray(y).astype(int)
    n = len(y)
    groups = _extract_groups(df, n)

    idx_dev, idx_hold = _holdout_split(y, groups, test_size, random_state)
    splits = _kfold_splits(y, groups, idx_dev, idx_hold, N_FOLDS, random_state)

    return splits
