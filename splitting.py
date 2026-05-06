"""
splitting.py — Train / validation / test split utilities (student-implementable).

``split_data`` receives the label array ``y`` and, optionally, the full
DataFrame ``df`` (for group-aware splits).  It must return a list of
``(idx_train, idx_val, idx_test)`` tuples of integer index arrays.

Contract
--------
* ``idx_train``, ``idx_val``, ``idx_test`` are 1-D NumPy arrays of integer
  indices into the full dataset.
* ``idx_val`` may be ``None`` if no separate validation fold is needed.
* All indices must be non-overlapping; together they must cover every sample.
* Return a **list** — one element for a single split, K elements for k-fold.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split


def split_data(
    y: np.ndarray,
    df: pd.DataFrame | None = None,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
) -> list[tuple[np.ndarray, np.ndarray | None, np.ndarray]]:
    """Split dataset indices into train, validation, and test subsets.

    The default strategy performs a single stratified random split preserving
    the class ratio in each subset.

    Args:
        y:            Label array of shape ``(N,)`` with values in ``{0, 1}``.
                      Used for stratification.
        df:           Optional full DataFrame (same row order as ``y``).
                      Required for group-aware splits.
        test_size:    Fraction of samples reserved for the held-out test set.
        val_size:     Fraction of samples reserved for validation.
        random_state: Random seed for reproducible splits.

    Returns:
        A list of ``(idx_train, idx_val, idx_test)`` tuples of integer index
        arrays.  ``idx_val`` may be ``None``.

    Student task:
        Replace or extend the skeleton below.  The only contract is that the
        function returns the list described above.
    """

    idx = np.arange(len(y))
    y = np.asarray(y)

    _, class_counts = np.unique(y, return_counts=True)
    max_stratified_folds = int(class_counts.min()) if len(class_counts) > 1 else 1
    n_splits = min(5, max_stratified_folds)

    if n_splits < 2:
        idx_train, idx_test = train_test_split(
            idx,
            test_size=test_size,
            random_state=random_state,
            stratify=None,
        )
        return [(idx_train, None, idx_test)]

    splitter = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )

    splits: list[tuple[np.ndarray, np.ndarray | None, np.ndarray]] = []
    for fold_idx, (idx_train_val, idx_test) in enumerate(splitter.split(idx, y)):
        y_train_val = y[idx_train_val]
        _, train_val_counts = np.unique(y_train_val, return_counts=True)
        can_make_stratified_val = (
            val_size > 0
            and len(idx_train_val) >= 4
            and len(train_val_counts) > 1
            and int(train_val_counts.min()) >= 2
        )

        if can_make_stratified_val:
            idx_train, idx_val = train_test_split(
                idx_train_val,
                test_size=val_size,
                random_state=random_state + fold_idx,
                stratify=y_train_val,
            )
        else:
            idx_train, idx_val = idx_train_val, None

        splits.append(
            (
                np.asarray(idx_train, dtype=int),
                None if idx_val is None else np.asarray(idx_val, dtype=int),
                np.asarray(idx_test, dtype=int),
            )
        )

    return splits
