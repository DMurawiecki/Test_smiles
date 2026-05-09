"""
probe.py — Hallucination probe classifier (student-implemented).

Implements ``HallucinationProbe``, a binary MLP that classifies feature
vectors as truthful (0) or hallucinated (1).  Called from ``solution.py``
via ``evaluate.run_evaluation``.  All four public methods (``fit``,
``fit_hyperparameters``, ``predict``, ``predict_proba``) must be implemented
and their signatures must not change.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


PROBE_TYPE = os.getenv("PROBE_TYPE", "logreg")
PCA_N_COMPONENTS = os.getenv("PCA_N_COMPONENTS", "128")
THRESHOLD_METRIC = os.getenv("THRESHOLD_METRIC", "accuracy")


IMPORTANCE_GROUPS = (
    "all_spectral",
    "top_eigenvalues",
    "sum_eigenvalues",
    "logdet",
    "effective_rank",
    "participation_ratio",
    "condition_number",
    "spectral_entropy",
    "spectral_by_window_8",
    "spectral_by_window_16",
    "spectral_by_window_32",
    "spectral_by_window_64",
)


def _env_float(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)))


def _env_int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def _env_class_weight() -> str | dict | None:
    value = os.getenv("CLASS_WEIGHT", "none").strip().lower()
    if value in {"", "none", "null"}:
        return None
    return value


def _parse_hidden_layers(value: str) -> tuple[int, ...]:
    cleaned = value.strip().replace("(", "").replace(")", "")
    if not cleaned:
        return (64,)
    return tuple(int(part.strip()) for part in cleaned.split(",") if part.strip())


def _safe_pca_components(X: np.ndarray) -> int | None:
    raw = os.getenv("PCA_N_COMPONENTS", PCA_N_COMPONENTS).strip().lower()
    if raw in {"none", "no", "false", "0"}:
        return None
    requested = int(raw)
    capped = min(requested, X.shape[0] - 1, X.shape[1])
    return capped if capped >= 1 else None


def _make_classifier(probe_type: str, n_samples: int):
    class_weight = _env_class_weight()
    if probe_type == "logreg":
        return LogisticRegression(
            penalty="l2",
            C=_env_float("LOGREG_C", 0.01),
            class_weight=class_weight,
            max_iter=5000,
            solver="lbfgs",
            random_state=42,
        )
    if probe_type == "logreg_l1":
        return LogisticRegression(
            penalty="l1",
            C=_env_float("LOGREG_C", 0.01),
            class_weight=class_weight,
            max_iter=5000,
            solver="liblinear",
            random_state=42,
        )
    if probe_type == "mlp":
        return MLPClassifier(
            hidden_layer_sizes=_parse_hidden_layers(os.getenv("MLP_HIDDEN", "64")),
            activation="relu",
            alpha=_env_float("MLP_ALPHA", 0.001),
            batch_size=min(64, max(1, n_samples)),
            max_iter=2000,
            early_stopping=True,
            random_state=42,
        )
    if probe_type == "svm_linear":
        return SVC(
            kernel="linear",
            probability=True,
            C=_env_float("SVM_C", 0.1),
            class_weight=class_weight,
            random_state=42,
        )
    if probe_type == "svm_rbf":
        return SVC(
            kernel="rbf",
            probability=True,
            C=_env_float("SVM_C", 1.0),
            gamma=os.getenv("SVM_GAMMA", "scale"),
            class_weight=class_weight,
            random_state=42,
        )
    if probe_type == "ridge":
        return RidgeClassifier(
            alpha=_env_float("RIDGE_ALPHA", 1.0),
            class_weight=class_weight,
            random_state=42,
        )
    if probe_type == "knn":
        return KNeighborsClassifier(
            n_neighbors=_env_int("KNN_NEIGHBORS", 5),
            weights=os.getenv("KNN_WEIGHTS", "distance"),
        )
    raise ValueError(
        "PROBE_TYPE must be one of: logreg, mlp, logreg_l1, svm_linear, "
        "svm_rbf, ridge, knn."
    )


def _sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-np.clip(values, -50.0, 50.0)))


def _feature_group_indices(feature_names: list[str]) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = {group: [] for group in IMPORTANCE_GROUPS}
    for idx, name in enumerate(feature_names):
        if not name.startswith("spectral__"):
            continue

        groups["all_spectral"].append(idx)

        if "__top_eig_" in name:
            groups["top_eigenvalues"].append(idx)
        if name.endswith("__sum_eigenvalues"):
            groups["sum_eigenvalues"].append(idx)
        if name.endswith("__logdet"):
            groups["logdet"].append(idx)
        if name.endswith("__effective_rank"):
            groups["effective_rank"].append(idx)
        if name.endswith("__participation_ratio"):
            groups["participation_ratio"].append(idx)
        if name.endswith("__condition_number"):
            groups["condition_number"].append(idx)
        if name.endswith("__spectral_entropy"):
            groups["spectral_entropy"].append(idx)

        for width in (8, 16, 32, 64):
            if f"__window_{width}__" in name:
                groups[f"spectral_by_window_{width}"].append(idx)

    return groups


def _score_probe(
    probe: "HallucinationProbe",
    X: np.ndarray,
    y: np.ndarray,
    metric: str,
) -> float:
    if metric == "accuracy":
        return float(accuracy_score(y, probe.predict(X)))
    if metric == "auroc":
        try:
            return float(roc_auc_score(y, probe.predict_proba(X)[:, 1]))
        except ValueError:
            return float("nan")
    raise ValueError("metric must be either 'accuracy' or 'auroc'.")


def permutation_importance_by_group(
    probe: "HallucinationProbe",
    X_val: np.ndarray,
    y_val: np.ndarray,
    feature_names: list[str],
    metrics: tuple[str, ...] = ("accuracy", "auroc"),
    n_repeats: int = 10,
    random_state: int = 42,
) -> pd.DataFrame:
    """Measure validation-only group permutation importance.

    The fitted probe is kept fixed.  For each spectral feature group, values
    are shuffled across validation rows and the score drop is recorded.
    """
    X_val = np.asarray(X_val, dtype=np.float32)
    y_val = np.asarray(y_val, dtype=int)
    rng = np.random.default_rng(random_state)
    groups = _feature_group_indices(feature_names)

    rows: list[dict] = []
    for metric in metrics:
        baseline_score = _score_probe(probe, X_val, y_val, metric)
        for group, indices in groups.items():
            if not indices:
                continue

            permuted_scores: list[float] = []
            for _ in range(n_repeats):
                X_perm = X_val.copy()
                order = rng.permutation(X_perm.shape[0])
                X_perm[:, indices] = X_perm[order][:, indices]
                permuted_scores.append(_score_probe(probe, X_perm, y_val, metric))

            scores = np.asarray(permuted_scores, dtype=float)
            importances = baseline_score - scores
            rows.append(
                {
                    "group": group,
                    "metric": metric,
                    "baseline_score": baseline_score,
                    "permuted_score_mean": float(np.nanmean(scores)),
                    "permuted_score_std": float(np.nanstd(scores)),
                    "importance_mean": float(np.nanmean(importances)),
                    "importance_std": float(np.nanstd(importances)),
                    "n_features": len(indices),
                }
            )

    return pd.DataFrame(rows)


class HallucinationProbe(nn.Module):
    """Binary classifier that detects hallucinations from hidden-state features.

    Extends ``torch.nn.Module`` for compatibility with the original template.
    By default, ``fit`` trains a small-data-friendly sklearn pipeline:
    StandardScaler, optional PCA, and class-balanced logistic regression.
    """

    def __init__(self) -> None:
        super().__init__()
        self._net: nn.Sequential | None = None  # built lazily in fit()
        self._scaler = StandardScaler()
        self._classifier: Pipeline | None = None
        self._threshold: float = 0.5  # tuned by fit_hyperparameters()
        self._probe_type: str = os.getenv("PROBE_TYPE", PROBE_TYPE)

    # ------------------------------------------------------------------
    # STUDENT: Replace or extend the network definition below.
    # ------------------------------------------------------------------
    def _build_network(self, input_dim: int) -> None:
        """Instantiate the network layers.

        Called once at the start of ``fit()`` when ``input_dim`` is known.

        Args:
            input_dim: Feature vector dimensionality.
        """
        self._net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass — returns raw logits of shape ``(n_samples,)``.

        Args:
            x: Float tensor of shape ``(n_samples, feature_dim)``.

        Returns:
            1-D tensor of raw (pre-sigmoid) logits.
        """
        if self._net is None:
            raise RuntimeError(
                "Network has not been built yet. Call fit() before forward()."
            )
        return self._net(x).squeeze(-1)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "HallucinationProbe":
        """Train the probe on labelled feature vectors.

        Scales features with ``StandardScaler``, applies PCA with a safe
        component count, and trains either logistic regression or an optional
        MLP depending on ``PROBE_TYPE``.

        Args:
            X: Feature matrix of shape ``(n_samples, feature_dim)``.
            y: Integer label vector of shape ``(n_samples,)``; 0 = truthful,
               1 = hallucinated.

        Returns:
            ``self`` (for method chaining).
        """
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=int)

        self._probe_type = os.getenv("PROBE_TYPE", PROBE_TYPE)
        n_components = _safe_pca_components(X)
        steps: list[tuple[str, object]] = [("scaler", StandardScaler())]
        if n_components is not None:
            steps.append(("pca", PCA(n_components=n_components, random_state=42)))

        steps.append(("classifier", _make_classifier(self._probe_type, X.shape[0])))

        self._classifier = Pipeline(steps)
        self._classifier.fit(X, y)
        return self

    def fit_hyperparameters(
        self, X_val: np.ndarray, y_val: np.ndarray
    ) -> "HallucinationProbe":
        """Tune the decision threshold on a validation set.

        The chosen threshold is stored in ``self._threshold`` and used by
        subsequent ``predict`` calls.  ``THRESHOLD_METRIC`` controls whether
        validation accuracy or F1 is maximized.

        Args:
            X_val: Validation feature matrix of shape
                   ``(n_val_samples, feature_dim)``.
            y_val: Integer label vector of shape ``(n_val_samples,)``;
                   0 = truthful, 1 = hallucinated.

        Returns:
            ``self`` (for method chaining).
        """
        probs = self.predict_proba(X_val)[:, 1]

        # Candidate thresholds: unique predicted probabilities plus a coarse grid.
        candidates = np.unique(np.concatenate([probs, np.linspace(0.0, 1.0, 101)]))

        metric = os.getenv("THRESHOLD_METRIC", THRESHOLD_METRIC).strip().lower()
        if metric not in {"accuracy", "f1"}:
            raise ValueError("THRESHOLD_METRIC must be either 'accuracy' or 'f1'.")

        best_threshold = 0.5
        best_score = -1.0
        for t in candidates:
            y_pred_t = (probs >= t).astype(int)
            if metric == "accuracy":
                score = accuracy_score(y_val, y_pred_t)
            else:
                score = f1_score(y_val, y_pred_t, zero_division=0)
            if score > best_score:
                best_score = score
                best_threshold = float(t)

        self._threshold = best_threshold
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict binary labels for feature vectors.

        Uses the decision threshold in ``self._threshold`` (default ``0.5``;
        updated by ``fit_hyperparameters``).

        Args:
            X: Feature matrix of shape ``(n_samples, feature_dim)``.

        Returns:
            Integer array of shape ``(n_samples,)`` with values in ``{0, 1}``.
        """
        return (self.predict_proba(X)[:, 1] >= self._threshold).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class probability estimates.

        Args:
            X: Feature matrix of shape ``(n_samples, feature_dim)``.

        Returns:
            Array of shape ``(n_samples, 2)`` where column 1 contains the
            estimated probability of the hallucinated class (label 1).
            Used to compute AUROC.
        """
        if self._classifier is None:
            raise RuntimeError("Probe has not been fitted yet.")
        X = np.asarray(X, dtype=np.float32)
        if hasattr(self._classifier, "predict_proba"):
            prob_pos = self._classifier.predict_proba(X)[:, 1]
        elif hasattr(self._classifier, "decision_function"):
            prob_pos = _sigmoid(self._classifier.decision_function(X))
        else:
            y_pred = self._classifier.predict(X)
            prob_pos = np.asarray(y_pred, dtype=np.float64)
        return np.stack([1.0 - prob_pos, prob_pos], axis=1)

    @property
    def best_threshold(self) -> float:
        """Decision threshold selected by ``fit_hyperparameters``."""
        return self._threshold
