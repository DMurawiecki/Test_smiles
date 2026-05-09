"""
Trajectory range ablation with validation-tuned SVM thresholds.

This experiment keeps the existing split protocol and the current baseline
representation fixed: mean(21-24) hidden features -> scaler -> PCA, trajectory
features -> separate scaler, concatenate, then linear SVM.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_trajectory_features import (  # noqa: E402
    _class_distribution,
    _parse_range,
    build_trajectory_feature_matrix,
    load_or_extract_representations,
)


DEFAULT_RANGES = (
    "6-24",
    "8-24",
    "10-24",
    "12-24",
    "14-24",
    "16-24",
    "10-18",
    "10-20",
    "10-22",
    "12-20",
    "12-22",
    "12-24",
)

TRAJECTORY_GROUPS = {
    "step": {
        "step_mean",
        "step_std",
        "step_max",
        "step_min",
        "step_median",
        "step_last",
        "step_early_mean",
        "step_mid_mean",
        "step_late_mean",
        "early_late_step_ratio",
    },
    "path": {"path_length", "endpoint_distance", "straightness"},
    "cosine": {
        "cos_step_mean",
        "cos_step_std",
        "cos_step_min",
        "cos_step_max",
        "cos_first_last",
        "cos_early_mean",
        "cos_mid_mean",
        "cos_late_mean",
    },
    "late_stability": {
        "late_l2_to_final_mean",
        "late_l2_to_final_max",
        "late_cos_to_final_mean",
        "late_cos_to_final_min",
    },
    "curvature": {
        "delta_cos_mean",
        "delta_cos_std",
        "delta_cos_min",
        "delta_cos_max",
        "curvature_mean",
        "curvature_max",
        "early_curvature_mean",
        "mid_curvature_mean",
        "late_curvature_mean",
    },
}

GROUP_ABLATIONS = (
    ("all", ("step", "path", "cosine", "late_stability", "curvature")),
    ("step_only", ("step",)),
    ("path_only", ("path",)),
    ("cosine_only", ("cosine",)),
    ("late_stability_only", ("late_stability",)),
    ("curvature_only", ("curvature",)),
    ("step_plus_cosine", ("step", "cosine")),
    ("step_plus_late_stability", ("step", "late_stability")),
    ("cosine_plus_curvature", ("cosine", "curvature")),
    (
        "step_plus_cosine_plus_late_stability_plus_curvature",
        ("step", "cosine", "late_stability", "curvature"),
    ),
)


@dataclass
class FoldArtifact:
    fold: int
    model: SVC
    X_val: np.ndarray
    y_val: np.ndarray
    threshold: float
    feature_names: list[str]


def evaluate_scores_with_threshold(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    """Evaluate thresholded predictions while keeping ROC-AUC score-based."""
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)
    y_pred = (scores >= threshold).astype(int)
    try:
        roc_auc = roc_auc_score(y_true, scores)
    except ValueError:
        roc_auc = float("nan")
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc),
        "threshold": float(threshold),
    }


def find_best_threshold_by_f1(
    y_true: np.ndarray,
    scores: np.ndarray,
    num_thresholds: int = 1000,
) -> tuple[float, dict[str, float]]:
    """Choose validation threshold by F1, breaking ties by precision, then accuracy."""
    scores = np.asarray(scores, dtype=float)
    finite = scores[np.isfinite(scores)]
    if finite.size == 0:
        candidates = np.asarray([0.0])
    elif np.unique(finite).size == 1:
        value = float(finite[0])
        candidates = np.asarray([value, 0.0])
    else:
        candidates = np.linspace(float(finite.min()), float(finite.max()), num_thresholds)
        candidates = np.unique(np.concatenate([candidates, np.asarray([0.0])]))

    best_threshold = 0.0
    best_metrics: dict[str, float] | None = None
    best_key = (-1.0, -1.0, -1.0)
    for threshold in candidates:
        metrics = evaluate_scores_with_threshold(y_true, scores, float(threshold))
        key = (metrics["f1"], metrics["precision"], metrics["accuracy"])
        if key > best_key:
            best_key = key
            best_threshold = float(threshold)
            best_metrics = metrics
    assert best_metrics is not None
    return best_threshold, best_metrics


class PostPcaTrajectoryBuilder:
    """Train-only scaler/PCA for hidden features and separate trajectory scaler."""

    def __init__(self, pca_components: int, use_hidden: bool, use_trajectory: bool) -> None:
        self.pca_components = pca_components
        self.use_hidden = use_hidden
        self.use_trajectory = use_trajectory
        self.hidden_scaler = StandardScaler()
        self.pca: PCA | None = None
        self.traj_scaler = StandardScaler()

    def fit_transform(self, X_hidden: np.ndarray, X_traj: np.ndarray) -> np.ndarray:
        parts: list[np.ndarray] = []
        if self.use_hidden:
            hidden_scaled = self.hidden_scaler.fit_transform(X_hidden)
            n_components = min(self.pca_components, hidden_scaled.shape[0] - 1, hidden_scaled.shape[1])
            self.pca = PCA(n_components=n_components, random_state=42)
            parts.append(self.pca.fit_transform(hidden_scaled))
        if self.use_trajectory:
            parts.append(self.traj_scaler.fit_transform(X_traj))
        return np.concatenate(parts, axis=1).astype(np.float32)

    def transform(self, X_hidden: np.ndarray, X_traj: np.ndarray) -> np.ndarray:
        parts: list[np.ndarray] = []
        if self.use_hidden:
            if self.pca is None:
                raise RuntimeError("PCA was not fitted.")
            parts.append(self.pca.transform(self.hidden_scaler.transform(X_hidden)))
        if self.use_trajectory:
            parts.append(self.traj_scaler.transform(X_traj))
        return np.concatenate(parts, axis=1).astype(np.float32)

    @property
    def hidden_pca_dim(self) -> int:
        return int(self.pca.n_components_) if self.pca is not None else 0


def _safe_matrix(X: np.ndarray, name: str) -> np.ndarray:
    if not np.isfinite(X).all():
        print(f"WARNING: non-finite values found in {name}; replacing with zeros.")
        return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X


def _feature_names(hidden_pca_dim: int, trajectory_names: list[str], use_traj: bool) -> list[str]:
    names = [f"pca_{i}" for i in range(hidden_pca_dim)]
    if use_traj:
        names.extend(trajectory_names)
    return names


def _aggregate_fold_metrics(fold_rows: list[dict]) -> dict[str, float]:
    out: dict[str, float] = {}
    for split in ("val", "test"):
        for threshold_mode in ("default", "tuned"):
            for metric in ("accuracy", "precision", "recall", "f1"):
                key = f"{split}_{metric}_{threshold_mode}"
                out[key] = float(np.mean([row[key] for row in fold_rows]))
        auc_key = f"{split}_roc_auc"
        out[auc_key] = float(np.mean([row[auc_key] for row in fold_rows]))
    out["best_threshold"] = float(np.mean([row["best_threshold"] for row in fold_rows]))
    out["final_feature_dim"] = float(np.mean([row["final_feature_dim"] for row in fold_rows]))
    return out


def evaluate_range_experiment(
    experiment: str,
    trajectory_range: str,
    X_hidden: np.ndarray,
    X_traj: np.ndarray,
    trajectory_names: list[str],
    y: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray | None, np.ndarray]],
    pca_components: int,
    svm_c: float,
    seed: int,
    num_thresholds: int,
    predictions_dir: Path | None,
    use_hidden: bool = True,
    use_trajectory: bool = True,
) -> tuple[dict, list[FoldArtifact]]:
    fold_rows: list[dict] = []
    artifacts: list[FoldArtifact] = []
    prediction_frames: list[pd.DataFrame] = []
    t0 = time.time()

    print(f"\n{experiment}")
    print(f"  hidden feature shape before PCA: {X_hidden.shape if use_hidden else (len(y), 0)}")
    print(f"  trajectory range: {trajectory_range}")
    print(f"  trajectory feature shape: {X_traj.shape if use_trajectory else (len(y), 0)}")

    for fold_idx, (idx_train, idx_val, idx_test) in enumerate(splits):
        if idx_val is None:
            raise ValueError("This experiment requires a validation split.")

        print(
            f"  Fold {fold_idx + 1}: train={len(idx_train)} {_class_distribution(y, idx_train)}  "
            f"val={len(idx_val)} {_class_distribution(y, idx_val)}  "
            f"test={len(idx_test)} {_class_distribution(y, idx_test)}"
        )
        builder = PostPcaTrajectoryBuilder(pca_components, use_hidden, use_trajectory)
        X_train = builder.fit_transform(X_hidden[idx_train], X_traj[idx_train])
        X_val = builder.transform(X_hidden[idx_val], X_traj[idx_val])
        X_test = builder.transform(X_hidden[idx_test], X_traj[idx_test])
        names = _feature_names(builder.hidden_pca_dim, trajectory_names, use_trajectory)

        model = SVC(C=svm_c, kernel="linear", probability=False, random_state=seed)
        model.fit(X_train, y[idx_train])

        val_scores = model.decision_function(X_val)
        test_scores = model.decision_function(X_test)
        _, val_default = 0.0, evaluate_scores_with_threshold(y[idx_val], val_scores, 0.0)
        test_default = evaluate_scores_with_threshold(y[idx_test], test_scores, 0.0)
        best_threshold, val_tuned = find_best_threshold_by_f1(
            y[idx_val],
            val_scores,
            num_thresholds=num_thresholds,
        )
        test_tuned = evaluate_scores_with_threshold(y[idx_test], test_scores, best_threshold)

        row = {
            "fold": fold_idx + 1,
            "best_threshold": best_threshold,
            "final_feature_dim": X_train.shape[1],
            "hidden_pca_dim": builder.hidden_pca_dim,
            "trajectory_feature_dim": X_traj.shape[1] if use_trajectory else 0,
            "val_roc_auc": val_default["roc_auc"],
            "test_roc_auc": test_default["roc_auc"],
        }
        for split_name, metrics in (
            ("val", val_default),
            ("test", test_default),
        ):
            for metric in ("accuracy", "precision", "recall", "f1"):
                row[f"{split_name}_{metric}_default"] = metrics[metric]
        for split_name, metrics in (
            ("val", val_tuned),
            ("test", test_tuned),
        ):
            for metric in ("accuracy", "precision", "recall", "f1"):
                row[f"{split_name}_{metric}_tuned"] = metrics[metric]
        fold_rows.append(row)
        artifacts.append(FoldArtifact(fold_idx + 1, model, X_val, y[idx_val], best_threshold, names))

        for split_name, indices, scores in (
            ("val", idx_val, val_scores),
            ("test", idx_test, test_scores),
        ):
            prediction_frames.append(
                pd.DataFrame(
                    {
                        "fold": fold_idx + 1,
                        "sample_id": indices,
                        "y_true": y[indices],
                        "decision_score": scores,
                        "y_pred_default": (scores >= 0.0).astype(int),
                        "y_pred_tuned": (scores >= best_threshold).astype(int),
                        "threshold_default": 0.0,
                        "threshold_tuned": best_threshold,
                        "split": split_name,
                    }
                )
            )

        print(
            f"    hidden after PCA: {(X_train.shape[0], builder.hidden_pca_dim)}  "
            f"final feature shape: {X_train.shape}"
        )

    if predictions_dir is not None:
        predictions_dir.mkdir(parents=True, exist_ok=True)
        pd.concat(prediction_frames, ignore_index=True).to_csv(
            predictions_dir / f"{experiment}.csv",
            index=False,
        )

    summary = {
        "experiment": experiment,
        "trajectory_range": trajectory_range,
        "pca_components": pca_components if use_hidden else 0,
        "svm_c": svm_c,
        "trajectory_feature_dim": X_traj.shape[1] if use_trajectory else 0,
        "runtime_seconds": time.time() - t0,
        **_aggregate_fold_metrics(fold_rows),
    }
    print(
        f"  default val_f1={summary['val_f1_default']:.4f}  "
        f"default test_f1={summary['test_f1_default']:.4f}  "
        f"best_threshold={summary['best_threshold']:.4f}  "
        f"tuned val_f1={summary['val_f1_tuned']:.4f}  "
        f"tuned test_f1={summary['test_f1_tuned']:.4f}  "
        f"val_roc_auc={summary['val_roc_auc']:.4f}  "
        f"test_roc_auc={summary['test_roc_auc']:.4f}"
    )
    return summary, artifacts


def permutation_importance_with_threshold(
    model: SVC,
    X: np.ndarray,
    y: np.ndarray,
    threshold: float,
    feature_names: list[str],
    scoring: str = "f1",
    n_repeats: int = 20,
    random_state: int = 42,
) -> pd.DataFrame:
    """Column permutation importance using a fixed tuned threshold."""
    if scoring != "f1":
        raise ValueError("Only scoring='f1' is supported.")

    rng = np.random.default_rng(random_state)
    base_scores = model.decision_function(X)
    base_pred = (base_scores >= threshold).astype(int)
    base_f1 = float(f1_score(y, base_pred, zero_division=0))
    rows: list[dict] = []

    for feature_index, feature_name in enumerate(feature_names):
        permuted_f1: list[float] = []
        for _ in range(n_repeats):
            X_perm = X.copy()
            order = rng.permutation(X_perm.shape[0])
            X_perm[:, feature_index] = X_perm[order, feature_index]
            scores = model.decision_function(X_perm)
            pred = (scores >= threshold).astype(int)
            permuted_f1.append(float(f1_score(y, pred, zero_division=0)))

        scores_arr = np.asarray(permuted_f1, dtype=float)
        rows.append(
            {
                "feature_name": feature_name,
                "feature_index": feature_index,
                "importance_mean": float(base_f1 - scores_arr.mean()),
                "importance_std": float(scores_arr.std()),
                "base_f1": base_f1,
                "permuted_f1_mean": float(scores_arr.mean()),
                "permuted_f1_std": float(scores_arr.std()),
                "feature_group": "hidden_pca" if feature_name.startswith("pca_") else "trajectory",
            }
        )
    return pd.DataFrame(rows).sort_values("importance_mean", ascending=False)


def _run_permutation_importance(
    artifacts: list[FoldArtifact],
    output_dir: Path,
    n_repeats: int,
    seed: int,
) -> pd.DataFrame:
    frames = []
    for artifact in artifacts:
        frame = permutation_importance_with_threshold(
            artifact.model,
            artifact.X_val,
            artifact.y_val,
            artifact.threshold,
            artifact.feature_names,
            n_repeats=n_repeats,
            random_state=seed + artifact.fold,
        )
        frame["fold"] = artifact.fold
        frames.append(frame)

    raw = pd.concat(frames, ignore_index=True)
    agg = (
        raw.groupby(["feature_name", "feature_index", "feature_group"], as_index=False)
        .agg(
            importance_mean=("importance_mean", "mean"),
            importance_std=("importance_mean", "std"),
            base_f1=("base_f1", "mean"),
            permuted_f1_mean=("permuted_f1_mean", "mean"),
            permuted_f1_std=("permuted_f1_std", "mean"),
        )
        .sort_values("importance_mean", ascending=False)
    )
    agg["importance_std"] = agg["importance_std"].fillna(0.0)
    agg.to_csv(output_dir / "permutation_importance_best_val.csv", index=False)
    traj = agg.loc[agg["feature_group"] == "trajectory"].copy()
    traj.to_csv(output_dir / "permutation_importance_trajectory_only.csv", index=False)
    return traj


def _trajectory_columns_for_groups(names: list[str], groups: tuple[str, ...]) -> list[int]:
    allowed = set().union(*(TRAJECTORY_GROUPS[group] for group in groups))
    indices = []
    for idx, name in enumerate(names):
        suffix = name.split("__", 1)[1] if "__" in name else name
        if suffix in allowed:
            indices.append(idx)
    return indices


def _apply_debug_subset(
    X_hidden: np.ndarray,
    layer_vectors: np.ndarray,
    y: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray | None, np.ndarray]],
    max_samples: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[tuple[np.ndarray, np.ndarray | None, np.ndarray]]]:
    if max_samples is None or max_samples >= len(y):
        return X_hidden, layer_vectors, y, splits

    keep = set(range(max_samples))
    remap = {old: new for new, old in enumerate(sorted(keep))}
    new_splits = []
    for tr, va, te in splits:
        tr_new = np.asarray([remap[i] for i in tr if int(i) in keep], dtype=int)
        va_new = np.asarray([remap[i] for i in va if int(i) in keep], dtype=int) if va is not None else None
        te_new = np.asarray([remap[i] for i in te if int(i) in keep], dtype=int)
        if tr_new.size and va_new is not None and va_new.size and te_new.size:
            new_splits.append((tr_new, va_new, te_new))
    if not new_splits:
        raise ValueError("--max-samples-debug removed all usable folds.")
    print(f"DEBUG subset active: using first {max_samples} samples and {len(new_splits)} filtered folds.")
    return X_hidden[:max_samples], layer_vectors[:max_samples], y[:max_samples], new_splits


def _maybe_copy_trajectory_cache(output_dir: Path, token_window: int) -> None:
    current = output_dir / f"trajectory_cache_tail{token_window}_mean21_24.npz"
    legacy = ROOT / "results" / "trajectory_features" / f"trajectory_cache_tail{token_window}_mean21_24.npz"
    if not current.exists() and legacy.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(legacy, current)
        print(f"Copied existing trajectory cache from {legacy} to {current}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pca-components", type=int, default=80)
    parser.add_argument("--svm-c", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "trajectory_range_ablation")
    parser.add_argument("--trajectory-ranges", nargs="+", default=list(DEFAULT_RANGES))
    parser.add_argument("--num-thresholds", type=int, default=1000)
    parser.add_argument("--run-permutation-importance", action="store_true")
    parser.add_argument("--importance-range", default=None)
    parser.add_argument("--importance-repeats", type=int, default=20)
    parser.add_argument("--run-group-ablation", action="store_true")
    parser.add_argument("--max-samples-debug", type=int, default=None)
    parser.add_argument("--token-window", type=int, default=32)
    args = parser.parse_args()

    np.random.seed(args.seed)
    output_dir: Path = args.output_dir
    predictions_dir = output_dir / "predictions"
    output_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "pca_components": args.pca_components,
        "svm_c": args.svm_c,
        "seed": args.seed,
        "trajectory_ranges": args.trajectory_ranges,
        "num_thresholds": args.num_thresholds,
        "importance_range": args.importance_range,
        "importance_repeats": args.importance_repeats,
        "run_group_ablation": args.run_group_ablation,
        "max_samples_debug": args.max_samples_debug,
        "token_window": args.token_window,
        "layer_mapping": "cached index 0 = transformer layer 1; hidden_states[1] = transformer layer 1 when embeddings are present",
    }
    with (output_dir / "config.json").open("w") as f:
        json.dump(config, f, indent=2)

    _maybe_copy_trajectory_cache(output_dir, args.token_window)
    X_hidden, layer_vectors, y, splits = load_or_extract_representations(output_dir, args.token_window)
    X_hidden, layer_vectors, y, splits = _apply_debug_subset(
        X_hidden,
        layer_vectors,
        y,
        splits,
        args.max_samples_debug,
    )
    print(f"Samples: {len(y)}")
    print(f"Hidden feature shape before PCA: {X_hidden.shape}")
    print(f"Layer vector tensor shape: {layer_vectors.shape}")
    print(config["layer_mapping"])

    range_to_matrix: dict[str, np.ndarray] = {}
    range_to_names: dict[str, list[str]] = {}
    unique_ranges = list(
        dict.fromkeys(
            [
                *args.trajectory_ranges,
                "10-24",
                *([] if args.importance_range is None else [args.importance_range]),
            ]
        )
    )
    for range_label in unique_ranges:
        X_traj, names = build_trajectory_feature_matrix(layer_vectors, _parse_range(range_label))
        X_traj = _safe_matrix(X_traj, f"trajectory range {range_label}")
        range_to_matrix[range_label] = X_traj
        range_to_names[range_label] = names
        print(f"Prepared trajectory range {range_label}: {X_traj.shape}")
    with (output_dir / "feature_names.json").open("w") as f:
        json.dump(
            {
                "baseline": [f"pca_{i}" for i in range(args.pca_components)],
                "trajectory": range_to_names,
            },
            f,
            indent=2,
        )

    zero_traj = np.zeros((len(y), 0), dtype=np.float32)
    summaries: list[dict] = []
    artifacts_by_range: dict[str, list[FoldArtifact]] = {}

    baseline_summary, baseline_artifacts = evaluate_range_experiment(
        "E0_baseline_only",
        "none",
        X_hidden,
        zero_traj,
        [],
        y,
        splits,
        args.pca_components,
        args.svm_c,
        args.seed,
        args.num_thresholds,
        predictions_dir,
        use_hidden=True,
        use_trajectory=False,
    )
    summaries.append(baseline_summary)
    artifacts_by_range["none"] = baseline_artifacts

    for range_label in args.trajectory_ranges:
        experiment = f"E_range_{range_label.replace('-', '_')}"
        summary, artifacts = evaluate_range_experiment(
            experiment,
            range_label,
            X_hidden,
            range_to_matrix[range_label],
            range_to_names[range_label],
            y,
            splits,
            args.pca_components,
            args.svm_c,
            args.seed,
            args.num_thresholds,
            predictions_dir,
            use_hidden=True,
            use_trajectory=True,
        )
        summaries.append(summary)
        artifacts_by_range[range_label] = artifacts

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(output_dir / "summary.csv", index=False)
    with (output_dir / "summary.json").open("w") as f:
        json.dump({"summary": summaries}, f, indent=2)

    importance_range = args.importance_range
    if importance_range is None:
        non_baseline = summary_df.loc[summary_df["trajectory_range"] != "none"]
        importance_range = str(
            non_baseline.sort_values("val_f1_tuned", ascending=False).iloc[0]["trajectory_range"]
        )

    if args.run_permutation_importance:
        print(f"\nPermutation importance range: {importance_range}")
        traj_importance = _run_permutation_importance(
            artifacts_by_range[importance_range],
            output_dir,
            args.importance_repeats,
            args.seed,
        )
        print("\nTop-20 trajectory features by validation tuned-threshold F1 importance")
        print(traj_importance.head(20).to_string(index=False, float_format=lambda value: f"{value:.4f}"))

    if args.run_group_ablation:
        group_rows = []
        base_range = importance_range or "10-24"
        names = range_to_names[base_range]
        X_all = range_to_matrix[base_range]
        for group_name, groups in GROUP_ABLATIONS:
            indices = _trajectory_columns_for_groups(names, groups)
            selected_names = [names[i] for i in indices]
            selected = X_all[:, indices]
            summary, _ = evaluate_range_experiment(
                f"group_{group_name}",
                base_range,
                X_hidden,
                selected,
                selected_names,
                y,
                splits,
                args.pca_components,
                args.svm_c,
                args.seed,
                args.num_thresholds,
                predictions_dir=None,
                use_hidden=True,
                use_trajectory=True,
            )
            summary["trajectory_groups"] = "+".join(groups)
            group_rows.append(summary)
        pd.DataFrame(group_rows).to_csv(output_dir / "group_ablation.csv", index=False)

    print("\nTrajectory range threshold ablation summary")
    display_cols = [
        "experiment",
        "trajectory_range",
        "final_feature_dim",
        "val_f1_default",
        "test_f1_default",
        "best_threshold",
        "val_f1_tuned",
        "test_f1_tuned",
        "val_roc_auc",
        "test_roc_auc",
    ]
    print(summary_df[display_cols].to_string(index=False, float_format=lambda value: f"{value:.4f}"))

    best_val = summary_df.sort_values("val_f1_tuned", ascending=False).iloc[0]
    best_test = summary_df.sort_values("test_f1_tuned", ascending=False).iloc[0]
    e0 = summary_df.loc[summary_df["experiment"] == "E0_baseline_only"].iloc[0]
    print(f"\nBest by val_f1_tuned: {best_val['experiment']} ({best_val['val_f1_tuned']:.4f})")
    print(f"Best by test_f1_tuned: {best_test['experiment']} ({best_test['test_f1_tuned']:.4f})")
    print(
        "Delta vs E0 baseline: "
        f"val_f1_tuned_delta={best_val['val_f1_tuned'] - e0['val_f1_tuned']:.4f}, "
        f"test_f1_tuned_delta={best_val['test_f1_tuned'] - e0['test_f1_tuned']:.4f}, "
        f"test_roc_auc_delta={best_val['test_roc_auc'] - e0['test_roc_auc']:.4f}"
    )
    print(
        f"The best trajectory range by validation tuned F1 is {best_val['trajectory_range']}."
    )
    print(
        "Compared to the baseline, it changes test F1 by "
        f"{best_val['test_f1_tuned'] - e0['test_f1_tuned']:.4f}."
    )
    if args.run_permutation_importance:
        print("The top trajectory features by permutation importance are printed above.")
    print(f"Saved outputs under: {output_dir}")


if __name__ == "__main__":
    main()
