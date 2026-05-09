"""
Compare the current linear SVM trajectory pipeline against small MLP heads.

The feature pipeline is intentionally unchanged:
  hidden mean(21-24) features -> train-only StandardScaler -> PCA
  trajectory features -> separate train-only StandardScaler
  concat -> classifier

This script only swaps the classifier head between:
  - linear SVM
  - 2-layer MLP with BatchNorm + Dropout
  - 3-layer MLP with BatchNorm + Dropout
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
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
from experiments.run_trajectory_range_threshold_ablation import (  # noqa: E402
    PostPcaTrajectoryBuilder,
    evaluate_scores_with_threshold,
    find_best_threshold_by_f1,
)


def _copy_trajectory_cache(output_dir: Path, token_window: int) -> None:
    target = output_dir / f"trajectory_cache_tail{token_window}_mean21_24.npz"
    for source in (
        ROOT / "results" / "trajectory_range_ablation" / f"trajectory_cache_tail{token_window}_mean21_24.npz",
        ROOT / "results" / "trajectory_features" / f"trajectory_cache_tail{token_window}_mean21_24.npz",
        ROOT / "results" / "truth_alignment_ablation" / f"trajectory_cache_tail{token_window}_mean21_24.npz",
    ):
        if not target.exists() and source.exists():
            output_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            print(f"Copied trajectory cache from {source} to {target}")
            return


class BatchNormDropoutMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_sizes: tuple[int, ...], dropout: float) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = input_dim
        for hidden in hidden_sizes:
            layers.extend(
                [
                    nn.Linear(prev, hidden),
                    nn.BatchNorm1d(hidden),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            prev = hidden
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def _parse_sizes(value: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


def _standardize_scores(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    std = scores.std()
    if std <= 1e-12:
        return scores - scores.mean()
    return (scores - scores.mean()) / std


def train_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    hidden_sizes: tuple[int, ...],
    dropout: float,
    lr: float,
    weight_decay: float,
    max_epochs: int,
    patience: int,
    batch_size: int,
    seed: int,
) -> BatchNormDropoutMLP:
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BatchNormDropoutMLP(X_train.shape[1], hidden_sizes, dropout).to(device)

    X_train_t = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32, device=device)
    X_val_t = torch.tensor(X_val, dtype=torch.float32, device=device)
    y_val_t = torch.tensor(y_val, dtype=torch.float32, device=device)

    pos = float(y_train.sum())
    neg = float(len(y_train) - y_train.sum())
    pos_weight = torch.tensor([neg / max(pos, 1.0)], dtype=torch.float32, device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_state = None
    best_loss = float("inf")
    stale = 0
    rng = np.random.default_rng(seed)

    for _ in range(max_epochs):
        model.train()
        order = rng.permutation(len(X_train))
        for start in range(0, len(order), batch_size):
            idx = torch.tensor(order[start : start + batch_size], dtype=torch.long, device=device)
            opt.zero_grad(set_to_none=True)
            logits = model(X_train_t.index_select(0, idx))
            loss = loss_fn(logits, y_train_t.index_select(0, idx))
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(model(X_val_t), y_val_t).item())
        if val_loss < best_loss - 1e-5:
            best_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model.cpu()


def mlp_scores(model: BatchNormDropoutMLP, X: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X, dtype=torch.float32)).numpy()
    return logits.astype(float)


def _mean(values: list[float]) -> float:
    valid = [v for v in values if not np.isnan(v)]
    return float(np.mean(valid)) if valid else float("nan")


def evaluate_classifier(
    classifier: str,
    X_hidden: np.ndarray,
    X_traj: np.ndarray,
    y: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray | None, np.ndarray]],
    pca_components: int,
    svm_c: float,
    mlp_hidden: tuple[int, ...],
    dropout: float,
    lr: float,
    weight_decay: float,
    max_epochs: int,
    patience: int,
    batch_size: int,
    seed: int,
    num_thresholds: int,
    use_trajectory: bool,
    predictions_dir: Path,
) -> dict:
    rows: list[dict] = []
    pred_frames: list[pd.DataFrame] = []
    t0 = time.time()

    for fold_idx, (idx_train, idx_val, idx_test) in enumerate(splits):
        if idx_val is None:
            raise ValueError("Validation split is required.")
        builder = PostPcaTrajectoryBuilder(pca_components, True, use_trajectory)
        X_train = builder.fit_transform(X_hidden[idx_train], X_traj[idx_train])
        X_val = builder.transform(X_hidden[idx_val], X_traj[idx_val])
        X_test = builder.transform(X_hidden[idx_test], X_traj[idx_test])

        if classifier == "svm_linear":
            model = SVC(C=svm_c, kernel="linear", probability=False, random_state=seed)
            model.fit(X_train, y[idx_train])
            val_scores = model.decision_function(X_val)
            test_scores = model.decision_function(X_test)
        else:
            model = train_mlp(
                X_train,
                y[idx_train],
                X_val,
                y[idx_val],
                mlp_hidden,
                dropout,
                lr,
                weight_decay,
                max_epochs,
                patience,
                batch_size,
                seed + fold_idx,
            )
            val_scores = mlp_scores(model, X_val)
            test_scores = mlp_scores(model, X_test)

        val_scores = _standardize_scores(val_scores)
        test_scores = _standardize_scores(test_scores)
        best_threshold, val_tuned = find_best_threshold_by_f1(y[idx_val], val_scores, num_thresholds)
        val_default = evaluate_scores_with_threshold(y[idx_val], val_scores, 0.0)
        test_default = evaluate_scores_with_threshold(y[idx_test], test_scores, 0.0)
        test_tuned = evaluate_scores_with_threshold(y[idx_test], test_scores, best_threshold)

        row = {
            "fold": fold_idx + 1,
            "final_feature_dim": X_train.shape[1],
            "hidden_pca_dim": builder.hidden_pca_dim,
            "trajectory_feature_dim": X_traj.shape[1] if use_trajectory else 0,
            "best_threshold": best_threshold,
            "val_roc_auc": val_default["roc_auc"],
            "test_roc_auc": test_default["roc_auc"],
        }
        for split_name, metrics in (("val", val_default), ("test", test_default)):
            for metric in ("accuracy", "precision", "recall", "f1"):
                row[f"{split_name}_{metric}_default"] = metrics[metric]
        for split_name, metrics in (("val", val_tuned), ("test", test_tuned)):
            for metric in ("accuracy", "precision", "recall", "f1"):
                row[f"{split_name}_{metric}_tuned"] = metrics[metric]
        rows.append(row)

        for split_name, indices, scores in (
            ("val", idx_val, val_scores),
            ("test", idx_test, test_scores),
        ):
            pred_frames.append(
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

    predictions_dir.mkdir(parents=True, exist_ok=True)
    pd.concat(pred_frames, ignore_index=True).to_csv(predictions_dir / f"{classifier}.csv", index=False)

    summary = {
        "classifier": classifier,
        "use_trajectory": use_trajectory,
        "mlp_hidden": ",".join(str(v) for v in mlp_hidden) if classifier.startswith("mlp") else "",
        "dropout": dropout if classifier.startswith("mlp") else 0.0,
        "pca_components": pca_components,
        "svm_c": svm_c if classifier == "svm_linear" else float("nan"),
        "final_feature_dim": _mean([r["final_feature_dim"] for r in rows]),
        "hidden_pca_dim": _mean([r["hidden_pca_dim"] for r in rows]),
        "trajectory_feature_dim": _mean([r["trajectory_feature_dim"] for r in rows]),
        "best_threshold": _mean([r["best_threshold"] for r in rows]),
        "runtime_seconds": time.time() - t0,
    }
    for split in ("val", "test"):
        for mode in ("default", "tuned"):
            for metric in ("accuracy", "precision", "recall", "f1"):
                key = f"{split}_{metric}_{mode}"
                summary[key] = _mean([r[key] for r in rows])
        summary[f"{split}_roc_auc"] = _mean([r[f"{split}_roc_auc"] for r in rows])
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "mlp_vs_svm_trajectory")
    parser.add_argument("--pca-components", type=int, default=80)
    parser.add_argument("--svm-c", type=float, default=0.1)
    parser.add_argument("--trajectory-range", default="10-24")
    parser.add_argument("--token-window", type=int, default=32)
    parser.add_argument("--num-thresholds", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--mlp2-hidden", default="128,64")
    parser.add_argument("--mlp3-hidden", default="256,128,64")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--max-epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--include-baseline-only", action="store_true")
    parser.add_argument("--max-samples-debug", type=int, default=None)
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "config.json").open("w") as f:
        json.dump({k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, f, indent=2)

    _copy_trajectory_cache(output_dir, args.token_window)
    X_hidden, layer_vectors, y, splits = load_or_extract_representations(output_dir, args.token_window)
    if args.max_samples_debug is not None:
        keep = set(range(args.max_samples_debug))
        remap = {old: new for new, old in enumerate(sorted(keep))}
        new_splits = []
        for tr, va, te in splits:
            tr_new = np.asarray([remap[int(i)] for i in tr if int(i) in keep], dtype=int)
            va_new = np.asarray([remap[int(i)] for i in va if int(i) in keep], dtype=int) if va is not None else None
            te_new = np.asarray([remap[int(i)] for i in te if int(i) in keep], dtype=int)
            if tr_new.size and va_new is not None and va_new.size and te_new.size:
                new_splits.append((tr_new, va_new, te_new))
        if not new_splits:
            raise ValueError("--max-samples-debug removed all usable folds.")
        X_hidden = X_hidden[: args.max_samples_debug]
        layer_vectors = layer_vectors[: args.max_samples_debug]
        y = y[: args.max_samples_debug]
        splits = new_splits

    X_traj, traj_names = build_trajectory_feature_matrix(layer_vectors, _parse_range(args.trajectory_range))
    print(f"Samples: {len(y)}")
    print(f"Hidden feature shape before PCA: {X_hidden.shape}")
    print(f"Trajectory range {args.trajectory_range}: {X_traj.shape}")
    for i, (tr, va, te) in enumerate(splits, 1):
        print(
            f"Fold {i}: train={len(tr)} {_class_distribution(y, tr)}  "
            f"val={len(va)} {_class_distribution(y, va)}  "
            f"test={len(te)} {_class_distribution(y, te)}"
        )
    with (output_dir / "feature_names.json").open("w") as f:
        json.dump(
            {
                "pca": [f"pca_{i}" for i in range(args.pca_components)],
                "trajectory": traj_names,
            },
            f,
            indent=2,
        )

    jobs: list[tuple[str, bool, tuple[int, ...]]] = [
        ("svm_linear", True, ()),
        ("mlp2", True, _parse_sizes(args.mlp2_hidden)),
        ("mlp3", True, _parse_sizes(args.mlp3_hidden)),
    ]
    if args.include_baseline_only:
        jobs.extend(
            [
                ("svm_linear_baseline_only", False, ()),
                ("mlp2_baseline_only", False, _parse_sizes(args.mlp2_hidden)),
                ("mlp3_baseline_only", False, _parse_sizes(args.mlp3_hidden)),
            ]
        )

    rows = []
    for classifier, use_traj, hidden in jobs:
        base_name = classifier.replace("_baseline_only", "")
        print(f"\nRunning {classifier}: use_trajectory={use_traj}, hidden={hidden or 'n/a'}")
        summary = evaluate_classifier(
            base_name,
            X_hidden,
            X_traj,
            y,
            splits,
            args.pca_components,
            args.svm_c,
            hidden,
            args.dropout,
            args.lr,
            args.weight_decay,
            args.max_epochs,
            args.patience,
            args.batch_size,
            args.seed,
            args.num_thresholds,
            use_traj,
            output_dir / "predictions" / classifier,
        )
        summary["experiment"] = classifier
        summary["trajectory_range"] = args.trajectory_range if use_traj else "none"
        rows.append(summary)
        pd.DataFrame(rows).to_csv(output_dir / "summary.csv", index=False)
        print(
            f"  val_f1_tuned={summary['val_f1_tuned']:.4f}  "
            f"test_f1_tuned={summary['test_f1_tuned']:.4f}  "
            f"val_auc={summary['val_roc_auc']:.4f}  "
            f"test_auc={summary['test_roc_auc']:.4f}"
        )

    results = pd.DataFrame(rows).sort_values("val_f1_tuned", ascending=False)
    results.to_csv(output_dir / "summary.csv", index=False)
    results.to_json(output_dir / "summary.json", orient="records", indent=2)
    print("\nMLP vs SVM trajectory summary")
    cols = [
        "experiment",
        "trajectory_range",
        "final_feature_dim",
        "val_f1_tuned",
        "test_f1_tuned",
        "val_roc_auc",
        "test_roc_auc",
        "best_threshold",
        "runtime_seconds",
    ]
    print(results[cols].to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    svm = results.loc[results["experiment"] == "svm_linear"].iloc[0]
    for name in ("mlp2", "mlp3"):
        if name in set(results["experiment"]):
            row = results.loc[results["experiment"] == name].iloc[0]
            print(
                f"Delta {name} - svm_linear: "
                f"val_f1={row['val_f1_tuned'] - svm['val_f1_tuned']:.4f}, "
                f"test_f1={row['test_f1_tuned'] - svm['test_f1_tuned']:.4f}, "
                f"test_auc={row['test_roc_auc'] - svm['test_roc_auc']:.4f}"
            )
    print(f"Saved outputs under: {output_dir}")


if __name__ == "__main__":
    main()
