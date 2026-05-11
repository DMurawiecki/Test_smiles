"""
Prompt-answer geometric compatibility ablation.

The baseline is the shared answer stack:
concat(mean(21-24) over answer last-token/last8/last16/last32).  Prompt-answer
geometry features are added as separate blocks so that the only experimental
difference is which extra block is enabled.  Scalers/PCA are fit on train only.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import aggregation  # noqa: E402
from experiments.run_trajectory_features import (  # noqa: E402
    DATA_FILE,
    DEFAULT_POOLING_MODES,
    N_TRANSFORMER_LAYERS,
    POOLING_MODE_SHORT,
    _class_distribution,
    _real_token_positions,
    _select_response_positions,
)
from experiments.run_trajectory_range_threshold_ablation import find_best_threshold_by_f1  # noqa: E402
from model import MAX_LENGTH, _DEFAULT_MODEL, get_model_and_tokenizer  # noqa: E402
from splitting import split_data  # noqa: E402


BATCH_SIZE = 4
EPS = 1e-8
POOLING_SUFFIX = {
    "response_last_token": "last",
    "response_last_8_mean": "last8",
    "response_last_16_mean": "last16",
    "response_last_32_mean": "last32",
}
SCALAR_METRICS = (
    "cos",
    "cos_dist",
    "l1",
    "l2",
    "linf",
    "dot",
    "prompt_norm",
    "answer_norm",
    "delta_norm",
    "norm_ratio",
    "abs_norm_diff",
)
TRAJECTORY_STATS = ("mean", "std", "min", "max", "first", "last", "last_minus_first", "slope", "auc")


@dataclass
class FoldArtifact:
    experiment_name: str
    fold: int
    model: SVC
    X_eval: np.ndarray
    y_eval: np.ndarray
    feature_names: list[str]
    threshold: float


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _safe_name(value: str) -> str:
    return value.lower().replace("/", "_").replace(".", "p").replace("-", "_")


def _prompt_answer_positions(tokenizer, prompt: str, full_text: str, attention_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    real_positions = _real_token_positions(attention_mask)
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
    answer_start = min(len(prompt_ids), max(len(full_ids) - 1, 0), int(real_positions[-1].item()))
    answer_end = min(len(full_ids), int(real_positions[-1].item()) + 1)
    answer_positions = real_positions[-1:] if answer_end <= answer_start else torch.arange(answer_start, answer_end, dtype=torch.long)
    prompt_positions = real_positions[real_positions < answer_start]
    if prompt_positions.numel() == 0:
        prompt_positions = real_positions[:1]
    if torch.isin(prompt_positions, answer_positions).any():
        raise ValueError("Prompt and answer token positions overlap.")
    return prompt_positions, answer_positions


def _select_tail_positions(positions: torch.Tensor, mode: str, token_window: int) -> torch.Tensor:
    if mode == "response_last_token":
        width = 1
    elif mode == "response_last_8_mean":
        width = 8
    elif mode == "response_last_16_mean":
        width = 16
    elif mode == "response_last_32_mean":
        width = 32
    else:
        width = token_window
    return positions[-min(width, positions.numel()) :]


def _pool_layers(hidden_states: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    hs = aggregation._as_layer_tensor(hidden_states).detach()
    positions = positions.to(device=hs.device)
    rows = []
    for layer_number in range(1, N_TRANSFORMER_LAYERS + 1):
        idx = aggregation.transformer_layer_to_index(layer_number, hs)
        rows.append(hs[idx].index_select(0, positions).float().mean(dim=0))
    return torch.stack(rows, dim=0).to(dtype=torch.float32)


def _cache_path(output_dir: Path, model_name: str, pooling_modes: list[str], token_window: int) -> Path:
    slug = "_".join(POOLING_MODE_SHORT.get(mode, mode).replace("response_", "") for mode in pooling_modes)
    return output_dir / f"prompt_answer_geometry_{_safe_name(model_name)}_tail{token_window}_{slug}.npz"


def load_or_extract_prompt_answer_vectors(
    output_dir: Path,
    model_name: str,
    pooling_modes: list[str],
    token_window: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray, list]:
    output_dir.mkdir(parents=True, exist_ok=True)
    cache = _cache_path(output_dir, model_name, pooling_modes, token_window)
    if cache.exists():
        data = np.load(cache, allow_pickle=False)
        answer = {mode: np.asarray(data[f"answer__{mode}"], dtype=np.float32) for mode in pooling_modes}
        prompt = {mode: np.asarray(data[f"prompt__{mode}"], dtype=np.float32) for mode in pooling_modes}
        y = np.asarray(data["y"], dtype=int)
        splits = []
        for i in range(int(data["n_splits"][0])):
            splits.append((
                np.asarray(data[f"split_{i}_train"], dtype=int),
                np.asarray(data[f"split_{i}_val"], dtype=int),
                np.asarray(data[f"split_{i}_test"], dtype=int),
            ))
        print(f"Loaded prompt-answer geometry cache: {cache}")
        return answer, prompt, y, splits

    df = pd.read_csv(DATA_FILE)
    for col in ("prompt", "response", "label"):
        if col not in df.columns:
            raise ValueError(f"Missing required column {col!r}; available columns: {df.columns.tolist()}")
    prompts = df["prompt"].astype(str).tolist()
    answers_text = df["response"].astype(str).tolist()
    texts = [f"{p}{a}" for p, a in zip(prompts, answers_text)]
    y = df["label"].astype(float).astype(int).to_numpy()
    splits = split_data(y, df)

    device = _device()
    print(f"Cache not found. Extracting prompt/answer hidden states on {device}.")
    print(f"Model: {model_name}")
    print(f"Pooling modes: {', '.join(pooling_modes)}")
    model, tokenizer = get_model_and_tokenizer(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.to(device)

    answer_layers: dict[str, list[torch.Tensor]] = {mode: [] for mode in pooling_modes}
    prompt_layers: dict[str, list[torch.Tensor]] = {mode: [] for mode in pooling_modes}
    for start in tqdm(range(0, len(texts), BATCH_SIZE), desc="Extracting prompt-answer vectors", unit="batch"):
        batch = texts[start : start + BATCH_SIZE]
        enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LENGTH)
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        hidden = torch.stack(outputs.hidden_states, dim=1).float()
        masks = attention_mask.cpu()
        for i in range(hidden.size(0)):
            p_all, a_all = _prompt_answer_positions(tokenizer, prompts[start + i], texts[start + i], masks[i])
            for mode in pooling_modes:
                p_pos = _select_tail_positions(p_all, mode, token_window)
                a_pos = _select_response_positions(a_all, mode, token_window)
                prompt_layers[mode].append(_pool_layers(hidden[i], p_pos).cpu())
                answer_layers[mode].append(_pool_layers(hidden[i], a_pos).cpu())

    answer = {mode: np.stack([x.numpy() for x in values]).astype(np.float32) for mode, values in answer_layers.items()}
    prompt = {mode: np.stack([x.numpy() for x in values]).astype(np.float32) for mode, values in prompt_layers.items()}
    payload: dict[str, object] = {
        "y": y,
        "n_splits": np.asarray([len(splits)], dtype=np.int64),
        "metadata": np.asarray([json.dumps({"model_name": model_name, "pooling_modes": pooling_modes})]),
    }
    for mode in pooling_modes:
        payload[f"answer__{mode}"] = answer[mode]
        payload[f"prompt__{mode}"] = prompt[mode]
    for i, (tr, va, te) in enumerate(splits):
        payload[f"split_{i}_train"] = tr
        payload[f"split_{i}_val"] = np.asarray([], dtype=np.int64) if va is None else va
        payload[f"split_{i}_test"] = te
    np.savez_compressed(cache, **payload)
    print(f"Saved prompt-answer geometry cache: {cache}")
    return answer, prompt, y, splits


def _stack_mean_layers(vectors: dict[str, np.ndarray], pooling_modes: list[str], layers: list[int]) -> np.ndarray:
    parts = [vectors[mode][:, [layer - 1 for layer in layers], :].mean(axis=1) for mode in pooling_modes]
    return np.concatenate(parts, axis=1).astype(np.float32)


def build_answer_stack(answer: dict[str, np.ndarray], pooling_modes: list[str], layers: list[int]) -> tuple[np.ndarray, list[str]]:
    X = _stack_mean_layers(answer, pooling_modes, layers)
    names = [f"answer_stack_{POOLING_SUFFIX[mode]}_{i}" for mode in pooling_modes for i in range(answer[mode].shape[-1])]
    return X, names


def build_prompt_stack(prompt: dict[str, np.ndarray], pooling_modes: list[str], layers: list[int]) -> tuple[np.ndarray, list[str]]:
    X = _stack_mean_layers(prompt, pooling_modes, layers)
    names = [f"prompt_stack_{POOLING_SUFFIX[mode]}_{i}" for mode in pooling_modes for i in range(prompt[mode].shape[-1])]
    return X, names


def build_delta_stack(answer: dict[str, np.ndarray], prompt: dict[str, np.ndarray], pooling_modes: list[str], layers: list[int]) -> tuple[np.ndarray, list[str]]:
    delta = {mode: answer[mode] - prompt[mode] for mode in pooling_modes}
    X = _stack_mean_layers(delta, pooling_modes, layers)
    names = [f"delta_stack_{POOLING_SUFFIX[mode]}_{i}" for mode in pooling_modes for i in range(answer[mode].shape[-1])]
    return X, names


def _safe_cosine_rows(p: np.ndarray, a: np.ndarray) -> np.ndarray:
    return np.sum(p * a, axis=1) / (np.linalg.norm(p, axis=1) * np.linalg.norm(a, axis=1) + EPS)


def _scalar_metric_arrays(p: np.ndarray, a: np.ndarray) -> dict[str, np.ndarray]:
    p_norm = np.linalg.norm(p, axis=1)
    a_norm = np.linalg.norm(a, axis=1)
    delta = a - p
    l2 = np.linalg.norm(delta, axis=1)
    cos = _safe_cosine_rows(p, a)
    return {
        "cos": cos,
        "cos_dist": 1.0 - cos,
        "l1": np.linalg.norm(delta, ord=1, axis=1),
        "l2": l2,
        "linf": np.linalg.norm(delta, ord=np.inf, axis=1),
        "dot": np.sum(p * a, axis=1),
        "prompt_norm": p_norm,
        "answer_norm": a_norm,
        "delta_norm": l2,
        "norm_ratio": a_norm / (p_norm + EPS),
        "abs_norm_diff": np.abs(a_norm - p_norm),
    }


def build_pa_scalar_features(
    answer: dict[str, np.ndarray],
    prompt: dict[str, np.ndarray],
    pooling_modes: list[str],
    layers: list[int],
) -> tuple[np.ndarray, list[str]]:
    columns, names = [], []
    for layer in layers:
        for mode in pooling_modes:
            metrics = _scalar_metric_arrays(prompt[mode][:, layer - 1, :], answer[mode][:, layer - 1, :])
            suffix = POOLING_SUFFIX[mode]
            for metric in SCALAR_METRICS:
                columns.append(metrics[metric])
                names.append(f"pa_{metric}_layer{layer}_{suffix}")
    X = np.vstack(columns).T.astype(np.float32)
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0), names


def _slope(values: np.ndarray) -> np.ndarray:
    n = values.shape[1]
    if n < 2:
        return np.zeros(values.shape[0], dtype=np.float64)
    x = np.arange(n, dtype=np.float64)
    x -= x.mean()
    denom = np.sum(x * x)
    centered = values - values.mean(axis=1, keepdims=True)
    return centered @ x / max(float(denom), EPS)


def _auc(values: np.ndarray) -> np.ndarray:
    if values.shape[1] == 1:
        return values[:, 0]
    area = values[:, 1:-1].sum(axis=1) + 0.5 * (values[:, 0] + values[:, -1])
    return area / max(1, values.shape[1] - 1)


def build_pa_trajectory_features(
    answer: dict[str, np.ndarray],
    prompt: dict[str, np.ndarray],
    pooling_modes: list[str],
    layer_start: int,
    layer_end: int,
) -> tuple[np.ndarray, list[str]]:
    columns, names = [], []
    layers = list(range(layer_start, layer_end + 1))
    for mode in pooling_modes:
        suffix = POOLING_SUFFIX[mode]
        metric_series = {metric: [] for metric in SCALAR_METRICS}
        for layer in layers:
            metrics = _scalar_metric_arrays(prompt[mode][:, layer - 1, :], answer[mode][:, layer - 1, :])
            for metric in SCALAR_METRICS:
                metric_series[metric].append(metrics[metric])
        for metric, series in metric_series.items():
            values = np.vstack(series).T.astype(np.float64)
            stats = {
                "mean": values.mean(axis=1),
                "std": values.std(axis=1),
                "min": values.min(axis=1),
                "max": values.max(axis=1),
                "first": values[:, 0],
                "last": values[:, -1],
                "last_minus_first": values[:, -1] - values[:, 0],
                "slope": _slope(values),
                "auc": _auc(values),
            }
            for stat in TRAJECTORY_STATS:
                columns.append(stats[stat])
                names.append(f"pa_{metric}_{suffix}_layers{layer_start}_{layer_end}_{stat}")
    X = np.vstack(columns).T.astype(np.float32)
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0), names


class FeatureBlockPipeline:
    def __init__(self, pca_components: int, vector_blocks: list[str], scalar_blocks: list[str], seed: int) -> None:
        self.pca_components = pca_components
        self.vector_blocks = vector_blocks
        self.scalar_blocks = scalar_blocks
        self.seed = seed
        self.vector_scalers = {name: StandardScaler() for name in vector_blocks}
        self.vector_pcas: dict[str, PCA] = {}
        self.scalar_scalers = {name: StandardScaler() for name in scalar_blocks}

    def fit_transform(self, blocks: dict[str, np.ndarray], idx: np.ndarray) -> tuple[np.ndarray, list[str]]:
        parts, names = [], []
        for block in self.vector_blocks:
            scaled = self.vector_scalers[block].fit_transform(blocks[block][idx])
            n = min(self.pca_components, scaled.shape[0] - 1, scaled.shape[1])
            pca = PCA(n_components=n, random_state=self.seed)
            self.vector_pcas[block] = pca
            parts.append(pca.fit_transform(scaled))
            names.extend([f"{block}_pca_{i}" for i in range(n)])
        for block in self.scalar_blocks:
            parts.append(self.scalar_scalers[block].fit_transform(blocks[block][idx]))
            names.extend(blocks[f"{block}__names"])
        return np.concatenate(parts, axis=1).astype(np.float32), names

    def transform(self, blocks: dict[str, np.ndarray], idx: np.ndarray) -> np.ndarray:
        parts = []
        for block in self.vector_blocks:
            parts.append(self.vector_pcas[block].transform(self.vector_scalers[block].transform(blocks[block][idx])))
        for block in self.scalar_blocks:
            parts.append(self.scalar_scalers[block].transform(blocks[block][idx]))
        return np.concatenate(parts, axis=1).astype(np.float32)

    def output_dim_before_pca(self, blocks: dict[str, np.ndarray]) -> int:
        return int(sum(blocks[name].shape[1] for name in [*self.vector_blocks, *self.scalar_blocks]))


def _metrics(y_true: np.ndarray, scores: np.ndarray, threshold: float = 0.0) -> dict[str, float]:
    pred = (scores >= threshold).astype(int)
    try:
        auc = roc_auc_score(y_true, scores)
    except ValueError:
        auc = float("nan")
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "roc_auc": float(auc),
    }


def run_single_experiment(
    spec: dict,
    blocks: dict[str, np.ndarray],
    y: np.ndarray,
    splits: list,
    pca_components: int,
    svm_c: float,
    seed: int,
) -> tuple[dict, list[FoldArtifact]]:
    rows, artifacts = [], []
    for fold, (tr, va, te) in enumerate(splits, 1):
        if va is None:
            raise ValueError("Validation split is required.")
        pipe = FeatureBlockPipeline(pca_components, spec["vector_blocks"], spec["scalar_blocks"], seed)
        X_train, names = pipe.fit_transform(blocks, tr)
        X_val = pipe.transform(blocks, va)
        X_test = pipe.transform(blocks, te)
        if not np.isfinite(X_train).all() or not np.isfinite(X_val).all() or not np.isfinite(X_test).all():
            raise ValueError(f"Non-finite values in final features for {spec['name']}")
        clf = SVC(C=svm_c, kernel="linear", probability=False, random_state=seed)
        clf.fit(X_train, y[tr])
        val_scores = clf.decision_function(X_val)
        test_scores = clf.decision_function(X_test)
        threshold, _ = find_best_threshold_by_f1(y[va], val_scores)
        row = {
            "fold": fold,
            "train_size": len(tr),
            "val_size_or_test_size": len(te),
            "output_feature_dim_before_pca": pipe.output_dim_before_pca(blocks),
            "output_feature_dim_after_pca": X_train.shape[1],
            "best_threshold": threshold,
        }
        row.update({f"val_{k}": v for k, v in _metrics(y[va], val_scores, threshold).items()})
        row.update({k: v for k, v in _metrics(y[te], test_scores, threshold).items()})
        rows.append(row)
        artifacts.append(FoldArtifact(spec["name"], fold, clf, X_val, y[va], names, threshold))

    df = pd.DataFrame(rows)
    out = {
        "experiment_name": spec["name"],
        "feature_mode": spec["feature_mode"],
        "layer_range": spec["layer_range"],
        "pooling_modes": ",".join(spec["pooling_modes"]),
        "pca_components": pca_components,
        "classifier": "linear_svm",
        "C": svm_c,
        "accuracy": float(df["accuracy"].mean()),
        "precision": float(df["precision"].mean()),
        "recall": float(df["recall"].mean()),
        "f1": float(df["f1"].mean()),
        "roc_auc": float(df["roc_auc"].mean()),
        "train_size": float(df["train_size"].mean()),
        "val_size_or_test_size": float(df["val_size_or_test_size"].mean()),
        "random_seed": seed,
        "output_feature_dim_before_pca": float(df["output_feature_dim_before_pca"].mean()),
        "output_feature_dim_after_pca": float(df["output_feature_dim_after_pca"].mean()),
        "val_f1": float(df["val_f1"].mean()),
        "val_roc_auc": float(df["val_roc_auc"].mean()),
        "best_threshold": float(df["best_threshold"].mean()),
    }
    return out, artifacts


def _feature_group(name: str) -> str:
    if name.startswith("answer_stack"):
        return "answer_stack"
    if name.startswith("prompt_stack"):
        return "prompt_stack"
    if name.startswith("delta_stack"):
        return "delta_stack"
    if "pa_cos_dist" in name or "pa_cos_" in name:
        return "pa_scalar_cosine"
    if "pa_l1_" in name:
        return "pa_scalar_l1"
    if "pa_l2_" in name:
        return "pa_scalar_l2"
    if "pa_linf_" in name:
        return "pa_scalar_linf"
    if any(token in name for token in ("prompt_norm", "answer_norm", "delta_norm", "norm_ratio", "abs_norm_diff")):
        return "pa_scalar_norms"
    if "layers10_24" in name:
        return "pa_trajectory"
    return "other"


def permutation_importance_grouped(artifacts: list[FoldArtifact], repeats: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for artifact in artifacts:
        groups: dict[str, list[int]] = {}
        for i, name in enumerate(artifact.feature_names):
            groups.setdefault(_feature_group(name), []).append(i)
        base = f1_score(
            artifact.y_eval,
            (artifact.model.decision_function(artifact.X_eval) >= artifact.threshold).astype(int),
            zero_division=0,
        )
        for group, cols in groups.items():
            vals = []
            for _ in range(repeats):
                Xp = artifact.X_eval.copy()
                order = rng.permutation(Xp.shape[0])
                Xp[:, cols] = Xp[order][:, cols]
                score = f1_score(
                    artifact.y_eval,
                    (artifact.model.decision_function(Xp) >= artifact.threshold).astype(int),
                    zero_division=0,
                )
                vals.append(score)
            vals_arr = np.asarray(vals)
            rows.append({
                "fold": artifact.fold,
                "group": group,
                "n_features": len(cols),
                "base_f1": float(base),
                "permuted_f1_mean": float(vals_arr.mean()),
                "permuted_f1_std": float(vals_arr.std()),
                "importance_mean": float(base - vals_arr.mean()),
            })
    raw = pd.DataFrame(rows)
    return (
        raw.groupby("group", as_index=False)
        .agg(
            n_features=("n_features", "mean"),
            base_f1=("base_f1", "mean"),
            permuted_f1_mean=("permuted_f1_mean", "mean"),
            permuted_f1_std=("permuted_f1_std", "mean"),
            importance_mean=("importance_mean", "mean"),
        )
        .sort_values("importance_mean", ascending=False)
    )


def _parse_layers(values: list[int]) -> list[int]:
    layers = [int(v) for v in values]
    if not layers or min(layers) < 1 or max(layers) > N_TRANSFORMER_LAYERS:
        raise ValueError(f"Invalid layers {layers}")
    return layers


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default=_DEFAULT_MODEL)
    parser.add_argument("--layers-answer", nargs="+", type=int, default=[21, 22, 23, 24])
    parser.add_argument("--layers-trajectory-start", type=int, default=10)
    parser.add_argument("--layers-trajectory-end", type=int, default=24)
    parser.add_argument("--pca-components", type=int, default=80)
    parser.add_argument("--svm-c", type=float, default=0.1)
    parser.add_argument("--output-csv", type=Path, default=ROOT / "results" / "prompt_answer_geometry_ablation.csv")
    parser.add_argument("--importance-csv", type=Path, default=ROOT / "results" / "prompt_answer_geometry_permutation_importance.csv")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "results" / "prompt_answer_geometry_cache")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--token-window", type=int, default=32)
    parser.add_argument("--pooling-modes", nargs="+", default=list(DEFAULT_POOLING_MODES))
    parser.add_argument("--importance-repeats", type=int, default=10)
    args = parser.parse_args()

    np.random.seed(args.seed)
    layers_answer = _parse_layers(args.layers_answer)
    unsupported = [mode for mode in args.pooling_modes if mode not in POOLING_SUFFIX]
    if unsupported:
        raise ValueError(f"Unsupported pooling modes {unsupported}; expected {list(POOLING_SUFFIX)}")

    answer, prompt, y, splits = load_or_extract_prompt_answer_vectors(
        args.cache_dir,
        args.model_name,
        args.pooling_modes,
        args.token_window,
    )
    print(f"Samples: {len(y)}")
    for i, (tr, va, te) in enumerate(splits, 1):
        print(f"Fold {i}: train={len(tr)} {_class_distribution(y, tr)} val={len(va)} {_class_distribution(y, va)} test={len(te)} {_class_distribution(y, te)}")

    answer_stack, answer_names = build_answer_stack(answer, args.pooling_modes, layers_answer)
    prompt_stack, prompt_names = build_prompt_stack(prompt, args.pooling_modes, layers_answer)
    delta_stack, delta_names = build_delta_stack(answer, prompt, args.pooling_modes, layers_answer)
    pa_scalar, pa_scalar_names = build_pa_scalar_features(answer, prompt, args.pooling_modes, layers_answer)
    pa_traj, pa_traj_names = build_pa_trajectory_features(
        answer,
        prompt,
        args.pooling_modes,
        args.layers_trajectory_start,
        args.layers_trajectory_end,
    )
    blocks: dict[str, np.ndarray | list[str]] = {
        "answer_stack": answer_stack,
        "answer_stack__names": answer_names,
        "prompt_stack": prompt_stack,
        "prompt_stack__names": prompt_names,
        "delta_stack": delta_stack,
        "delta_stack__names": delta_names,
        "pa_scalar": pa_scalar,
        "pa_scalar__names": pa_scalar_names,
        "pa_trajectory": pa_traj,
        "pa_trajectory__names": pa_traj_names,
    }
    for name in ("answer_stack", "prompt_stack", "delta_stack", "pa_scalar", "pa_trajectory"):
        X = blocks[name]
        assert isinstance(X, np.ndarray)
        if not np.isfinite(X).all():
            print(f"WARNING: non-finite values in {name}; replacing with zeros.")
            blocks[name] = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        print(f"{name}: {blocks[name].shape}")

    feature_names_path = args.output_csv.parent / "prompt_answer_geometry_feature_names.json"
    feature_names_path.parent.mkdir(parents=True, exist_ok=True)
    with feature_names_path.open("w") as f:
        json.dump(
            {
                "answer_stack": answer_names,
                "prompt_stack": prompt_names,
                "delta_stack": delta_names,
                "pa_scalar": pa_scalar_names,
                "pa_trajectory": pa_traj_names,
            },
            f,
            indent=2,
        )

    common = {"pooling_modes": args.pooling_modes}
    specs = [
        {"name": "E0_answer_stack_baseline", "feature_mode": "answer_stack", "layer_range": "21-24", "vector_blocks": ["answer_stack"], "scalar_blocks": [], **common},
        {"name": "E1_prompt_stack_baseline", "feature_mode": "prompt_stack", "layer_range": "21-24", "vector_blocks": ["prompt_stack"], "scalar_blocks": [], **common},
        {"name": "E2_pa_scalar_21_24_only", "feature_mode": "pa_scalar", "layer_range": "21-24", "vector_blocks": [], "scalar_blocks": ["pa_scalar"], **common},
        {"name": "E3_answer_stack_plus_pa_scalar_21_24", "feature_mode": "answer_stack+pa_scalar", "layer_range": "21-24", "vector_blocks": ["answer_stack"], "scalar_blocks": ["pa_scalar"], **common},
        {"name": "E4_pa_trajectory_10_24_only", "feature_mode": "pa_trajectory", "layer_range": "10-24", "vector_blocks": [], "scalar_blocks": ["pa_trajectory"], **common},
        {"name": "E5_answer_stack_plus_pa_trajectory_10_24", "feature_mode": "answer_stack+pa_trajectory", "layer_range": "10-24", "vector_blocks": ["answer_stack"], "scalar_blocks": ["pa_trajectory"], **common},
        {"name": "E6_delta_stack_only", "feature_mode": "delta_stack", "layer_range": "21-24", "vector_blocks": ["delta_stack"], "scalar_blocks": [], **common},
        {"name": "E7_answer_stack_plus_delta_stack", "feature_mode": "answer_stack+delta_stack", "layer_range": "21-24", "vector_blocks": ["answer_stack", "delta_stack"], "scalar_blocks": [], **common},
        {"name": "E8_answer_stack_plus_delta_stack_plus_pa_scalar", "feature_mode": "answer_stack+delta_stack+pa_scalar", "layer_range": "21-24", "vector_blocks": ["answer_stack", "delta_stack"], "scalar_blocks": ["pa_scalar"], **common},
    ]

    t0 = time.time()
    rows, artifact_by_name = [], {}
    for spec in specs:
        print(f"\nRunning {spec['name']}: vectors={spec['vector_blocks']} scalars={spec['scalar_blocks']}")
        row, artifacts = run_single_experiment(
            spec,
            blocks,  # type: ignore[arg-type]
            y,
            splits,
            args.pca_components,
            args.svm_c,
            args.seed,
        )
        rows.append(row)
        artifact_by_name[spec["name"]] = artifacts
        print(f"  f1={row['f1']:.4f} roc_auc={row['roc_auc']:.4f} val_f1={row['val_f1']:.4f}")

    result = pd.DataFrame(rows).sort_values(["f1", "roc_auc"], ascending=False)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output_csv, index=False)

    combined = result[result["experiment_name"].isin([
        "E3_answer_stack_plus_pa_scalar_21_24",
        "E5_answer_stack_plus_pa_trajectory_10_24",
        "E7_answer_stack_plus_delta_stack",
        "E8_answer_stack_plus_delta_stack_plus_pa_scalar",
    ])]
    best_name = str(combined.sort_values(["f1", "roc_auc"], ascending=False).iloc[0]["experiment_name"])
    importance = permutation_importance_grouped(artifact_by_name[best_name], args.importance_repeats, args.seed)
    args.importance_csv.parent.mkdir(parents=True, exist_ok=True)
    importance.to_csv(args.importance_csv, index=False)

    print("\nPrompt-answer geometry ablation sorted by F1/ROC-AUC")
    print(result.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("\nSorted by ROC-AUC")
    print(result.sort_values(["roc_auc", "f1"], ascending=False).to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nBest combined experiment for permutation importance: {best_name}")
    print(importance.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nSaved results to {args.output_csv}")
    print(f"Saved permutation importance to {args.importance_csv}")
    print(f"Saved feature names to {feature_names_path}")
    print(f"Runtime seconds: {time.time() - t0:.1f}")


if __name__ == "__main__":
    main()
