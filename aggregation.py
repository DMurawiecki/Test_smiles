"""
aggregation.py — Token aggregation strategy and feature extraction
               (student-implemented).

Converts per-token, per-layer hidden states from the extraction loop in
``solution.py`` into flat feature vectors for the probe classifier.

Two stages can be customised independently:

  1. ``aggregate`` — select layers and token positions, pool into a vector.
  2. ``extract_geometric_features`` — optional hand-crafted features
     (enabled by setting ``USE_GEOMETRIC = True`` in ``solution.py``).

Both stages are combined by ``aggregation_and_feature_extraction``, the
single entry point called from the notebook.
"""

from __future__ import annotations

import torch


FINAL_TOKEN_WINDOWS = (1, 8, 16, 32, 64, 128, 256, 512)
FINAL_LAYER_WINDOW = (21, 24)
FINAL_TRAJECTORY_RANGE = (10, 18)
TRAJECTORY_FEATURE_DIM = 34

AGGREGATION_MODE = "last_1_8_16_32_64_128_256_512_range"
SPECTRAL_MODE = "none"

SUPPORTED_AGGREGATION_MODES = (
    AGGREGATION_MODE,
)

SUPPORTED_SPECTRAL_MODES = (
    "none",
    "top_eigenvalues",
    "sum_eigenvalues",
    "logdet",
    "effective_rank",
    "participation_ratio",
    "condition_number",
    "spectral_entropy",
    "all",
    "all_without_condition_number",
)

SPECTRAL_WINDOWS = (8, 16, 32, 64)
SPECTRAL_TOP_K = 5
_LAST_FEATURE_NAMES: list[str] = []


def _as_layer_tensor(hidden_states: torch.Tensor | tuple | list) -> torch.Tensor:
    """Return hidden states as (n_layers, seq_len, hidden_dim)."""
    if isinstance(hidden_states, torch.Tensor):
        return hidden_states
    return torch.stack(list(hidden_states), dim=0)


def _has_embedding_layer(hidden_states: torch.Tensor) -> bool:
    """Infer whether index 0 is the embedding layer.

    Qwen2.5-0.5B normally returns 25 tensors: embeddings plus 24 transformer
    layers. Some pipelines drop embeddings and keep only the 24 transformer
    layers. This helper keeps conceptual layer numbers stable in both cases.
    """
    return hidden_states.size(0) == 25


def transformer_layer_to_index(
    layer_number: int,
    hidden_states: torch.Tensor | tuple | list,
) -> int:
    """Map conceptual transformer layer number (1-based) to a valid index."""
    hs = _as_layer_tensor(hidden_states)
    n_layers = hs.size(0)
    n_transformer_layers = n_layers - 1 if _has_embedding_layer(hs) else n_layers
    if layer_number < 1 or layer_number > n_transformer_layers:
        raise ValueError(
            f"Transformer layer {layer_number} is not available for "
            f"{n_layers} hidden-state tensors."
        )
    return layer_number if _has_embedding_layer(hs) else layer_number - 1


def transformer_window_to_indices(
    start_layer: int,
    end_layer: int,
    hidden_states: torch.Tensor | tuple | list,
) -> list[int]:
    """Return valid tensor indices for an inclusive conceptual layer window."""
    hs = _as_layer_tensor(hidden_states)
    n_layers = hs.size(0)
    n_transformer_layers = n_layers - 1 if _has_embedding_layer(hs) else n_layers
    start = max(1, start_layer)
    end = min(end_layer, n_transformer_layers)
    if start > end:
        raise ValueError(
            f"Layer window {start_layer}-{end_layer} has no valid layers for "
            f"{n_layers} hidden-state tensors."
        )
    return [transformer_layer_to_index(layer, hs) for layer in range(start, end + 1)]


def _real_token_positions(attention_mask: torch.Tensor) -> torch.Tensor:
    """Return indices of non-padding tokens; fall back to the final position."""
    mask = attention_mask.detach()
    if mask.dim() != 1:
        mask = mask.reshape(-1)
    positions = torch.nonzero(mask.to(dtype=torch.bool), as_tuple=False).flatten()
    if positions.numel() == 0:
        return torch.tensor([mask.numel() - 1], device=mask.device, dtype=torch.long)
    return positions.to(dtype=torch.long)


def _tail_mean(real_sequence: torch.Tensor, width: int) -> torch.Tensor:
    tail = real_sequence[-min(width, real_sequence.size(0)) :]
    return tail.mean(dim=0)


def _pool_tail_tokens(
    sequence_repr: torch.Tensor,
    real_positions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pool the answer tail: last token and means over last 8/16/32 tokens."""
    positions = real_positions.to(device=sequence_repr.device)
    real_sequence = sequence_repr.index_select(0, positions).to(dtype=torch.float32)
    pooled = [
        real_sequence[-1],
        _tail_mean(real_sequence, 8),
        _tail_mean(real_sequence, 16),
        _tail_mean(real_sequence, 32),
    ]
    return torch.cat(pooled, dim=0), real_sequence


def _tail_range_stack(
    sequence_repr: torch.Tensor,
    real_positions: torch.Tensor,
    windows: tuple[int, ...] = FINAL_TOKEN_WINDOWS,
) -> torch.Tensor:
    """Concatenate range-pooled tail windows from the final hidden sequence."""
    positions = real_positions.to(device=sequence_repr.device)
    real_sequence = sequence_repr.index_select(0, positions).to(dtype=torch.float32)
    parts: list[torch.Tensor] = []
    for width in windows:
        tail = real_sequence[-min(width, real_sequence.size(0)) :]
        parts.append(tail.max(dim=0).values - tail.min(dim=0).values)
    return torch.cat(parts, dim=0)


def _safe_cosine(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    denom = a.norm(p=2) * b.norm(p=2)
    if float(denom.detach().cpu()) <= eps:
        return torch.tensor(0.0, device=a.device, dtype=torch.float32)
    return (torch.dot(a, b) / (denom + eps)).to(dtype=torch.float32)


def _segment_means(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if values.numel() == 0:
        zero = torch.tensor(0.0, device=values.device, dtype=torch.float32)
        return zero, zero, zero
    chunks = torch.tensor_split(values, 3)
    means = [
        chunk.mean() if chunk.numel() else torch.tensor(0.0, device=values.device)
        for chunk in chunks
    ]
    return tuple(mean.to(dtype=torch.float32) for mean in means)


def _layer_trajectory_vectors(
    hidden_states: torch.Tensor,
    real_positions: torch.Tensor,
) -> torch.Tensor:
    """Return one pooled vector per transformer layer, shape [24, hidden_dim]."""
    vectors: list[torch.Tensor] = []
    positions = real_positions.to(device=hidden_states.device)
    for layer_number in range(1, 25):
        idx = transformer_layer_to_index(layer_number, hidden_states)
        vectors.append(hidden_states[idx].index_select(0, positions).float().mean(dim=0))
    return torch.stack(vectors, dim=0)


def _trajectory_feature_names(layer_start: int, layer_end: int) -> list[str]:
    suffixes = (
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
        "path_length",
        "endpoint_distance",
        "straightness",
        "cos_step_mean",
        "cos_step_std",
        "cos_step_min",
        "cos_step_max",
        "cos_first_last",
        "cos_early_mean",
        "cos_mid_mean",
        "cos_late_mean",
        "late_l2_to_final_mean",
        "late_l2_to_final_max",
        "late_cos_to_final_mean",
        "late_cos_to_final_min",
        "delta_cos_mean",
        "delta_cos_std",
        "delta_cos_min",
        "delta_cos_max",
        "curvature_mean",
        "curvature_max",
        "early_curvature_mean",
        "mid_curvature_mean",
        "late_curvature_mean",
    )
    return [f"trajectory_10_18__{suffix}" for suffix in suffixes]


def _trajectory_features(
    hidden_states: torch.Tensor,
    real_positions: torch.Tensor,
    layer_start: int = FINAL_TRAJECTORY_RANGE[0],
    layer_end: int = FINAL_TRAJECTORY_RANGE[1],
) -> torch.Tensor:
    """Compute the 34 scalar layer-trajectory features from experiment E_range_10_18."""
    eps = 1e-8
    all_layers = _layer_trajectory_vectors(hidden_states, real_positions)
    Z = all_layers[layer_start - 1 : layer_end].to(dtype=torch.float32)
    if Z.size(0) < 2:
        return torch.zeros(TRAJECTORY_FEATURE_DIM, device=hidden_states.device)

    deltas = Z[1:] - Z[:-1]
    step_norms = deltas.norm(p=2, dim=1)
    early_step, mid_step, late_step = _segment_means(step_norms)
    path_length = step_norms.sum()
    endpoint_distance = (Z[-1] - Z[0]).norm(p=2)

    cos_steps = torch.stack([_safe_cosine(Z[i], Z[i - 1], eps) for i in range(1, Z.size(0))])
    early_cos, mid_cos, late_cos = _segment_means(cos_steps)

    k = min(4, Z.size(0))
    late_before_final = Z[-k:-1]
    if late_before_final.numel():
        late_l2 = (late_before_final - Z[-1]).norm(p=2, dim=1)
        late_cos_to_final = torch.stack([_safe_cosine(z, Z[-1], eps) for z in late_before_final])
        late_l2_mean = late_l2.mean()
        late_l2_max = late_l2.max()
        late_cos_mean = late_cos_to_final.mean()
        late_cos_min = late_cos_to_final.min()
    else:
        late_l2_mean = late_l2_max = late_cos_mean = late_cos_min = torch.tensor(
            0.0, device=hidden_states.device
        )

    if deltas.size(0) >= 2:
        delta_cos = torch.stack(
            [_safe_cosine(deltas[i], deltas[i + 1], eps) for i in range(deltas.size(0) - 1)]
        )
        curvature = 1.0 - delta_cos
        early_curv, mid_curv, late_curv = _segment_means(curvature)
        delta_values = [
            delta_cos.mean(),
            delta_cos.std(unbiased=False),
            delta_cos.min(),
            delta_cos.max(),
            curvature.mean(),
            curvature.max(),
            early_curv,
            mid_curv,
            late_curv,
        ]
    else:
        delta_values = [torch.tensor(0.0, device=hidden_states.device)] * 9

    values = [
        step_norms.mean(),
        step_norms.std(unbiased=False),
        step_norms.max(),
        step_norms.min(),
        step_norms.median(),
        step_norms[-1],
        early_step,
        mid_step,
        late_step,
        early_step / (late_step + eps),
        path_length,
        endpoint_distance,
        endpoint_distance / (path_length + eps),
        cos_steps.mean(),
        cos_steps.std(unbiased=False),
        cos_steps.min(),
        cos_steps.max(),
        _safe_cosine(Z[0], Z[-1], eps),
        early_cos,
        mid_cos,
        late_cos,
        late_l2_mean,
        late_l2_max,
        late_cos_mean,
        late_cos_min,
        *delta_values,
    ]
    return torch.nan_to_num(torch.stack(values).to(dtype=torch.float32), nan=0.0, posinf=0.0, neginf=0.0)


def _spectral_stats_enabled(mode: str) -> tuple[str, ...]:
    """Return spectral statistic names included by ``mode``."""
    stats = {
        "none": (),
        "top_eigenvalues": tuple(f"top_eig_{i}" for i in range(1, SPECTRAL_TOP_K + 1)),
        "sum_eigenvalues": ("sum_eigenvalues",),
        "logdet": ("logdet",),
        "effective_rank": ("effective_rank",),
        "participation_ratio": ("participation_ratio",),
        "condition_number": ("log_condition_number",),
        "spectral_entropy": ("spectral_entropy",),
        "all_without_condition_number": (
            *(f"top_eig_{i}" for i in range(1, SPECTRAL_TOP_K + 1)),
            "sum_eigenvalues",
            "logdet",
            "effective_rank",
            "participation_ratio",
            "spectral_entropy",
        ),
        "all": (
            *(f"top_eig_{i}" for i in range(1, SPECTRAL_TOP_K + 1)),
            "sum_eigenvalues",
            "logdet",
            "effective_rank",
            "participation_ratio",
            "log_condition_number",
            "spectral_entropy",
        ),
    }
    if mode not in stats:
        raise ValueError(
            f"Unknown SPECTRAL_MODE={mode!r}. Supported modes: "
            f"{', '.join(SUPPORTED_SPECTRAL_MODES)}"
        )
    return stats[mode]


def _zero_spectral_values(
    stats: tuple[str, ...],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    return {
        stat: torch.tensor(0.0, device=device, dtype=torch.float32)
        for stat in stats
    }


def _spectral_features_for_window(
    real_sequence: torch.Tensor,
    width: int,
    stats: tuple[str, ...],
) -> dict[str, torch.Tensor]:
    """Compute stable Gram-spectrum features for a tail token window."""
    if not stats:
        return {}

    device = real_sequence.device
    if real_sequence.size(0) < 2:
        return _zero_spectral_values(stats, device)

    tail = real_sequence[-min(width, real_sequence.size(0)) :].to(dtype=torch.float64)
    if tail.size(0) < 2:
        return _zero_spectral_values(stats, device)

    eps = 1e-8
    centered = tail - tail.mean(dim=0, keepdim=True)
    gram = centered @ centered.T / max(float(centered.size(1)), 1.0)
    eig = torch.linalg.eigvalsh(gram).clamp_min(eps)
    eig = torch.nan_to_num(eig, nan=eps, posinf=eps, neginf=eps)
    eig_desc = torch.sort(eig, descending=True).values

    eig_sum = eig.sum()
    spectral_entropy = -(
        (eig / (eig_sum + eps)) * torch.log((eig / (eig_sum + eps)) + eps)
    ).sum()
    condition_number = eig.max() / eig.min().clamp_min(eps)

    values: dict[str, torch.Tensor] = {}
    for i in range(SPECTRAL_TOP_K):
        key = f"top_eig_{i + 1}"
        values[key] = (
            eig_desc[i] if i < eig_desc.numel() else torch.tensor(0.0, device=device)
        )
    values.update(
        {
            "sum_eigenvalues": eig_sum,
            "logdet": torch.log(eig).sum(),
            "spectral_entropy": spectral_entropy,
            "effective_rank": torch.exp(spectral_entropy),
            "participation_ratio": eig_sum.pow(2) / (eig.pow(2).sum() + eps),
            "log_condition_number": torch.log(condition_number + eps),
        }
    )

    return {
        stat: torch.nan_to_num(
            values[stat].to(device=device, dtype=torch.float32),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        for stat in stats
    }


def _spectral_feature_names(mode: str) -> list[str]:
    stats = _spectral_stats_enabled(mode)
    names: list[str] = []
    for width in SPECTRAL_WINDOWS:
        for stat in stats:
            display_stat = (
                "condition_number" if stat == "log_condition_number" else stat
            )
            names.append(
                "spectral__block_mean_21_24"
                f"__window_{width}__{display_stat}"
            )
    return names


def _spectral_features(
    hidden_states: torch.Tensor,
    real_positions: torch.Tensor,
) -> tuple[torch.Tensor, list[str]]:
    """Return spectral geometry features for the mean(21-24) block."""
    stats = _spectral_stats_enabled(SPECTRAL_MODE)
    names = _spectral_feature_names(SPECTRAL_MODE)
    if not stats:
        return torch.zeros(0, device=hidden_states.device), names

    sequence_repr = _layer_window_repr(hidden_states, 21, 24)
    positions = real_positions.to(device=sequence_repr.device)
    real_sequence = sequence_repr.index_select(0, positions).to(dtype=torch.float32)

    values: list[torch.Tensor] = []
    for width in SPECTRAL_WINDOWS:
        window_values = _spectral_features_for_window(real_sequence, width, stats)
        values.extend(window_values[stat] for stat in stats)

    if not values:
        return torch.zeros(0, device=hidden_states.device), names
    return torch.stack(values).to(dtype=torch.float32), names


def _layer_window_repr(
    hidden_states: torch.Tensor,
    start_layer: int,
    end_layer: int,
) -> torch.Tensor:
    """Average hidden states over an inclusive middle/late transformer window."""
    indices = transformer_window_to_indices(start_layer, end_layer, hidden_states)
    layer_ids = torch.tensor(indices, device=hidden_states.device, dtype=torch.long)
    return hidden_states.index_select(0, layer_ids).to(dtype=torch.float32).mean(dim=0)


def _final_layer_repr(hidden_states: torch.Tensor) -> torch.Tensor:
    final_layer_number = (
        hidden_states.size(0) - 1
        if _has_embedding_layer(hidden_states)
        else hidden_states.size(0)
    )
    final_idx = transformer_layer_to_index(
        final_layer_number,
        hidden_states,
    )
    return hidden_states[final_idx].to(dtype=torch.float32)


def _selected_blocks(mode: str) -> list[tuple[str, int | None, int | None]]:
    """Translate an aggregation mode into ordered layer blocks."""
    blocks = {
        "final_only": [("final", None, None)],
        "mean_10_20": [("window", 10, 20)],
        "mean_13_18": [("window", 13, 18)],
        "mean_21_24": [("window", 21, 24)],
        "mean_10_20_plus_21_24": [("window", 10, 20), ("window", 21, 24)],
        "mean_10_20_plus_final": [("window", 10, 20), ("final", None, None)],
        "mean_10_20_plus_21_24_plus_final": [
            ("window", 10, 20),
            ("window", 21, 24),
            ("final", None, None),
        ],
    }
    if mode not in blocks:
        raise ValueError(
            f"Unknown AGGREGATION_MODE={mode!r}. Supported modes: "
            f"{', '.join(SUPPORTED_AGGREGATION_MODES)}"
        )
    return blocks[mode]


def aggregate(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Convert per-token hidden states into a single feature vector.

    Args:
        hidden_states:  Tensor of shape ``(n_layers, seq_len, hidden_dim)``.
                        Layer index 0 is the token embedding; index -1 is the
                        final transformer layer.
        attention_mask: 1-D tensor of shape ``(seq_len,)`` with 1 for real
                        tokens and 0 for padding.

    Returns:
        A 1-D feature tensor of shape ``(hidden_dim,)`` or
        ``(k * hidden_dim,)`` if multiple layers are concatenated.

    Student task:
        Replace or extend the skeleton below with alternative layer selection,
        token pooling (mean, max, weighted), or multi-layer fusion strategies.
    """
    hs = _as_layer_tensor(hidden_states).detach()
    real_positions = _real_token_positions(attention_mask).to(device=hs.device)

    global _LAST_FEATURE_NAMES

    sequence_repr = _layer_window_repr(hs, *FINAL_LAYER_WINDOW)
    hidden_features = _tail_range_stack(sequence_repr, real_positions)
    trajectory_features = _trajectory_features(hs, real_positions)
    out = torch.cat([hidden_features, trajectory_features], dim=0)
    out = torch.nan_to_num(
        out.to(dtype=torch.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).cpu()
    hidden_names = [
        f"hidden_range_last_{window}_{dim}"
        for window in FINAL_TOKEN_WINDOWS
        for dim in range(sequence_repr.size(-1))
    ]
    _LAST_FEATURE_NAMES = hidden_names + _trajectory_feature_names(*FINAL_TRAJECTORY_RANGE)
    return out


def get_last_feature_names() -> list[str]:
    """Return feature names from the most recent ``aggregate`` call."""
    return list(_LAST_FEATURE_NAMES)


def build_feature_names(feature_dim: int) -> list[str]:
    """Build final hidden-stack names plus trajectory feature names."""
    trajectory_names = _trajectory_feature_names(*FINAL_TRAJECTORY_RANGE)
    n_hidden = feature_dim - len(trajectory_names)
    if n_hidden < 0:
        raise ValueError(
            f"feature_dim={feature_dim} is smaller than the trajectory feature "
            f"count {len(trajectory_names)}."
        )
    return [f"hidden_{i}" for i in range(n_hidden)] + trajectory_names


def count_spectral_features(mode: str | None = None) -> int:
    """Return the number of scalar spectral features for a spectral mode."""
    mode = SPECTRAL_MODE if mode is None else mode
    return len(SPECTRAL_WINDOWS) * len(_spectral_stats_enabled(mode))


def extract_geometric_features(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Extract hand-crafted geometric / statistical features from hidden states.

    Called only when ``USE_GEOMETRIC = True`` in ``solution.ipynb``.  The
    returned tensor is concatenated with the output of ``aggregate``.

    Args:
        hidden_states:  Tensor of shape ``(n_layers, seq_len, hidden_dim)``.
        attention_mask: 1-D tensor of shape ``(seq_len,)`` with 1 for real
                        tokens and 0 for padding.

    Returns:
        A 1-D float tensor of shape ``(n_geometric_features,)``.  The length
        must be the same for every sample.

    Student task:
        Replace the stub below.  Possible features: layer-wise activation
        norms, inter-layer cosine similarity (representation drift), or
        sequence length.
    """
    # ------------------------------------------------------------------
    # STUDENT: Replace or extend the geometric feature extraction below.
    # ------------------------------------------------------------------

    # Placeholder: returns an empty tensor (no geometric features).
    return torch.zeros(0)


def aggregation_and_feature_extraction(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    use_geometric: bool = False,
) -> torch.Tensor:
    """Aggregate hidden states and optionally append geometric features.

    Main entry point called from ``solution.ipynb`` for each sample.
    Concatenates the output of ``aggregate`` with that of
    ``extract_geometric_features`` when ``use_geometric=True``.

    Args:
        hidden_states:  Tensor of shape ``(n_layers, seq_len, hidden_dim)``
                        for a single sample.
        attention_mask: 1-D tensor of shape ``(seq_len,)`` with 1 for real
                        tokens and 0 for padding.
        use_geometric:  Whether to append geometric features.  Controlled by
                        the ``USE_GEOMETRIC`` flag in ``solution.ipynb``.

    Returns:
        A 1-D float tensor of shape ``(feature_dim,)`` where
        ``feature_dim = hidden_dim`` (or larger for multi-layer or geometric
        concatenations).
    """
    agg_features = aggregate(hidden_states, attention_mask)  # (feature_dim,)

    if use_geometric:
        geo_features = extract_geometric_features(hidden_states, attention_mask)
        return torch.cat([agg_features, geo_features], dim=0)

    return agg_features
