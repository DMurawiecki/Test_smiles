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

import os

import torch


AGGREGATION_MODE = os.getenv("AGGREGATION_MODE", "mean_21_24")
SPECTRAL_MODE = os.getenv("SPECTRAL_MODE", "none")

SUPPORTED_AGGREGATION_MODES = (
    "final_only",
    "mean_10_20",
    "mean_13_18",
    "mean_21_24",
    "mean_10_20_plus_21_24",
    "mean_10_20_plus_final",
    "mean_10_20_plus_21_24_plus_final",
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
    n_real = float(real_positions.numel())
    seq_len = float(max(int(attention_mask.numel()), 1))

    global _LAST_FEATURE_NAMES

    features: list[torch.Tensor] = []
    diagnostics: list[torch.Tensor] = [
        torch.tensor(
            [n_real / max(seq_len, 512.0)],
            device=hs.device,
            dtype=torch.float32,
        )
    ]

    # Windows test where hallucination signal lives in Qwen2.5-0.5B:
    # 10-20 broad middle-to-late, 13-18 narrower late-middle, 21-24 output-near.
    for block_type, start_layer, end_layer in _selected_blocks(AGGREGATION_MODE):
        if block_type == "final":
            sequence_repr = _final_layer_repr(hs)
        else:
            assert start_layer is not None and end_layer is not None
            sequence_repr = _layer_window_repr(hs, start_layer, end_layer)

        pooled, real_sequence = _pool_tail_tokens(sequence_repr, real_positions)
        features.append(pooled)

        norm_tail = real_sequence[-min(16, real_sequence.size(0)) :].norm(
            p=2,
            dim=1,
        )
        diagnostics.append(
            torch.stack(
                [
                    norm_tail.mean(),
                    norm_tail.std(unbiased=False),
                ]
            ).to(dtype=torch.float32)
        )

    spectral_features, spectral_names = _spectral_features(hs, real_positions)
    out = torch.cat(features + diagnostics + [spectral_features], dim=0)
    out = torch.nan_to_num(
        out.to(dtype=torch.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).cpu()
    n_hidden = out.numel() - len(spectral_names)
    _LAST_FEATURE_NAMES = [f"hidden_{i}" for i in range(n_hidden)] + spectral_names
    return out


def get_last_feature_names() -> list[str]:
    """Return feature names from the most recent ``aggregate`` call."""
    return list(_LAST_FEATURE_NAMES)


def build_feature_names(feature_dim: int) -> list[str]:
    """Build generic hidden names plus explicit spectral feature names."""
    spectral_names = _spectral_feature_names(SPECTRAL_MODE)
    n_hidden = feature_dim - len(spectral_names)
    if n_hidden < 0:
        raise ValueError(
            f"feature_dim={feature_dim} is smaller than the spectral feature "
            f"count {len(spectral_names)}."
        )
    return [f"hidden_{i}" for i in range(n_hidden)] + spectral_names


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
