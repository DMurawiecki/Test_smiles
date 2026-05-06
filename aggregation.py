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


AGGREGATION_MODE = os.getenv(
    "AGGREGATION_MODE", "mean_10_20_plus_21_24_plus_final"
)

SUPPORTED_AGGREGATION_MODES = (
    "final_only",
    "mean_10_20",
    "mean_13_18",
    "mean_21_24",
    "mean_10_20_plus_21_24",
    "mean_10_20_plus_final",
    "mean_10_20_plus_21_24_plus_final",
)


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

    return torch.cat(features + diagnostics, dim=0).to(dtype=torch.float32).cpu()


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
