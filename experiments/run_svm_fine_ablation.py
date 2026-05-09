"""
Fine sweep around the best PCA/probe result.

Keeps feature extraction fixed to mean_21_24 + SPECTRAL_MODE=none and varies
only PCA_N_COMPONENTS, SVM_C, and CLASS_WEIGHT for the linear SVM probe.
"""

from __future__ import annotations

import os
import sys
import time
import traceback
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_pca_probe_ablation import (  # noqa: E402
    AGGREGATION_MODE,
    RESULTS_DIR,
    SPECTRAL_MODE,
    THRESHOLD_METRIC,
    _extract_current_features,
    _run_probe_evaluation,
)


OUTPUT_FILE = RESULTS_DIR / "svm_fine_ablation.csv"
SORTED_OUTPUT_FILE = RESULTS_DIR / "svm_fine_ablation_sorted.csv"

PROBE_TYPE = "svm_linear"
PCA_COMPONENTS = ("48", "56", "64", "72", "80")
SVM_C_VALUES = ("0.03", "0.05", "0.1", "0.2", "0.3", "1.0")
CLASS_WEIGHTS = ("none", "balanced")


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    X, y, _, splits = _extract_current_features()
    print(f"Feature matrix : {X.shape}")
    print(f"Splits         : {len(splits)}")
    print(f"Probe          : {PROBE_TYPE}")
    print(f"PCA grid       : {', '.join(PCA_COMPONENTS)}")
    print(f"SVM_C grid     : {', '.join(SVM_C_VALUES)}")
    print(f"CLASS_WEIGHT   : {', '.join(CLASS_WEIGHTS)}")

    rows: list[dict] = []
    for pca_components in PCA_COMPONENTS:
        for svm_c in SVM_C_VALUES:
            for class_weight in CLASS_WEIGHTS:
                os.environ["PROBE_TYPE"] = PROBE_TYPE
                os.environ["PCA_N_COMPONENTS"] = pca_components
                os.environ["SVM_C"] = svm_c
                os.environ["CLASS_WEIGHT"] = class_weight
                os.environ["THRESHOLD_METRIC"] = THRESHOLD_METRIC

                t0 = time.time()
                row = {
                    "aggregation_mode": AGGREGATION_MODE,
                    "spectral_mode": SPECTRAL_MODE,
                    "probe_type": PROBE_TYPE,
                    "pca_components": pca_components,
                    "svm_c": svm_c,
                    "class_weight": class_weight,
                    "threshold_metric": THRESHOLD_METRIC,
                    "val_accuracy": float("nan"),
                    "val_f1": float("nan"),
                    "val_auroc": float("nan"),
                    "test_accuracy": float("nan"),
                    "test_f1": float("nan"),
                    "test_auroc": float("nan"),
                    "best_threshold": float("nan"),
                    "runtime_seconds": float("nan"),
                    "status": "failed",
                    "error_message": "",
                }

                print(
                    f"\nRun: probe={PROBE_TYPE}  PCA_N_COMPONENTS={pca_components}  "
                    f"SVM_C={svm_c}  CLASS_WEIGHT={class_weight}"
                )
                try:
                    metrics = _run_probe_evaluation(X, y, splits)
                    row.update(metrics)
                    row["status"] = "ok"
                    print(
                        f"  val_acc={metrics['val_accuracy']:.4f}  "
                        f"val_auroc={metrics['val_auroc']:.4f}  "
                        f"test_acc={metrics['test_accuracy']:.4f}  "
                        f"test_auroc={metrics['test_auroc']:.4f}  "
                        f"threshold={metrics['best_threshold']:.4f}"
                    )
                except Exception as exc:  # noqa: BLE001 - keep sweep robust.
                    row["error_message"] = "".join(
                        traceback.format_exception_only(type(exc), exc)
                    ).strip()
                    print(f"  FAILED: {row['error_message']}")
                finally:
                    row["runtime_seconds"] = time.time() - t0
                    rows.append(row)
                    pd.DataFrame(rows).to_csv(OUTPUT_FILE, index=False)

    results = pd.DataFrame(rows)
    sorted_results = results.sort_values(
        by=["val_accuracy", "test_accuracy"],
        ascending=False,
        na_position="last",
    )
    results.to_csv(OUTPUT_FILE, index=False)
    sorted_results.to_csv(SORTED_OUTPUT_FILE, index=False)

    print("\nSVM fine ablation summary")
    print(
        sorted_results[
            [
                "pca_components",
                "svm_c",
                "class_weight",
                "val_accuracy",
                "val_f1",
                "val_auroc",
                "test_accuracy",
                "test_f1",
                "test_auroc",
                "best_threshold",
                "status",
            ]
        ].to_string(index=False, float_format=lambda value: f"{value:.4f}")
    )
    print(f"\nSaved: {OUTPUT_FILE}")
    print(f"Saved: {SORTED_OUTPUT_FILE}")


if __name__ == "__main__":
    main()
