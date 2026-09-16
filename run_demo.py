"""Run a reproducible, synthetic demonstration of the battery pipeline."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/battery-matplotlib")

import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from battery_pipeline import (
    FEATURE_NAMES,
    align_tab_to_top,
    build_multiview_input,
    extract_geometry_features,
    generate_synthetic_cell,
    occlusion_sensitivity,
)


ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "assets"
RESULTS = ROOT / "demo_results.csv"
SUMMARY = ROOT / "demo_summary.json"


def anomaly_score(model: IsolationForest, scaler: StandardScaler, image: np.ndarray) -> float:
    features = extract_geometry_features(image)
    return float(-model.decision_function(scaler.transform(features.reshape(1, -1)))[0])


def save_alignment_figure(original: np.ndarray, aligned: np.ndarray, angle: float) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2), facecolor="#020d2a")
    for axis, image, title in zip(
        axes,
        (original, aligned),
        (f"Original orientation ({angle:+.1f}°)", "Tab aligned to 12 o'clock"),
    ):
        axis.imshow(image, cmap="gray", vmin=0, vmax=255)
        axis.set_title(title, color="white", fontsize=11, pad=10)
        axis.axis("off")
    fig.suptitle("Orientation normalization without resizing", color="#54dccf", weight="bold")
    fig.tight_layout()
    fig.savefig(ASSETS / "alignment_demo.png", dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)


def save_combined_figure(combined: np.ndarray) -> None:
    fig, axis = plt.subplots(figsize=(5, 8), facecolor="#020d2a")
    axis.imshow(combined, cmap="gray", vmin=0, vmax=255)
    axis.set_title("Radial + axial top + axial bottom", color="#54dccf", weight="bold", pad=12)
    axis.axis("off")
    fig.tight_layout()
    fig.savefig(ASSETS / "combined_input_demo.png", dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)


def save_score_figure(scores: np.ndarray, labels: np.ndarray) -> None:
    order = np.argsort(scores)
    colors = np.where(labels[order] == "synthetic anomaly", "#f59e8b", "#54dccf")
    fig, axis = plt.subplots(figsize=(10, 4.8), facecolor="#020d2a")
    axis.set_facecolor("#07183c")
    axis.bar(np.arange(len(scores)), scores[order], color=colors, width=0.78)
    axis.axhline(0, color="#8fa2c0", linewidth=0.8, alpha=0.6)
    axis.set_xlabel("Synthetic cell, sorted by anomaly score", color="#b4c0d4")
    axis.set_ylabel("Isolation Forest anomaly score", color="#b4c0d4")
    axis.set_title("Pipeline smoke test on labeled synthetic geometry", color="white", weight="bold")
    axis.tick_params(colors="#8fa2c0")
    for spine in axis.spines.values():
        spine.set_color("#26466f")
    fig.tight_layout()
    fig.savefig(ASSETS / "anomaly_scores.png", dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)


def save_occlusion_figure(image: np.ndarray, heat: np.ndarray) -> None:
    positive = np.maximum(heat, 0)
    scale = np.percentile(positive[positive > 0], 95) if np.any(positive > 0) else 1.0
    normalized = np.clip(positive / max(scale, 1e-9), 0, 1)

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2), facecolor="#020d2a")
    axes[0].imshow(image, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title("Synthetic anomaly", color="white")
    axes[1].imshow(image, cmap="gray", vmin=0, vmax=255)
    overlay = axes[1].imshow(normalized, cmap="magma", alpha=normalized * 0.78, vmin=0, vmax=1)
    axes[1].set_title("Occlusion sensitivity", color="#54dccf")
    for axis in axes:
        axis.axis("off")
    fig.colorbar(overlay, ax=axes[1], fraction=0.046, pad=0.04, label="score contribution")
    fig.suptitle("Which regions influence the anomaly score?", color="white", weight="bold")
    fig.tight_layout()
    fig.savefig(ASSETS / "occlusion_demo.png", dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    images: list[np.ndarray] = []
    features: list[np.ndarray] = []

    for index in range(32):
        is_anomaly = index >= 24
        radial, axial, true_angle = generate_synthetic_cell(1000 + index, anomaly=is_anomaly)
        combined, alignment = build_multiview_input(radial, axial)
        images.append(alignment.image)
        features.append(extract_geometry_features(alignment.image))
        records.append(
            {
                "cell_id": f"synthetic_{index + 1:02d}",
                "label": "synthetic anomaly" if is_anomaly else "synthetic reference",
                "true_tab_angle": round(true_angle, 3),
                "detected_rotation": round(alignment.rotation_degrees, 3),
                "tab_found": alignment.tab_found,
            }
        )

        if index == 2:
            save_alignment_figure(radial, alignment.image, alignment.rotation_degrees)
            save_combined_figure(combined)

    matrix = np.vstack(features)
    labels = np.array([record["label"] for record in records])
    reference_mask = labels == "synthetic reference"
    scaler = StandardScaler().fit(matrix[reference_mask])
    model = IsolationForest(n_estimators=250, contamination=0.08, random_state=17)
    model.fit(scaler.transform(matrix[reference_mask]))
    scores = -model.decision_function(scaler.transform(matrix))
    predictions = model.predict(scaler.transform(matrix))

    for record, feature_values, score, prediction in zip(records, matrix, scores, predictions):
        record["anomaly_score"] = round(float(score), 6)
        record["flagged"] = bool(prediction == -1)
        record.update({name: round(float(value), 6) for name, value in zip(FEATURE_NAMES, feature_values)})

    ranked = np.argsort(scores)[::-1]
    top_eight_precision = float(np.mean(labels[ranked[:8]] == "synthetic anomaly"))
    tab_detection_rate = float(np.mean([record["tab_found"] for record in records]))
    rotation_error = np.array(
        [
            min(abs(float(record["detected_rotation"]) - float(record["true_tab_angle"])),
                360 - abs(float(record["detected_rotation"]) - float(record["true_tab_angle"])))
            for record in records
        ]
    )

    with RESULTS.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)

    summary = {
        "scope": "synthetic pipeline validation only",
        "samples": len(records),
        "synthetic_reference_samples": int(np.sum(reference_mask)),
        "synthetic_anomaly_samples": int(np.sum(~reference_mask)),
        "tab_detection_rate": round(tab_detection_rate, 4),
        "median_rotation_error_degrees": round(float(np.median(rotation_error)), 3),
        "top_8_anomaly_precision": round(top_eight_precision, 4),
    }
    SUMMARY.write_text(json.dumps(summary, indent=2) + "\n")

    save_score_figure(scores, labels)
    target_index = int(ranked[0])
    target = images[target_index]
    score_function = lambda image: anomaly_score(model, scaler, image)
    heat = occlusion_sensitivity(target, score_function, patch_size=64, stride=32)
    save_occlusion_figure(target, heat)

    print(json.dumps(summary, indent=2))
    print(f"Wrote {RESULTS.relative_to(ROOT)} and {SUMMARY.relative_to(ROOT)}")
    print(f"Wrote {len(list(ASSETS.glob('*.png')))} figures to {ASSETS.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
