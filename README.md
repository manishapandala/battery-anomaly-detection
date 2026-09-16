# Battery CT Anomaly Detection

A geometry-preserving computer-vision pipeline for cylindrical battery CT images. The project standardizes radial scan orientation, preserves native pixel geometry, combines radial and axial evidence, extracts interpretable descriptors, ranks unusual cells, and produces patch-level occlusion maps.

> **Project status:** the preprocessing, feature extraction, synthetic validation, tests, and explainability workflow are complete. The repository does not contain proprietary CT data, and the synthetic metrics below are pipeline checks—not production accuracy claims.

## Why geometry preservation matters

Small differences in can circularity, jelly-roll spacing, eccentricity, tab position, and electrode overhang may span only a few pixels. Resizing can blur or distort those signals. This pipeline therefore uses rotation, native-width cropping, and symmetric padding, but never resizes source images.

## Pipeline

1. **Can and tab localization** — estimate the outer can and detect the bright collector tab within an annular search region.
2. **Orientation normalization** — rotate the radial slice so the tab is at 12 o'clock while keeping the original resolution.
3. **Multi-view construction** — stack the aligned radial slice with the native-width top and bottom 20% axial crops.
4. **Interpretable features** — compute center offset, symmetry error, edge density, ring count, spacing variation, intensity variation, and residual tab-alignment error.
5. **Anomaly ranking** — fit an Isolation Forest on reference-cell features and assign higher scores to unusual geometry.
6. **Occlusion sensitivity** — hide one patch at a time and measure how each patch changes the anomaly score.

## Reproducible synthetic validation

Run the public demo without any private CT files:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python run_demo.py
pytest -q
```

The deterministic demo creates 24 reference cells and eight cells with injected geometric abnormalities.

| Check | Result |
|---|---:|
| Synthetic cells processed | 32 |
| Collector tabs detected | 100% |
| Median rotation error | 0.02° |
| Injected anomalies in top eight scores | 8/8 |
| Automated tests | 11 passing |

These results verify the implementation against known synthetic geometry. They do not estimate sensitivity, specificity, or generalization on real manufacturing data.

## Outputs

- `assets/alignment_demo.png` — before/after orientation normalization
- `assets/combined_input_demo.png` — radial and axial multi-view input
- `assets/anomaly_scores.png` — sorted synthetic anomaly scores
- `assets/occlusion_demo.png` — patch-level sensitivity map
- `demo_results.csv` — features, alignment metadata, scores, and flags
- `demo_summary.json` — compact validation summary

## Real-data integration

For real CT scans, load the native-resolution radial and axial images, then call:

```python
import cv2
from battery_pipeline import build_multiview_input, extract_geometry_features

radial = cv2.imread("radial.png", cv2.IMREAD_GRAYSCALE)
axial = cv2.imread("axial.png", cv2.IMREAD_GRAYSCALE)

combined, alignment = build_multiview_input(radial, axial)
features = extract_geometry_features(alignment.image)
```

Before making manufacturing claims, the next evaluation stage should use cell-level train/validation/test splits, compare results across battery batches, and report precision-recall metrics with error analysis. The full-resolution multi-view output is also ready for a DINOv2 embedding experiment once approved data is available.

## Repository structure

```text
.
├── battery_pipeline.py       # preprocessing, features, and explainability
├── run_demo.py               # deterministic synthetic evaluation
├── tests/test_pipeline.py    # unit and geometry tests
├── assets/                   # generated project figures
├── demo_results.csv          # synthetic result table
├── demo_summary.json         # synthetic validation summary
├── index.html                # GitHub Pages project page
└── requirements.txt
```

## Limitations

- Synthetic CT-like images are used because the source dataset is not public.
- Isolation Forest operates on interpretable geometry descriptors, not DINOv2 embeddings.
- Occlusion maps explain score sensitivity; they do not prove causal defect localization.
- Thresholds and Hough parameters require validation on each scanner and acquisition protocol.

## Author

**Manisha Pandala** — M.S. Data Science student at Arizona State University and AI/ML enthusiast.

