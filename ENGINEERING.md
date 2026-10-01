# Engineering Review

This document records the architecture and review contract outside conversational history. It should change whenever preprocessing, geometry features, ranking, or published evaluation claims change.

## Architecture

| Layer | Responsibility | File |
|---|---|---|
| Image domain | Generate fixtures and detect can/tab geometry | `battery_pipeline.py` |
| Preprocessing | Align without resizing and construct multi-view input | `battery_pipeline.py` |
| Features | Extract interpretable geometry descriptors | `battery_pipeline.py` |
| Ranking | Scale features and fit Isolation Forest | `run_demo.py` |
| Explanation | Measure patch-level score sensitivity | `battery_pipeline.py` |
| Evaluation | Reproduce metrics and visual artifacts | `run_demo.py` |

Feature extraction is independent of model fitting. Native-resolution geometry is preserved through preprocessing.

## Verification contract

```bash
python -m ruff check .
python -m compileall -q .
python -m pytest -q
python run_demo.py
```

Tests cover synthetic generation, tab detection, rotation, non-resizing alignment, multi-view dimensions, feature shape, and occlusion behavior.

## Change-safety checklist

- Review every resize, crop, padding, and rotation change for geometry distortion.
- Require a failing fixture before changing thresholds or Hough parameters.
- Keep proprietary scans, labels, and model artifacts out of Git.
- Re-run numeric tests and inspect generated figures after image-processing changes.
- Compare runtime before adding operations to the per-image path.
- Never relabel synthetic checks as real-world sensitivity or specificity.
- Separate preprocessing, model, and presentation commits when practical.

## Known boundaries

Synthetic CT-like images validate implementation behavior but not scanner variation or manufacturing performance. Isolation Forest scores are prioritization signals, and occlusion sensitivity describes score response rather than causal defect localization.
