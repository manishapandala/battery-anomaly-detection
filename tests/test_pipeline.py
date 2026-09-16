import numpy as np
import pytest

from battery_pipeline import (
    FEATURE_NAMES,
    align_tab_to_top,
    build_multiview_input,
    crop_axial_overhangs,
    detect_tab_angle,
    extract_geometry_features,
    generate_synthetic_cell,
    pad_to_width,
)


def angular_error(value: float, target: float) -> float:
    return abs((value - target + 180) % 360 - 180)


@pytest.mark.parametrize("seed", [3, 7, 11, 19])
def test_tab_alignment_moves_tab_to_top(seed):
    radial, _, _ = generate_synthetic_cell(seed)
    result = align_tab_to_top(radial)
    assert result.tab_found

    center_x, center_y = result.can_center
    angle, position = detect_tab_angle(result.image, center_x, center_y, result.can_radius)
    assert position is not None
    assert angular_error(angle, 0.0) < 8.0


def test_axial_crops_preserve_native_width():
    axial = np.arange(200 * 123, dtype=np.uint8).reshape(200, 123)
    top, bottom = crop_axial_overhangs(axial, 0.2)
    assert top.shape == (40, 123)
    assert bottom.shape == (40, 123)
    np.testing.assert_array_equal(top, axial[:40])
    np.testing.assert_array_equal(bottom, axial[-40:])


def test_padding_does_not_resize_original_pixels():
    image = np.arange(30, dtype=np.uint8).reshape(5, 6)
    padded = pad_to_width(image, 10)
    assert padded.shape == (5, 10)
    np.testing.assert_array_equal(padded[:, 2:8], image)


def test_multiview_input_uses_expected_geometry():
    radial, axial, _ = generate_synthetic_cell(21, size=256)
    combined, alignment = build_multiview_input(radial, axial, crop_fraction=0.2, gap_pixels=6)
    expected_crop_height = int(round(axial.shape[0] * 0.2))
    assert alignment.image.shape == radial.shape
    assert combined.shape[1] == max(radial.shape[1], axial.shape[1])
    assert combined.shape[0] == radial.shape[0] + 2 * expected_crop_height + 12


def test_geometry_features_are_finite_and_named():
    radial, _, _ = generate_synthetic_cell(42, anomaly=True)
    aligned = align_tab_to_top(radial).image
    features = extract_geometry_features(aligned)
    assert features.shape == (len(FEATURE_NAMES),)
    assert np.all(np.isfinite(features))


@pytest.mark.parametrize("fraction", [0, -0.1, 0.51])
def test_invalid_crop_fraction_is_rejected(fraction):
    with pytest.raises(ValueError):
        crop_axial_overhangs(np.zeros((100, 40), dtype=np.uint8), fraction)
