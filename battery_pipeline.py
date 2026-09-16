"""Geometry-preserving preprocessing for cylindrical battery CT images.

The module deliberately avoids resizing. Pixel geometry can carry the signal
for layer spacing, can eccentricity, and electrode overhang, so inputs are
rotated, cropped, or padded only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import cv2
import numpy as np
from scipy.signal import find_peaks


@dataclass(frozen=True)
class AlignmentResult:
    image: np.ndarray
    rotation_degrees: float
    tab_found: bool
    can_center: tuple[int, int]
    can_radius: int


def detect_can_circle(gray: np.ndarray) -> tuple[int, int, int]:
    """Estimate the outer can circle with gradient-based Hough detection."""
    if gray.ndim != 2:
        raise ValueError("detect_can_circle expects a grayscale image")

    height, width = gray.shape
    # Prefer the bright, continuous outer wall when it is visible. Using its
    # enclosing contour prevents an inner jelly-roll ring from winning the
    # Hough vote simply because it has a sharper local edge.
    bright_wall = np.zeros_like(gray, dtype=np.uint8)
    bright_wall[gray >= 170] = 255
    bright_wall = cv2.morphologyEx(
        bright_wall,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
    )
    contours, _ = cv2.findContours(bright_wall, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    wall_candidates: list[tuple[float, float, float]] = []
    for contour in contours:
        (cx, cy), candidate_radius = cv2.minEnclosingCircle(contour)
        if min(height, width) * 0.30 <= candidate_radius <= min(height, width) * 0.49:
            center_distance = np.hypot(cx - width / 2, cy - height / 2)
            if center_distance <= min(height, width) * 0.16:
                wall_candidates.append((cx, cy, candidate_radius))

    if wall_candidates:
        cx, cy, candidate_radius = max(wall_candidates, key=lambda candidate: candidate[2])
        return int(round(cx)), int(round(cy)), int(round(candidate_radius))

    blurred = cv2.GaussianBlur(gray, (5, 5), 1.2)
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=max(20, min(height, width) // 3),
        param1=90,
        param2=24,
        minRadius=int(min(height, width) * 0.28),
        maxRadius=int(min(height, width) * 0.49),
    )

    if circles is not None:
        candidates = np.round(circles[0]).astype(int)
        image_center = np.array([width / 2, height / 2])
        best = min(
            candidates,
            key=lambda circle: np.linalg.norm(circle[:2] - image_center) - 0.05 * circle[2],
        )
        return int(best[0]), int(best[1]), int(best[2])

    return width // 2, height // 2, int(min(height, width) * 0.42)


def detect_tab_angle(
    gray: np.ndarray,
    center_x: int,
    center_y: int,
    radius: int,
) -> tuple[float, tuple[int, int] | None]:
    """Locate a bright current-collector tab in the outer jelly-roll annulus.

    The returned angle is measured clockwise from 12 o'clock. The outer can
    wall is excluded so that it cannot dominate the bright-pixel mask.
    """
    yy, xx = np.ogrid[: gray.shape[0], : gray.shape[1]]
    distance = np.sqrt((xx - center_x) ** 2 + (yy - center_y) ** 2)
    annulus = (distance >= radius * 0.58) & (distance <= radius * 0.90)

    pixels = gray[annulus]
    if pixels.size == 0:
        return 0.0, None

    threshold = max(175.0, float(np.percentile(pixels, 99.2)))
    mask = np.zeros_like(gray, dtype=np.uint8)
    mask[annulus & (gray >= threshold)] = 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
    candidates: list[tuple[float, int]] = []
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < 18:
            continue
        cx, cy = centroids[label]
        radial_position = np.hypot(cx - center_x, cy - center_y) / max(radius, 1)
        if 0.56 <= radial_position <= 0.92:
            candidates.append((area * (1.0 - abs(radial_position - 0.78)), label))

    if not candidates:
        return 0.0, None

    _, best_label = max(candidates)
    tab_x, tab_y = centroids[best_label]
    dx = float(tab_x - center_x)
    dy = float(tab_y - center_y)
    angle = float(np.degrees(np.arctan2(dx, -dy)))
    return angle, (int(round(tab_x)), int(round(tab_y)))


def align_tab_to_top(gray: np.ndarray) -> AlignmentResult:
    """Rotate a CT slice so its detected tab is positioned at 12 o'clock."""
    center_x, center_y, radius = detect_can_circle(gray)
    angle, tab_position = detect_tab_angle(gray, center_x, center_y, radius)

    if tab_position is None:
        return AlignmentResult(gray.copy(), 0.0, False, (center_x, center_y), radius)

    matrix = cv2.getRotationMatrix2D((center_x, center_y), angle, 1.0)
    rotated = cv2.warpAffine(
        gray,
        matrix,
        (gray.shape[1], gray.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return AlignmentResult(rotated, angle, True, (center_x, center_y), radius)


def crop_axial_overhangs(
    axial: np.ndarray,
    crop_fraction: float = 0.20,
) -> tuple[np.ndarray, np.ndarray]:
    """Return native-width top and bottom axial crops without resizing."""
    if axial.ndim != 2:
        raise ValueError("crop_axial_overhangs expects a grayscale image")
    if not 0 < crop_fraction <= 0.5:
        raise ValueError("crop_fraction must be greater than 0 and at most 0.5")

    crop_height = max(1, int(round(axial.shape[0] * crop_fraction)))
    return axial[:crop_height].copy(), axial[-crop_height:].copy()


def pad_to_width(image: np.ndarray, target_width: int) -> np.ndarray:
    """Symmetrically pad an image to a target width."""
    if image.shape[1] > target_width:
        raise ValueError("target_width cannot be smaller than image width")
    total = target_width - image.shape[1]
    left = total // 2
    right = total - left
    return cv2.copyMakeBorder(image, 0, 0, left, right, cv2.BORDER_CONSTANT, value=0)


def build_multiview_input(
    radial: np.ndarray,
    axial: np.ndarray,
    crop_fraction: float = 0.20,
    gap_pixels: int = 8,
) -> tuple[np.ndarray, AlignmentResult]:
    """Stack aligned radial, axial-top, and axial-bottom views without scaling."""
    aligned = align_tab_to_top(radial)
    top, bottom = crop_axial_overhangs(axial, crop_fraction)
    width = max(aligned.image.shape[1], top.shape[1], bottom.shape[1])
    views = [pad_to_width(view, width) for view in (aligned.image, top, bottom)]

    if gap_pixels < 0:
        raise ValueError("gap_pixels must be non-negative")
    if gap_pixels == 0:
        return np.vstack(views), aligned

    gap = np.zeros((gap_pixels, width), dtype=radial.dtype)
    return np.vstack((views[0], gap, views[1], gap, views[2])), aligned


def _radial_profile(gray: np.ndarray, center: tuple[int, int], radius: int) -> np.ndarray:
    yy, xx = np.ogrid[: gray.shape[0], : gray.shape[1]]
    distances = np.sqrt((xx - center[0]) ** 2 + (yy - center[1]) ** 2).astype(int)
    valid = distances <= radius
    sums = np.bincount(distances[valid].ravel(), weights=gray[valid].ravel(), minlength=radius + 1)
    counts = np.bincount(distances[valid].ravel(), minlength=radius + 1)
    return sums[: radius + 1] / np.maximum(counts[: radius + 1], 1)


FEATURE_NAMES = (
    "center_offset",
    "radius_fraction",
    "symmetry_mae",
    "edge_density",
    "ring_count",
    "ring_spacing_cv",
    "inner_intensity_std",
    "tab_alignment_error",
)


def extract_geometry_features(gray: np.ndarray) -> np.ndarray:
    """Extract interpretable geometry descriptors from one aligned radial slice."""
    center_x, center_y, radius = detect_can_circle(gray)
    height, width = gray.shape
    image_center = np.array([width / 2, height / 2])
    center_offset = np.linalg.norm(np.array([center_x, center_y]) - image_center) / max(radius, 1)

    rotated_180 = cv2.rotate(gray, cv2.ROTATE_180)
    yy, xx = np.ogrid[:height, :width]
    inner_mask = (xx - center_x) ** 2 + (yy - center_y) ** 2 <= (radius * 0.88) ** 2
    symmetry_mae = np.mean(np.abs(gray.astype(float) - rotated_180.astype(float))[inner_mask]) / 255.0

    edges = cv2.Canny(gray, 40, 110)
    edge_density = float(np.mean(edges[inner_mask] > 0))

    profile = _radial_profile(gray, (center_x, center_y), int(radius * 0.90))
    peaks, _ = find_peaks(profile, prominence=5.0, distance=4)
    spacing = np.diff(peaks)
    spacing_cv = float(np.std(spacing) / np.mean(spacing)) if spacing.size > 1 and np.mean(spacing) else 0.0

    tab_angle, tab = detect_tab_angle(gray, center_x, center_y, radius)
    tab_error = abs(tab_angle) / 180.0 if tab is not None else 1.0

    features = np.array(
        [
            center_offset,
            radius / max(min(height, width), 1),
            symmetry_mae,
            edge_density,
            len(peaks) / 30.0,
            spacing_cv,
            float(np.std(gray[inner_mask])) / 255.0,
            tab_error,
        ],
        dtype=np.float64,
    )
    return features


def occlusion_sensitivity(
    image: np.ndarray,
    score_function: Callable[[np.ndarray], float],
    patch_size: int = 64,
    stride: int = 32,
) -> np.ndarray:
    """Measure how occluding each patch changes an anomaly score."""
    if patch_size <= 0 or stride <= 0:
        raise ValueError("patch_size and stride must be positive")

    baseline = float(score_function(image))
    heat = np.zeros(image.shape, dtype=np.float64)
    coverage = np.zeros(image.shape, dtype=np.float64)
    fill = int(np.median(image))

    for top in range(0, max(1, image.shape[0] - patch_size + 1), stride):
        for left in range(0, max(1, image.shape[1] - patch_size + 1), stride):
            bottom = min(top + patch_size, image.shape[0])
            right = min(left + patch_size, image.shape[1])
            occluded = image.copy()
            occluded[top:bottom, left:right] = fill
            contribution = baseline - float(score_function(occluded))
            heat[top:bottom, left:right] += contribution
            coverage[top:bottom, left:right] += 1

    return heat / np.maximum(coverage, 1)


def generate_synthetic_cell(
    seed: int,
    anomaly: bool = False,
    size: int = 384,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Generate a deterministic CT-like radial/axial pair for public demos.

    Synthetic images validate data flow and tests only; they are not a proxy
    for performance on production CT scans.
    """
    rng = np.random.default_rng(seed)
    center = size // 2
    offset_x = int(rng.integers(-3, 4))
    offset_y = int(rng.integers(-3, 4))
    if anomaly:
        offset_x += int(rng.choice((-15, 15)))
        offset_y += int(rng.choice((-10, 10)))

    cx, cy = center + offset_x, center + offset_y
    yy, xx = np.ogrid[:size, :size]
    stretch_x = 1.0 + (0.08 if anomaly else rng.normal(0, 0.006))
    distance = np.sqrt(((xx - cx) / stretch_x) ** 2 + (yy - cy) ** 2)
    radius = int(size * 0.41)
    radial = np.zeros((size, size), dtype=np.float64)
    radial[(distance >= radius - 5) & (distance <= radius)] = 205

    ring_radii = list(range(radius - 10, 24, -9))
    if anomaly:
        ring_radii.pop(5)
        ring_radii[7] += 5
    for index, ring_radius in enumerate(ring_radii):
        intensity = 122 if index % 2 == 0 else 76
        radial[np.abs(distance - ring_radius) < 1.7] = intensity

    radial[distance <= 20] = 16
    if anomaly:
        defect_angle = np.arctan2(yy - cy, xx - cx)
        missing_arc = (distance > radius * 0.55) & (distance < radius * 0.82) & (defect_angle > 0.3) & (defect_angle < 0.75)
        radial[missing_arc] *= 0.25

    tab_angle = float(rng.uniform(-170, 170))
    angle_radians = np.deg2rad(tab_angle)
    tab_x = cx + radius * 0.76 * np.sin(angle_radians)
    tab_y = cy - radius * 0.76 * np.cos(angle_radians)
    tab = ((xx - tab_x) / 13) ** 2 + ((yy - tab_y) / 5) ** 2 <= 1
    radial[tab] = 242
    radial += rng.normal(0, 3.0, radial.shape)
    radial = np.clip(radial, 0, 255).astype(np.uint8)

    axial_height = 460
    axial = np.zeros((axial_height, size), dtype=np.float64)
    layer_center = size // 2 + (12 if anomaly else 0)
    for x_position in range(layer_center - 135, layer_center + 136, 9):
        top = 30 + int(rng.integers(-2, 3))
        bottom = axial_height - 30 + int(rng.integers(-2, 3))
        if anomaly and x_position > layer_center + 70:
            bottom -= 20
        axial[top:bottom, max(0, x_position - 1) : min(size, x_position + 2)] = 115
    axial += rng.normal(8, 2.0, axial.shape)
    axial = np.clip(axial, 0, 255).astype(np.uint8)
    return radial, axial, tab_angle
