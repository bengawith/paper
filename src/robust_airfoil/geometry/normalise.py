from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class NormalisedGeometry:
    coordinates: np.ndarray
    upper: np.ndarray
    lower: np.ndarray
    chord: float
    rotation_radians: float


def resample_surfaces_to_common_x(
    upper: np.ndarray,
    lower: np.ndarray,
    x_grid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate displaced surfaces onto one prescribed chordwise grid."""

    def interpolate(surface: np.ndarray) -> np.ndarray:
        if surface.ndim != 2 or surface.shape[1] != 2:
            raise ValueError("Expected Nx2 surface coordinates")
        ordered = surface[np.argsort(surface[:, 0])]
        unique_x, inverse = np.unique(np.round(ordered[:, 0], 12), return_inverse=True)
        y = np.asarray([ordered[inverse == index, 1].mean() for index in range(len(unique_x))])
        if x_grid[0] < unique_x[0] - 1e-10 or x_grid[-1] > unique_x[-1] + 1e-10:
            raise ValueError("Common x grid falls outside displaced surface extent")
        return np.column_stack([x_grid, np.interp(x_grid, unique_x, y)])

    return interpolate(upper), interpolate(lower)


def normalise_geometry(points: np.ndarray, cosine_points: int = 201) -> NormalisedGeometry:
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("Expected Nx2 coordinate array")
    le_index = int(np.argmin(points[:, 0]))
    if le_index == 0 or le_index == len(points) - 1:
        midpoint = np.median(points[:, 1])
        upper_raw = points[(points[:, 1] >= midpoint)][np.argsort(points[(points[:, 1] >= midpoint), 0])]
        lower_raw = points[(points[:, 1] < midpoint)][np.argsort(points[(points[:, 1] < midpoint), 0])]
        if len(upper_raw) < 3 or len(lower_raw) < 3:
            raise ValueError("Unable to split upper and lower surfaces")
    else:
        first, second = points[: le_index + 1], points[le_index:]
        upper_raw, lower_raw = (first, second) if np.nanmean(first[:, 1]) >= np.nanmean(second[:, 1]) else (second, first)
    le = points[le_index]
    te_candidates = np.vstack([upper_raw[np.argmax(upper_raw[:, 0])], lower_raw[np.argmax(lower_raw[:, 0])]])
    te = te_candidates.mean(axis=0)
    vector = te - le
    chord = float(np.linalg.norm(vector))
    if not np.isfinite(chord) or chord <= 0:
        raise ValueError("Invalid chord")
    angle = float(np.arctan2(vector[1], vector[0]))
    rotation = np.array([[np.cos(-angle), -np.sin(-angle)], [np.sin(-angle), np.cos(-angle)]])

    def transform(surface: np.ndarray) -> np.ndarray:
        transformed = (surface - le) @ rotation.T / chord
        transformed = transformed[np.argsort(transformed[:, 0])]
        unique_x, inverse = np.unique(np.round(transformed[:, 0], 12), return_inverse=True)
        y = np.array([transformed[inverse == i, 1].mean() for i in range(len(unique_x))])
        x_grid = (1 - np.cos(np.linspace(0, np.pi, cosine_points))) / 2
        return np.column_stack([x_grid, np.interp(x_grid, unique_x, y)])

    upper, lower = transform(upper_raw), transform(lower_raw)
    if np.nanmean(upper[:, 1] - lower[:, 1]) < 0:
        upper, lower = lower, upper
    coordinates = np.vstack([upper[::-1], lower[1:]])
    return NormalisedGeometry(coordinates, upper, lower, chord, angle)
