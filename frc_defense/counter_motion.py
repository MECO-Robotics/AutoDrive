"""Explicit nominal CAD surface-motion conventions, not hardware calibration."""
from __future__ import annotations

import copy
import math
from collections.abc import Mapping, Sequence
from numbers import Real

import numpy as np


DRUM_NAME = 'GE-26408: Counter Roller Tube'


def configure_drum_motion(manifest: Mapping, *, source_path: Sequence[str], factor: int = 1) -> dict:
    """Return a private manifest with one source occurrence's omega multiplied.

    The full occurrence path resolves identity independently of compiled triangle
    or part ordering. The default preserves the existing nominal convention.
    A negative factor expresses the reviewed nominal functional interpretation;
    it does not establish actual motor polarity, speed, or as-built geometry.
    """
    if type(factor) is not int or factor not in (-1, 1):
        raise ValueError('Drum factor must be integer -1 or +1')
    if not isinstance(manifest, Mapping):
        raise ValueError('Manifest must be an object')
    if not isinstance(source_path, Sequence) or isinstance(source_path, (str, bytes)) or not source_path or any(
            not isinstance(item, str) or not item for item in source_path):
        raise ValueError('A complete source occurrence path is required')
    result = copy.deepcopy(dict(manifest))
    parts = result.get('parts')
    if not isinstance(parts, list) or not all(isinstance(part, dict) for part in parts):
        raise ValueError('Manifest parts must be objects')
    matched = [part for part in parts if part.get('path') == list(source_path)]
    if len(matched) != 1 or matched[0].get('name') != DRUM_NAME:
        raise ValueError('Missing, ambiguous, or non-drum source occurrence')
    part = matched[0]
    # Validate the source coordinates before admitting a sign override.
    surface_velocity(part, part.get('axis_point'), time_s=0.)
    part['omega_per_surface_speed'] *= factor
    return result


def surface_velocity(part: Mapping, point, *, time_s: float) -> np.ndarray:
    """World-frame velocity at a contact, with the existing 0.25-second ramp."""
    try:
        axis = np.asarray(part['axis'], dtype=float)
        origin = np.asarray(part['axis_point'], dtype=float)
        point = np.asarray(point, dtype=float)
        omega = part['omega_per_surface_speed']
        if any(isinstance(value, bool) or not isinstance(value, Real) for value in (omega,time_s)):
            raise ValueError('Motion parameters must be real numbers')
        omega, time_s = float(omega), float(time_s)
    except (KeyError, TypeError, OverflowError) as exc:
        raise ValueError('Malformed surface coordinates') from exc
    if (any(value.shape != (3,) or not np.isfinite(value).all() for value in (axis, origin, point))
            or not math.isfinite(omega) or not math.isfinite(time_s) or time_s < 0):
        raise ValueError('Nonfinite or unsupported surface coordinates')
    if omega != 0 and abs(float(np.linalg.norm(axis))-1.) > 1e-12:
        raise ValueError('Moving surface requires a unit source axis')
    return np.cross(axis*omega*min(time_s/.25, 1.), point-origin)
