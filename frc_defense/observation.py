"""Fixed physical observation scaling shared by training and policy adapters."""
from __future__ import annotations

import math


def normalize_numpy_observation_in_place(obs, field_length, field_width,
                                        max_speeds, max_omegas):
    """Normalize one 35-value observation without fitting rollout statistics."""
    obs[0] /= field_length; obs[1] /= field_width; obs[2] /= math.pi
    obs[3:5] /= max(float(max_speeds[0]), .1); obs[5] /= max(float(max_omegas[0]), .1)
    obs[6] /= field_length; obs[7] /= field_width; obs[8] /= math.pi
    obs[9:11] /= max(float(max_speeds[1]), .1); obs[11] /= max(float(max_omegas[1]), .1)
    obs[12] /= field_length; obs[13] /= field_width
    obs[14] /= max(field_length, field_width)
    obs[15] /= 16.54; obs[16] /= 8.07
    obs[17:21] /= 1.0  # robot dimensions, in meters
    for col in (23, 26, 29, 32):
        obs[col] /= field_length
        obs[col + 1] /= field_width
    return obs


def normalize_tensor_observation_batch_in_place(obs, field_length, field_width,
                                                max_speeds, max_omegas):
    """Apply the same fixed scales to an ``[N, 35]`` Torch observation batch."""
    obs[:, 0].div_(field_length); obs[:, 1].div_(field_width); obs[:, 2].div_(math.pi)
    obs[:, 3:5].div_(max_speeds[:, 0:1].clamp_min(.1))
    obs[:, 5].div_(max_omegas[:, 0].clamp_min(.1))
    obs[:, 6].div_(field_length); obs[:, 7].div_(field_width); obs[:, 8].div_(math.pi)
    obs[:, 9:11].div_(max_speeds[:, 1:2].clamp_min(.1))
    obs[:, 11].div_(max_omegas[:, 1].clamp_min(.1))
    obs[:, 12].div_(field_length); obs[:, 13].div_(field_width)
    obs[:, 14].div_(max(field_length, field_width))
    obs[:, 15].div_(16.54); obs[:, 16].div_(8.07)
    obs[:, 17:21].div_(1.0)
    for col in (23, 26, 29, 32):
        obs[:, col].div_(field_length)
        obs[:, col + 1].div_(field_width)
    return obs
