"""Restricted experimental bridge between Plan A PPO and shadow Adam.

PPO still performs host replay checks, row selection and metric reads. The
optimizer must be capturable Adam; module buffers are rejected by ShadowAdam.
Only a single stream with no concurrent observers is
supported. All process RNG use and the current GPU device must be exclusive
to the update. RNG use on other GPU devices is unsupported. Host RNG state
reads and the validity check synchronize with the host. Parameter publication
and RNG installation are not crash-atomic; publication errors are unrecoverable.
"""

import hashlib
import json
import random

import torch

from .experimental_shadow_adam import ShadowAdam
from .curriculum_training import source_identity
from .plan_a_euler_prototype import graph_path_integral_segmented_checked
from .meco_track_training import _ppo_update


def _require_current_source(expected_sha256):
    identity = source_identity()
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True,
                                       allow_nan=False).encode()).hexdigest()
    if digest != expected_sha256:
        raise RuntimeError("PPO transaction source identity changed")


def transactional_ppo_update(track, model, value_head, optimizer, rows, config,
                             bootstrap_value, *, update_number,
                             diagnostic_output, source_identity_sha256):
    """Speculate a complete learned-P update, then conditionally publish it.

    Invalid path metadata leaves live parameter, gradient and initialized Adam
    state tensors unchanged. Other replay checks retain their ordinary host
    error paths. This function is experimental and makes no GPU-only claim.
    """
    _require_current_source(source_identity_sha256)
    if track not in ("A1", "A2", "A3") or config.p_mode != "learned":
        raise ValueError("requires a learned-P Plan A update")
    if not rows or any(row.get("support") is None for row in rows):
        raise ValueError("requires frozen Plan A support for every row")
    transaction = ShadowAdam(optimizer)
    if transaction.device.type != "cuda":
        raise ValueError("requires a HIP device")
    speculative_model, speculative_value = transaction.clone_modules(model, value_head)
    flags = []

    def checked_integral(node_cost, support):
        return graph_path_integral_segmented_checked(node_cost, support, flags)

    # A private generator reserves the baseline permutations without advancing
    # the default GPU generator until the transaction is accepted.
    device = transaction.device
    current_device_before = torch.cuda.current_device()
    gpu_before = torch.cuda.get_rng_state(device)
    cpu_before = torch.get_rng_state()
    python_before = random.getstate()
    permutation_generator = torch.Generator(device=device)
    permutation_generator.set_state(gpu_before)
    published = False
    try:
        metrics = _ppo_update(track, speculative_model, speculative_value,
                              transaction.shadow, rows, config, bootstrap_value,
                              update_number=update_number,
                              diagnostic_output=diagnostic_output,
                              source_identity_sha256=source_identity_sha256,
                              path_integral=checked_integral,
                              permutation_generator=permutation_generator)
        if not flags:
            raise RuntimeError("no Plan A path integrals were checked")
        _require_current_source(source_identity_sha256)
        # Other device RNG consumption would change baseline ordering.
        if torch.cuda.current_device() != current_device_before:
            raise RuntimeError("current GPU device changed during PPO")
        if not torch.equal(torch.cuda.get_rng_state(device), gpu_before):
            raise RuntimeError("unexpected default GPU RNG consumption during PPO")
        transaction.require_all(flags)
        accepted = transaction.commit_after_validation()
        if not accepted:
            raise RuntimeError("invalid Plan A path metadata; PPO update discarded")
        # Publication and RNG installation are not crash-atomic.
        published = True
        torch.cuda.set_rng_state(permutation_generator.get_state(), device)
        return metrics
    finally:
        if not published:
            if torch.cuda.current_device() != current_device_before:
                torch.cuda.set_device(current_device_before)
            torch.cuda.set_rng_state(gpu_before, device)
            torch.set_rng_state(cpu_before)
            random.setstate(python_before)
