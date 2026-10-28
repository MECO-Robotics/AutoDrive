"""Strict matched-prefix counterfactual labels for MECO Phase 2.

This collector branches a frozen diagnostic teacher trajectory at archived
Plan A/Plan C descriptors. It emits simulator returns, never expert labels.
Replay parity failures abort collection; no nearest or approximate match is
accepted. It is intentionally a separate v2 schema from v1 behavior rows.
"""
from __future__ import annotations

import hashlib
import json
import math
import enum
import os
from contextlib import contextmanager
from dataclasses import replace
import inspect
import platform
from pathlib import Path
import random
import sys
import types
from functools import lru_cache

import numpy as np
import torch
from torch.nn import functional as F

from .curriculum_scenarios import make_curriculum_env, teacher_actions
from .meco_candidates import (CANDIDATE_BOUNDARY,
    ROUTE_COST_POINTS_PER_SECOND, build_route_feasible_candidates)
from .meco_plan_a_policy import (SUPPORT_ORDER, build_plan_a_support,
    plan_a_support_digest_cpu_reference)
from .meco_plan_a_support_tree import support_digest_v1_compatible as plan_a_support_digest
from .meco_pose_graph import build_legal_pose_graph
from .meco_tracks import (PREDICTION_MODES, candidate_digest,
    prediction_contract)
from .training_curriculum import get_scenario

DATASET_SCHEMA = "meco-counterfactual-trajectory-v3"
TARGET_SCHEMA = "meco-matched-prefix-gross-return-v3"
PLAN_A_P_ROUTE_DATASET_SCHEMA = "meco-counterfactual-plan-a-p-route-v2"
PLAN_A_P_ROUTE_TARGET_SCHEMA = "meco-executed-route-gross-return-ranking-v1"
PLAN_A_P_ROUTE_ORACLE_DATASET_SCHEMA = "meco-counterfactual-plan-a-p-route-oracle-v3"
PLAN_A_P_ROUTE_ORACLE_TARGET_SCHEMA = "meco-executed-route-independent-cost-v1"
PLAN_A_P_ROUTE_ORACLE_VERSION = "base-bump-plus-effective-p-grid-edge-integral-v1"
PLAN_A_LEGAL_NODE_DATASET_SCHEMA = "meco-counterfactual-plan-a-legal-node-v1"
PLAN_A_LEGAL_NODE_TARGET_SCHEMA = "meco-plan-a-legal-node-utility-and-feasible-preference-v1"
PLAN_A_U_ORACLE_VERSION = "matched-prefix-equal-feasible-behavior-gross-return-v1"
PLAN_A_U_ORACLE_SEMANTICS = (
    "gross undiscounted reward through the fixed horizon after one uniformly sampled "
    "feasible descriptor per feasible behavior, equal-weighted across feasible behaviors; "
    "each repeat index uses common random numbers across goal/behavior branches; fixed "
    "deterministic teacher continuation; sampled node only; no optimality claim")
PLAN_A_EXPERT_GOAL_SOURCE_SCHEMA = "meco-plan-a-expert-goal-source-v1"
PLAN_A_EXPERT_GOAL_SOURCE_TARGET_SCHEMA = "meco-plan-a-explicit-selected-goal-trace-v1"
PLAN_A_LEGAL_NODE_MAX_BRANCHES_PER_PREFIX = 512
PLAN_A_LEGAL_NODE_MAX_DATASET_BRANCHES = 12000
PLAN_A_LEGAL_NODE_REPEATS = 2
PLAN_C_MATCHED_REPEAT_COUNT = 2
PLAN_C_REPEAT_TARGET_SCHEMA = "meco-plan-c-matched-repeated-gross-return-v1"
PLAN_C_REPEAT_SEED_DERIVATION = "sha256-canonical-json-u32be-v1"
PLAN_C_REPEAT_GENERATOR_SEED_DERIVATION = "sha256-repeat-seed-plus-stable-attribute-path-u64-v1"
PLAN_C_PREFERENCE_POLICY = "same-strict-sign-for-every-paired-repeat-else-abstain-v1"
# v2 hashes all canonical environment state except the root environment's
# CUDA-graph execution cache (`env._physics_graphs`). Captured graph objects are
# device/runtime execution machinery, not simulator state; every other field
# remains subject to the same fail-closed traversal as v1.
STATE_DIGEST_SCHEMA = "meco-full-environment-state-v2"
DEFAULT_GAMMA = 0.993
COUNTERFACTUAL_DISCOUNT_SEMANTICS = (
    "one discount transition per macro decision, independent of macro duration; "
    "first branch reward exponent 0 and each continuation decision increments "
    "exponent once, including a final partial macro")


@contextmanager
def _counterfactual_rng_scope(seed):
    """Seed simulator RNGs for reproducible collection/replay, then restore callers."""
    if type(seed) is not int or seed < 0:
        raise ValueError("counterfactual seed must be a nonnegative integer")
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        random.seed(seed)
        np.random.seed(seed % (2 ** 32))
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)
CONTINUATION_POLICY = "curriculum_scenarios.teacher_actions; deterministic sensed-state policy"
MIN_ELIGIBLE_BRANCHES_PER_SEED = 2
_VALIDATED_EPISODE_ELIGIBILITY_CACHE = {}


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                            allow_nan=False).encode()).hexdigest()


def plan_c_repeat_seed(*, protocol_id, scenario_id, category, seed,
                       decision_index, repeat_index):
    """Derive a candidate-matched branch RNG stream from a frozen identity."""
    if (not isinstance(protocol_id, str) or not protocol_id or
            not isinstance(scenario_id, str) or not scenario_id or
            not isinstance(category, str) or not category or
            any(type(value) is not int or value < 0 for value in
                (seed, decision_index, repeat_index)) or
            repeat_index >= PLAN_C_MATCHED_REPEAT_COUNT):
        raise ValueError("invalid Plan C matched-repeat seed identity")
    material = {"schema": PLAN_C_REPEAT_TARGET_SCHEMA,
        "protocol_id": protocol_id, "scenario_id": scenario_id,
        "category": category, "seed": seed,
        "decision_index": decision_index, "repeat_index": repeat_index}
    digest = hashlib.sha256(json.dumps(material, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode()).digest()
    return int.from_bytes(digest[:4], "big")


def plan_c_repeat_protocol(protocol_id):
    """Bind repeat count and seed/pairing rules to the qualification protocol."""
    if not isinstance(protocol_id, str) or not protocol_id:
        raise ValueError("Plan C repeat protocol requires a qualification protocol ID")
    body = {"schema": PLAN_C_REPEAT_TARGET_SCHEMA,
        "qualification_protocol_id": protocol_id,
        "repeat_count": PLAN_C_MATCHED_REPEAT_COUNT,
        "seed_derivation": PLAN_C_REPEAT_SEED_DERIVATION,
        "environment_generator_seed_derivation": PLAN_C_REPEAT_GENERATOR_SEED_DERIVATION,
        "common_random_numbers": "same repeat seed for all candidates at a prefix",
        "prefix_replay": "replay archived prefix from base train seed; reseed global and environment RNGs at branch",
        "continuation_policy": CONTINUATION_POLICY,
        "preference_policy": PLAN_C_PREFERENCE_POLICY,
        "variance_note": "two repeats support stability abstention, not precise variance calibration"}
    return {**body, "repeat_protocol_id": _hash(body)}


def _seed_plan_c_environment_generators(root, repeat_seed):
    """Reseed every simulator-local Torch generator using a stable field path."""
    if type(repeat_seed) is not int or repeat_seed < 0:
        raise ValueError("Plan C repeat generator seed must be a nonnegative integer")
    stack, seen, seeded = [("env", root)], set(), []
    while stack:
        path, value = stack.pop()
        if isinstance(value, torch.Generator):
            identity = id(value)
            if identity in seen:
                continue
            seen.add(identity)
            material = f"plan-c-environment-generator-v1:{repeat_seed}:{path}"
            seed = int.from_bytes(hashlib.sha256(material.encode()).digest()[:8], "big") & ((1 << 63) - 1)
            value.manual_seed(seed)
            seeded.append({"path": path, "seed": seed, "device": str(value.device)})
            continue
        if isinstance(value, (torch.Tensor, torch.nn.Module, types.ModuleType)) or callable(value):
            continue
        identity = id(value)
        if identity in seen:
            continue
        seen.add(identity)
        if isinstance(value, dict):
            entries = []
            for key, item in value.items():
                if type(key) not in (str, int, bool):
                    continue
                entries.append((f"{path}[{type(key).__name__}:{key}]", item))
            stack.extend(reversed(sorted(entries, key=lambda pair: pair[0])))
        elif isinstance(value, (list, tuple)):
            stack.extend((f"{path}[{index}]", item)
                         for index, item in reversed(list(enumerate(value))))
        elif hasattr(value, "__dict__"):
            stack.extend((f"{path}.{name}", vars(value)[name])
                         for name in reversed(sorted(vars(value)))
                         if not name.startswith("__"))
    if not seeded:
        raise RuntimeError("Plan C repeat found no simulator-local Torch generators to reseed")
    seeded.sort(key=lambda item: item["path"])
    return {"generator_count": len(seeded), "seed_manifest_sha256": _hash(seeded)}


def plan_c_repeat_statistics(left_returns, right_returns):
    """Calculate paired repeat statistics and a conservative stable preference."""
    if (not isinstance(left_returns, (list, tuple)) or
            not isinstance(right_returns, (list, tuple)) or
            len(left_returns) != PLAN_C_MATCHED_REPEAT_COUNT or
            len(right_returns) != PLAN_C_MATCHED_REPEAT_COUNT or
            any(type(value) not in (int, float) or not math.isfinite(value)
                for value in (*left_returns, *right_returns))):
        raise ValueError("Plan C preference requires every finite matched repeat")
    differences = [float(left) - float(right)
                   for left, right in zip(left_returns, right_returns)]
    mean = sum(differences) / len(differences)
    variance = sum((value - mean) ** 2 for value in differences) / (len(differences) - 1)
    standard_error = math.sqrt(variance / len(differences))
    stable = all(value > 0 for value in differences) or all(value < 0 for value in differences)
    return {"repeat_differences": differences,
        "mean_difference": mean,
        "sample_variance": variance,
        "standard_error": standard_error,
        "stable_preference": stable,
        "preferred_side": ("left" if differences[0] > 0 else "right") if stable else None}


def _plan_c_repeated_outcome(descriptor, repeats):
    """Summarize complete repeated candidate returns without dropping evidence."""
    if (not isinstance(repeats, list) or len(repeats) != PLAN_C_MATCHED_REPEAT_COUNT or
            any(not isinstance(item, dict) or
                not all(type(item.get(key)) in (int, float) and math.isfinite(item[key])
                        for key in ("gross_macro_return", "gross_horizon_return",
                                    "discounted_horizon_return"))
                for item in repeats)):
        raise ValueError("Plan C candidate is missing a finite matched repeat")
    means = {}
    for key in ("gross_macro_return", "gross_horizon_return", "discounted_horizon_return"):
        values = [float(item[key]) for item in repeats]
        mean = sum(values) / len(values)
        means[key] = mean
    returns = [float(item["gross_horizon_return"]) for item in repeats]
    mean = means["gross_horizon_return"]
    variance = sum((value - mean) ** 2 for value in returns) / (len(returns) - 1)
    return {"descriptor": descriptor, "repeat_count": PLAN_C_MATCHED_REPEAT_COUNT,
        "repeat_returns": repeats, **means,
        "repeat_sample_variance_gross_horizon_return": variance,
        "continuation_policy": CONTINUATION_POLICY,
        "discount_semantics": COUNTERFACTUAL_DISCOUNT_SEMANTICS}


def _plan_c_pairwise_labels(outcomes):
    """Build complete pair rows; unstable pairs are explicit abstentions."""
    labels = []
    for index, left in enumerate(outcomes):
        for right in outcomes[index + 1:]:
            left_id = left["descriptor"]["candidate_id"]
            right_id = right["descriptor"]["candidate_id"]
            left_returns = [item["gross_horizon_return"] for item in left["repeat_returns"]]
            right_returns = [item["gross_horizon_return"] for item in right["repeat_returns"]]
            stats = plan_c_repeat_statistics(left_returns, right_returns)
            preferred = (left_id if stats["preferred_side"] == "left" else
                         right_id if stats["preferred_side"] == "right" else None)
            labels.append({"left_id": left_id, "right_id": right_id,
                "gross_horizon_return_difference": left["gross_horizon_return"] -
                    right["gross_horizon_return"],
                "repeat_mean_difference": stats["mean_difference"],
                "repeat_differences": stats["repeat_differences"],
                "paired_sample_variance": stats["sample_variance"],
                "paired_standard_error": stats["standard_error"],
                "stable_preference": stats["stable_preference"],
                "preferred_id": preferred})
    return labels


def _plan_a_u_evidence_sha256(support_digest, target, pair_outcomes):
    """Bind a sampled-node U estimand to every matched rollout used to form it."""
    node = target.get("node_id")
    evidence = []
    for pair in pair_outcomes:
        if pair.get("node_id") != node:
            continue
        repeats = pair.get("repeat_returns", pair.get("repeat_outcomes", []))
        evidence.append({"pair_index": pair.get("pair_index"),
            "behavior_id": pair.get("behavior_id"),
            "descriptor_id": pair.get("descriptor_id"),
            "status": pair.get("status"),
            "repeat_statuses": [item.get("status") for item in repeats],
            "repeat_seeds": [item.get("repeat_seed") for item in repeats],
            "gross_horizon_returns": [item.get("gross_horizon_return")
                if item.get("status") == "available" else None for item in repeats]})
    return _hash({"oracle_version": PLAN_A_U_ORACLE_VERSION,
        "support_digest": support_digest, "node_id": node,
        "feasible_behavior_ids": target.get("feasible_behavior_ids"),
        "u_target": target.get("u_target"), "pair_evidence": evidence})


def _native_fuel_extension_identity():
    """Return the active/default native HIP identity without triggering a build."""
    from . import fuel_physics_hip
    return fuel_physics_hip.native_extension_identity()


def _verify_native_fuel_extension(identity):
    """Fail closed if the loaded extension or its on-disk path was replaced."""
    current = _native_fuel_extension_identity()
    if current != identity:
        raise RuntimeError("loaded native HIP extension identity changed")
    if identity is None or identity.get("status") == "missing":
        return
    path = Path(identity["path"])
    if identity.get("status") == "unprepared":
        try:
            stat = path.stat()
        except FileNotFoundError:
            if any(key in identity for key in ("candidate_device", "candidate_inode",
                    "candidate_size", "candidate_mtime_ns")):
                raise RuntimeError("unprepared native HIP extension appeared or disappeared")
            return
        except OSError as exc:
            raise RuntimeError("unprepared native HIP extension path is unavailable") from exc
        candidate = (identity.get("candidate_device"), identity.get("candidate_inode"),
            identity.get("candidate_size"), identity.get("candidate_mtime_ns"))
        actual = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
        if candidate != actual:
            raise RuntimeError("unprepared native HIP extension path changed")
        return
    try:
        stat = path.stat()
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise RuntimeError("loaded native HIP extension path is unavailable") from exc
    disk_identity = {**identity, "device": stat.st_dev, "inode": stat.st_ino,
        "size": stat.st_size, "mtime_ns": stat.st_mtime_ns, "sha256": digest}
    if disk_identity != identity:
        raise RuntimeError("native HIP extension path was replaced during collection or replay")


def _artifact_identity_sha256(sources_sha256, native_extension_identity):
    return _hash({"source_identity_sha256": sources_sha256,
        "native_extension_identity": native_extension_identity})


def _require_native_extension_manifest_identity(manifest):
    device = manifest.get("device", {}).get("device") if isinstance(
        manifest.get("device"), dict) else None
    identity = manifest.get("native_extension_identity")
    if manifest.get("native_extension_identity_schema") != 1:
        raise ValueError("legacy counterfactual artifact lacks native identity schema v1; recollect it")
    if device and device.startswith("cuda") and (
            not isinstance(identity, dict) or identity.get("status") != "present"):
        raise ValueError("legacy counterfactual v2 artifact lacks required native HIP identity; recollect it")
    if manifest.get("artifact_identity_sha256") != _artifact_identity_sha256(
            manifest.get("source_identity_sha256"), identity):
        raise ValueError("counterfactual native HIP artifact identity digest mismatch")
    sources = manifest.get("sources")
    if not isinstance(sources, dict) or sources.get("_native_fuel_extension") != identity:
        raise ValueError("counterfactual source identity does not bind its native HIP binary")
    _verify_native_fuel_extension(identity)
    return identity


def _tensor_digest(tensor):
    if tensor.layout != torch.strided:
        raise TypeError(f"unsupported tensor layout for state digest: {tensor.layout}")
    # `contiguous()` may preserve arbitrary strides on size-one dimensions.
    # Clone with an explicit format so the byte reinterpretation below always
    # sees canonical dense storage, including degenerate singleton shapes.
    value = tensor.detach().to(device="cpu").clone(memory_format=torch.contiguous_format)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(json.dumps(list(value.shape)).encode())
    digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _stable_key(value):
    """Encode immutable mapping/set keys without touching traversal references."""
    if value is None:
        item = ["none"]
    elif type(value) is bool:
        item = ["bool", value]
    elif type(value) is int:
        item = ["int", str(value)]
    elif type(value) is str:
        item = ["str", value]
    elif type(value) is float and math.isfinite(value):
        item = ["float", value.hex()]
    elif isinstance(value, enum.Enum):
        item = ["enum", f"{type(value).__module__}.{type(value).__qualname__}",
                _stable_key(value.value)]
    elif isinstance(value, tuple):
        item = ["tuple", [_stable_key(part) for part in value]]
    elif isinstance(value, (torch.device, torch.dtype, Path)):
        item = [type(value).__name__, str(value)]
    else:
        raise TypeError(f"unsupported unordered key type: {type(value)!r}")
    return json.dumps(item, sort_keys=True, separators=(",", ":"), allow_nan=False)


@lru_cache(maxsize=8)
def _module_file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _state_value(value, seen, path, *, excluded_attributes=frozenset()):
    """Canonicalize mutable simulator/environment state; unknown state fails closed."""
    # Subclasses use generic object inspection below so extra Python state is
    # included and opaque native state fails closed.
    if type(value) is torch.Tensor:
        object_id = id(value)
        if object_id in seen:
            return {"ref": seen[object_id]}
        stable_id = len(seen)
        seen[object_id] = stable_id
        return {"object_id": stable_id, "tensor": _tensor_digest(value),
                "device": str(value.device), "requires_grad": value.requires_grad}
    if type(value) is torch.Generator:
        object_id = id(value)
        if object_id in seen:
            return {"ref": seen[object_id]}
        stable_id = len(seen)
        seen[object_id] = stable_id
        return {"object_id": stable_id, "torch_generator": str(value.device),
                "state": _state_value(value.get_state(), seen, path + ".state")}
    if type(value) is np.ndarray:
        object_id = id(value)
        if object_id in seen:
            return {"ref": seen[object_id]}
        stable_id = len(seen)
        seen[object_id] = stable_id
        array = np.ascontiguousarray(value)
        return {"object_id": stable_id,
                "numpy_array": hashlib.sha256(array.tobytes()).hexdigest(),
                "dtype": str(array.dtype), "shape": list(array.shape)}
    if isinstance(value, np.generic):
        return value.item()
    if value is None or type(value) in (str, int, bool):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else {"nonfinite_float": repr(value)}
    if isinstance(value, (torch.device, torch.dtype, torch.Size, Path)):
        return {type(value).__name__: str(value)}
    if isinstance(value, enum.Enum):
        return {"enum": f"{type(value).__module__}.{type(value).__qualname__}",
                "value": _state_value(value.value, seen, path + ".value")}
    if type(value) is random.Random:
        object_id = id(value)
        if object_id in seen:
            return {"ref": seen[object_id]}
        stable_id = len(seen)
        seen[object_id] = stable_id
        return {"object_id": stable_id,
                "python_random_state": _state_value(value.getstate(), seen,
                                                     path + ".random_state")}
    if type(value) is np.random.RandomState:
        object_id = id(value)
        if object_id in seen:
            return {"ref": seen[object_id]}
        stable_id = len(seen)
        seen[object_id] = stable_id
        return {"object_id": stable_id,
                "numpy_random_state": _state_value(value.get_state(), seen,
                                                    path + ".random_state")}
    if type(value) is np.random.Generator:
        object_id = id(value)
        if object_id in seen:
            return {"ref": seen[object_id]}
        stable_id = len(seen)
        seen[object_id] = stable_id
        return {"object_id": stable_id,
                "numpy_generator": _state_value(value.bit_generator.state, seen,
                                                 path + ".bit_generator_state")}

    object_id = id(value)
    if object_id in seen:
        return {"ref": seen[object_id]}
    stable_id = len(seen)
    seen[object_id] = stable_id

    if type(value) is dict:
        keyed = [(_stable_key(key), key) for key in value]
        keyed.sort(key=lambda pair: pair[0])
        if any(keyed[index - 1][0] == keyed[index][0]
               for index in range(1, len(keyed))):
            raise TypeError(f"mapping key encoding collision at {path}")
        entries = [(token, _state_value(value[key], seen, f"{path}[{token}]") )
                   for token, key in keyed]
        return {"dict_id": stable_id, "entries": entries}
    if type(value) in (list, tuple):
        return {"sequence_id": stable_id, "kind": type(value).__name__,
                "items": [_state_value(item, seen, f"{path}[{index}]")
                          for index, item in enumerate(value)]}
    if type(value) in (set, frozenset):
        keyed = [(_stable_key(item), item) for item in value]
        keyed.sort(key=lambda pair: pair[0])
        if any(keyed[index - 1][0] == keyed[index][0]
               for index in range(1, len(keyed))):
            raise TypeError(f"set item encoding collision at {path}")
        items = [token for token, _ in keyed]
        return {"set_id": stable_id, "items": items}
    if isinstance(value, types.MethodType):
        return {"object_id": stable_id,
                "bound_method": f"{value.__func__.__module__}.{value.__func__.__qualname__}",
                "function_state": _state_value(value.__func__, seen, path + ".function"),
                "self": _state_value(value.__self__, seen, path + ".self")}
    if isinstance(value, types.BuiltinMethodType):
        return {"object_id": stable_id,
                "bound_builtin_method": f"{value.__module__}.{value.__qualname__}",
                "self": _state_value(value.__self__, seen, path + ".self")}
    if isinstance(value, types.FunctionType):
        closure_items = []
        for cell in value.__closure__ or ():
            try:
                closure_items.append(cell.cell_contents)
            except ValueError:
                closure_items.append({"empty_closure_cell": True})
        return {"object_id": stable_id,
                "function": f"{value.__module__}.{value.__qualname__}",
                "defaults": _state_value(value.__defaults__, seen, path + ".defaults"),
                "kwdefaults": _state_value(value.__kwdefaults__, seen, path + ".kwdefaults"),
                "closure": _state_value(tuple(closure_items), seen, path + ".closure")}
    if isinstance(value, types.BuiltinFunctionType):
        owner = getattr(value, "__self__", None)
        return {"object_id": stable_id,
                "callable_or_type": f"{value.__module__}.{value.__qualname__}",
                "owner": (None if owner is None else
                          _state_value(owner, seen, path + ".self"))}
    if isinstance(value, type):
        return {"object_id": stable_id,
                "callable_or_type": f"{getattr(value, '__module__', type(value).__module__)}."
                f"{getattr(value, '__qualname__', type(value).__qualname__)}"}

    if isinstance(value, types.ModuleType):
        if value.__name__ != "autodrive_fuel_physics_hip":
            raise TypeError(f"unreviewed module state at {path}: {value.__name__}")
        module_path = getattr(value, "__file__", None)
        if not module_path or not Path(module_path).is_file():
            raise TypeError(f"native extension has no immutable file identity at {path}")
        return {"object_id": stable_id,
                "stateless_native_extension": value.__name__,
                "file_sha256": _module_file_sha256(str(Path(module_path).resolve()))}

    cls = type(value)
    if cls is types.SimpleNamespace:
        class_source = "builtin:types.SimpleNamespace"
    else:
        class_source = None
        for base in cls.__mro__:
            if base in (object, types.SimpleNamespace):
                continue
            try:
                source_path = inspect.getsourcefile(base)
            except (TypeError, OSError):
                source_path = None
            if source_path is None:
                raise TypeError(f"opaque/native base state at {path}: {base!r}")
            class_source = source_path

    attributes = getattr(value, "__dict__", None)
    slot_values = {}
    for base in cls.__mro__:
        declared = base.__dict__.get("__slots__", ())
        if isinstance(declared, str):
            declared = (declared,)
        for name in declared:
            if name in ("__dict__", "__weakref__"):
                continue
            if name in excluded_attributes:
                continue
            actual_name = name
            if name.startswith("__") and not name.endswith("__"):
                actual_name = f"_{base.__name__.lstrip('_')}{name}"
            if hasattr(value, actual_name):
                slot_values[f"{base.__module__}.{base.__qualname__}.{name}"] = getattr(value, actual_name)
    if attributes is None and not slot_values:
        raise TypeError(f"cannot canonicalize opaque state at {path}: {type(value)!r}")
    attr_values = {} if attributes is None else attributes
    if excluded_attributes and type(value) is not dict:
        # Callers can omit narrowly scoped execution caches while preserving
        # strict traversal for every other field and nested object.
        attr_values = {name: item for name, item in attr_values.items()
                       if name not in excluded_attributes}
    return {"object_id": stable_id,
            "type": f"{cls.__module__}.{cls.__qualname__}",
            "class_source": class_source,
            "attributes": {name: _state_value(item, seen, f"{path}.{name}")
                           for name, item in sorted(attr_values.items())},
            "slots": {name: _state_value(item, seen, f"{path}.{name}")
                      for name, item in sorted(slot_values.items())}}


def _simulator_state_digest(env):
    """Hash simulator, reward, planner, fuel-physics, environment, and RNG state."""
    state = {"environment": _state_value(env, {}, "env",
            excluded_attributes=frozenset({"_physics_graphs"})),
        "python_global_rng": _state_value(random.getstate(), {}, "python_rng"),
        "numpy_global_rng": _state_value(np.random.get_state(), {}, "numpy_rng"),
        "torch_global_rng": _state_value(torch.random.get_rng_state(), {}, "torch_rng"),
        "torch_cuda_global_rng": [_state_value(value, {}, f"cuda_rng[{index}]")
            for index, value in enumerate(torch.cuda.get_rng_state_all())]
            if torch.cuda.is_available() else []}
    return _hash({"schema": STATE_DIGEST_SCHEMA, "state": state})


def _observation_digest(observation):
    if observation is None:
        raise RuntimeError("simulator has no current observation")
    return _tensor_digest(observation)


def _device_identity(device):
    from .meco_device import require_indexed_cuda_device
    device = str(require_indexed_cuda_device(device,
        purpose="MECO counterfactual collection or fitting"))
    resolved = torch.device(device)
    identity = {"torch": str(torch.__version__), "python": sys.version,
                "device": str(resolved), "platform": platform.platform()}
    if resolved.type == "cuda":
        index = resolved.index if resolved.index is not None else torch.cuda.current_device()
        identity.update(device_index=index,
                        device_name=torch.cuda.get_device_name(index),
                        capability=list(torch.cuda.get_device_capability(index)))
    return identity


def _reward_step(env, action, action_ticks, goal_xy=None, goal_heading=None,
                 allow_action7_traversal=False,
                 planner_traversal_cost_override=None,
                 planner_traversal_cost_scale=1.0,
                 planner_force_goal_replan=False,
                 planner_route_trace=None, return_execution_flags=False):
    from .meco_track_training import _run_macro_step
    values = _run_macro_step(env, action, action_ticks, goal_xy,
        goal_heading=goal_heading,
        allow_action7_traversal=allow_action7_traversal,
        planner_traversal_cost_override=planner_traversal_cost_override,
        planner_traversal_cost_scale=planner_traversal_cost_scale,
        planner_force_goal_replan=planner_force_goal_replan,
        planner_route_trace=planner_route_trace,
        return_execution_flags=return_execution_flags)
    reward, ended, timeout = values[:3]
    result = (float(reward.detach().cpu().item()), bool(ended), bool(timeout),
            list(values[3]), values[4], values[5], bool(values[7]))
    if return_execution_flags:
        requested_action = int(action)
        remasked = any(item != requested_action for item in values[3])
        return (*result, bool(values[8]), remasked, bool(values[9]))
    return result


def _plan_a_p_map_digest(cost_map):
    if cost_map is None:
        return None
    value = torch.as_tensor(cost_map).detach().contiguous().cpu()
    metadata = json.dumps({"shape": list(value.shape), "dtype": str(value.dtype)},
                          sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(metadata + b"\0" + value.numpy().tobytes()).hexdigest()


def _plan_a_p_intervention_map(env, intervention_seed):
    """Model-independent smooth positive route probe map, reproducible from seed."""
    if type(intervention_seed) is not int or not 0 <= intervention_seed < 2**63:
        raise ValueError("Plan A P intervention seed is invalid")
    nx, ny = int(env.planner.nx), int(env.planner.ny)
    coarse_x, coarse_y = max(2, (nx + 5) // 6), max(2, (ny + 5) // 6)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(intervention_seed)
    coarse = torch.rand((1, 1, coarse_x, coarse_y), generator=generator,
                        dtype=torch.float32)
    smooth = F.interpolate(coarse, size=(nx, ny), mode="bilinear",
                           align_corners=False)[0, 0]
    # Keep the probe scale fixed in normalized planner-edge units.
    result = smooth.to(device=env.device, dtype=env.sim.pose.dtype) * 0.5
    if (result.shape != (nx, ny) or not bool(torch.isfinite(result).all()) or
            bool((result < 0).any())):
        raise RuntimeError("Plan A P intervention map failed validation")
    return result


def _plan_a_p_route_record(env, *, intervention_map, intervention_seed,
                           selected_action, selected_goal,
                           selected_goal_descriptor_sha256,
                           scale, route_trace):
    planner = env.planner
    path_length = int(planner.last_lengths[3].detach().item())
    if path_length < 1:
        raise RuntimeError("Plan A P route intervention produced no native route")
    points = planner.last_path[3, :path_length].detach().cpu().tolist()
    if not all(len(point) == 2 and all(math.isfinite(float(v)) for v in point)
               for point in points):
        raise RuntimeError("Plan A P native route contains invalid coordinates")
    if not route_trace or route_trace[-1]["route_points_m"] != [
            [float(v) for v in point] for point in points]:
        raise RuntimeError("Plan A P route trace does not end at the native planner route")
    signature = getattr(env, "_last_applied_planner_traversal_signature", None)
    if intervention_map is None:
        if signature is not None:
            raise RuntimeError("P0 branch unexpectedly retained a weighted route signature")
        applied = False
    else:
        if (not isinstance(signature, dict) or signature.get("map") is None or
                signature.get("action") != selected_action or
                signature.get("goal") != tuple(float(v) for v in selected_goal) or
                float(signature.get("scale", -1.)) != scale):
            raise RuntimeError("P route intervention was not applied to its selected goal")
        applied = True
    if any(item.get("p_map_applied") != applied for item in route_trace):
        raise RuntimeError("Plan A P map application changed within its frozen macro")
    return {"intervention_seed": intervention_seed,
        "map_sha256": _plan_a_p_map_digest(intervention_map),
        "map_applied": applied, "map_scale": scale if applied else None,
        "route_points_m": [[float(v) for v in point] for point in points],
        "route_sha256": _hash([[float(v) for v in point] for point in points]),
        "route_trace_sha256": _hash([item["route_sha256"] for item in route_trace]),
        "route_trace": route_trace,
        "selected_action": selected_action,
        "selected_goal_xy_m": [float(v) for v in selected_goal],
        "selected_goal_descriptor_sha256": selected_goal_descriptor_sha256,
        "selected_goal_refresh_forced": True}


def _prefix_snapshot(env, graph, episode_id, *, action_ticks, candidate_k,
                     prediction_mode, max_c_candidates, max_a_pairs,
                     materialize=False, require_hip_plan_a=True):
    from .meco_track_training import (_candidate_replay_digest, _route_cost_tensor)
    candidates = build_route_feasible_candidates(env, env._last_obs, graph,
        episode_id=episode_id, action_ticks=action_ticks,
        candidate_k=candidate_k, prediction_mode=prediction_mode)
    observation = candidates.observation
    route = _route_cost_tensor(candidates, ROUTE_COST_POINTS_PER_SECOND,
                               device=env.device)
    risk = candidates.risk_cost_points.unsqueeze(0)
    candidate_hash = _candidate_replay_digest(candidates.table, route, risk)
    support = build_plan_a_support(env, graph, env._last_obs,
        observation.snapshot_id,
        require_hip_shortest_path=require_hip_plan_a)

    valid_c = [index for index, valid in enumerate(candidates.table.valid_mask) if valid]
    valid_c = _deterministic_sample(valid_c, max_c_candidates,
        f"C:{episode_id}:{int(env.steps[0].item())}")
    c_descriptors = []
    for index in valid_c:
        record = candidates.route_records[index]
        c_descriptors.append({"candidate_index": index,
            "candidate_id": candidates.table.candidate_ids[index],
            "action": int(candidates.action_indices[index]),
            "goal_xy_m": list(record["goal_xy_m"]),
            "descriptor_sha256": _hash({"candidate_id": candidates.table.candidate_ids[index],
                "action": int(candidates.action_indices[index]),
                "goal_xy_m": record["goal_xy_m"],
                "route_record": record})})

    a_sample_key = f"A:{episode_id}:{int(env.steps[0].item())}"
    if require_hip_plan_a:
        # Keep the historical SHA-256 ordering exactly, but apply the full
        # support mask on device and copy back only the selected descriptor IDs.
        # The CPU creates an index permutation only; it never reads graph
        # reachability, costs, or valid support rows.
        pair_indices = sorted(range(support.valid_mask.numel()), key=lambda index:
            hashlib.sha256(f"{a_sample_key}:{index}".encode()).digest())
        ranked = torch.tensor(pair_indices, device=support.valid_mask.device,
                              dtype=torch.long)
        valid_a = ranked[support.valid_mask.index_select(0, ranked)][:max_a_pairs]
        valid_a = valid_a.detach().cpu().tolist()
    else:
        valid_a = torch.nonzero(support.valid_mask,
            as_tuple=False).reshape(-1).tolist()
        valid_a = _deterministic_sample(valid_a, max_a_pairs, a_sample_key)
    skip_reasons = []
    if len(valid_c) < 2:
        skip_reasons.append("fewer_than_two_plan_c_descriptors")
    if len(valid_a) < 2:
        skip_reasons.append("fewer_than_two_plan_a_descriptors")
    eligibility = {"eligible": not skip_reasons,
        "plan_c_descriptor_count": len(valid_c),
        "plan_a_descriptor_count": len(valid_a),
        "skip_reasons": skip_reasons}
    eligibility["metadata_sha256"] = _hash(eligibility)
    a_descriptors = []
    for pair in valid_a:
        node = int(support.pair_node_ids[pair].item())
        item = {"pair_index": pair, "node_id": node,
            "behavior_id": int(support.behavior_ids[pair].item()),
            "descriptor_id": int(support.descriptor_ids[pair].item()),
            "action": int(support.action_ids[pair].item()),
            "goal_xy_m": [float(value) for value in support.goal_xy[node].tolist()],
            "goal_heading_rad": float(support.goal_heading[node].item())}
        item["descriptor_sha256"] = _hash(item)
        a_descriptors.append(item)
    # The selected descriptor rows are now materialized at this counterfactual
    # artifact boundary. Only now may dynamic support bytes be digested; the
    # live HIP path uses device SHA-256 and copies its fixed-size digest here.
    support_hash = (plan_a_support_digest(support, graph) if require_hip_plan_a
                    else plan_a_support_digest_cpu_reference(support, graph))
    support = replace(support, support_digest=support_hash)
    if not support.support_digest or support.support_digest != support_hash:
        raise RuntimeError("Plan A support digest failed post-selection binding")
    result = {"snapshot_id": observation.snapshot_id,
        "macro_ticks": action_ticks,
        "simulator_state_sha256": _simulator_state_digest(env),
        "observation_sha256": _observation_digest(env._last_obs),
        "safe_observation_sha256": _hash({"snapshot_id": observation.snapshot_id,
            "raster": list(observation.raster), "globals": list(observation.globals)}),
        "candidate_digest": candidate_digest(candidates.table),
        "candidate_replay_digest": candidate_hash,
        "support_digest": support_hash,
        "teacher_action": int(teacher_actions(env, env._last_obs)[0, 3].item()),
        "candidate_descriptors": c_descriptors,
        "plan_a_descriptors": a_descriptors,
        "branch_eligibility": eligibility}
    if materialize:
        result["_training_inputs"] = {"observation": observation,
            "candidate_set": candidates, "route_cost": route,
            "risk_cost": risk, "support": support}
    return result


def replay_counterfactual_branch(record, manifest, *, require_hip_plan_a=True):
    """Replay under the record seed while restoring the caller's RNG streams."""
    device = manifest.get("device", {}).get("device") if isinstance(
        manifest.get("device"), dict) else None
    if device is None:
        raise ValueError("counterfactual replay manifest has no device identity")
    if device.startswith("cuda"):
        from .fuel_physics_hip import prepare_native_fuel_extension
        prepare_native_fuel_extension(device)
    identity = _require_native_extension_manifest_identity(manifest)
    if identity is not None:
        _verify_native_fuel_extension(identity)
    seed = record.get("seed") if isinstance(record, dict) else None
    with _counterfactual_rng_scope(seed):
        return _replay_counterfactual_branch_seeded(record, manifest,
            require_hip_plan_a=require_hip_plan_a)


def _replay_episode_eligibility(record, manifest, *, require_hip_plan_a=True):
    """Replay the complete teacher episode to verify post-branch eligibility metadata."""
    from .meco_track_training import _reset_episode
    scenario_id, category, seed = record["scenario_id"], record["category"], record["seed"]
    episode = record.get("episode_transcript")
    if (not isinstance(episode, list) or not episode or
            _hash(episode) != record.get("episode_transcript_sha256")):
        raise ValueError("final eligible row lacks the complete episode transcript")
    summary = record.get("seed_branch_summary")
    if not isinstance(summary, dict) or manifest.get("seed_branch_summaries", {}).get(
            str(record["seed"])) != summary:
        raise ValueError("episode eligibility summary differs from the manifest")
    replay_config_fields = (
        "horizon_physics_ticks", "max_macro_ticks", "candidate_k",
        "prediction_mode", "prediction_contract",
        "max_plan_c_descriptors_per_prefix", "max_plan_a_pairs_per_prefix",
        "state_digest_schema", "protocol_id", "qualification_protocol_sha256",
        "candidate_boundary", "plan_a_support_order", "route_cost_points_per_second",
        "continuation_policy", "discount_semantics", "gamma_per_macro")
    cache_key = _hash({
        "scenario_id": record.get("scenario_id"),
        "category": record.get("category"),
        "seed": record["seed"],
        "device": manifest.get("device"),
        "replay_config": {key: manifest.get(key) for key in replay_config_fields},
        "episode_transcript_sha256": record["episode_transcript_sha256"],
        "summary_sha256": summary.get("summary_sha256"),
        "source_identity_sha256": manifest.get("source_identity_sha256"),
        "native_extension_identity": manifest.get("native_extension_identity"),
        "artifact_identity_sha256": manifest.get("artifact_identity_sha256")})
    cache_key = _hash({"cache_key": cache_key,
        "require_hip_plan_a": bool(require_hip_plan_a)})
    cached = _VALIDATED_EPISODE_ELIGIBILITY_CACHE.get(cache_key)
    if cached is not None:
        return list(cached)
    env = make_curriculum_env(scenario_id, category, 1, manifest["device"]["device"],
        seed, horizon=manifest["horizon_physics_ticks"])
    try:
        graph = build_legal_pose_graph(env, slot=3)
        episode_id = f"counterfactual-{scenario_id}-{category}-{seed}"
        _reset_episode(env, seed)
        elapsed = 0
        statuses = []
        ended_episode = False
        for index, expected in enumerate(episode):
            ticks = expected.get("macro_ticks")
            remaining = manifest["horizon_physics_ticks"] - elapsed
            if (type(ticks) is not int or not 1 <= ticks <= manifest["max_macro_ticks"] or
                    ticks > remaining or ticks != min(manifest["max_macro_ticks"], remaining)):
                raise ValueError("episode eligibility transcript has invalid macro ticks")
            actual = _prefix_snapshot(env, graph, episode_id, action_ticks=ticks,
                candidate_k=manifest["candidate_k"],
                prediction_mode=manifest["prediction_mode"],
                max_c_candidates=manifest["max_plan_c_descriptors_per_prefix"],
                max_a_pairs=manifest["max_plan_a_pairs_per_prefix"],
                require_hip_plan_a=require_hip_plan_a)
            for key in ("snapshot_id", "simulator_state_sha256", "observation_sha256",
                    "safe_observation_sha256", "candidate_digest", "candidate_replay_digest",
                    "support_digest", "macro_ticks", "teacher_action", "branch_eligibility"):
                if expected.get(key) != actual.get(key):
                    raise RuntimeError(f"full episode eligibility replay mismatch at decision {index}: {key}")
            statuses.append(actual["branch_eligibility"])
            reward, ended, timeout, trace, _, _, _ = _reward_step(env,
                actual["teacher_action"], ticks)
            if (len(trace) > remaining or
                    trace != expected.get("executed_action_trace") or
                    not math.isfinite(reward) or reward != expected.get("gross_macro_return")):
                raise RuntimeError(f"full episode teacher replay mismatch at decision {index}")
            elapsed += len(trace)
            if (ended or timeout) and index != len(episode) - 1:
                raise RuntimeError("full episode terminated before archived suffix ended")
            if elapsed >= manifest["horizon_physics_ticks"] and index != len(episode) - 1:
                raise RuntimeError("full episode transcript continues beyond the horizon")
            ended_episode = bool(ended or timeout)
        if not ended_episode and elapsed < manifest["horizon_physics_ticks"]:
            raise RuntimeError("full episode transcript ends before termination or configured horizon")
        _VALIDATED_EPISODE_ELIGIBILITY_CACHE[cache_key] = tuple(statuses)
        return statuses
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()


def _replay_counterfactual_branch_seeded(record, manifest, *,
                                        require_hip_plan_a=True):
    """Replay a v2 prefix with strict digest/trace parity and return branch inputs."""
    from .meco_track_training import _reset_episode

    scenario_id, category, seed = (record.get("scenario_id"),
                                   record.get("category"), record.get("seed"))
    if type(seed) is not int or seed < 0:
        raise ValueError("counterfactual record has an invalid seed")
    horizon_ticks = manifest["horizon_physics_ticks"]
    action_ticks = manifest["max_macro_ticks"]
    candidate_k = manifest["candidate_k"]
    prediction_mode = manifest["prediction_mode"]
    max_c = manifest["max_plan_c_descriptors_per_prefix"]
    max_a = manifest["max_plan_a_pairs_per_prefix"]
    device = manifest["device"]["device"]
    decision_index = record["decision_index"]
    transcript = record["prefix_transcript"]
    if (type(decision_index) is not int or decision_index < 0 or
            len(transcript) != decision_index or
            _hash(transcript) != record.get("prefix_transcript_sha256")):
        raise ValueError("counterfactual transcript length or digest mismatch")
    if (sum(item["macro_ticks"] for item in transcript) != record.get("macro_start_tick") or
            record.get("prefix_sha256") != _hash(transcript)):
        raise ValueError("counterfactual prefix digest/tick position mismatch")

    env = make_curriculum_env(scenario_id, category, 1, device, seed,
                              horizon=horizon_ticks)
    try:
        _verify_native_fuel_extension(manifest.get("native_extension_identity"))
        graph = build_legal_pose_graph(env, slot=3)
        episode_id = f"counterfactual-{scenario_id}-{category}-{seed}"
        _reset_episode(env, seed)
        for index, expected in enumerate(transcript):
            ticks = expected["macro_ticks"]
            if type(ticks) is not int or not 1 <= ticks <= action_ticks:
                raise ValueError("counterfactual transcript has invalid macro ticks")
            current = _prefix_snapshot(env, graph, episode_id,
                action_ticks=ticks, candidate_k=candidate_k,
                prediction_mode=prediction_mode, max_c_candidates=max_c,
                max_a_pairs=max_a,
                require_hip_plan_a=require_hip_plan_a)
            if expected.get("branch_eligibility") != current["branch_eligibility"]:
                raise RuntimeError(f"strict v2 eligibility parity mismatch at transcript[{index}]")
            mapping = {"snapshot_id": "snapshot_id",
                "predecision_simulator_state_sha256": "simulator_state_sha256",
                "predecision_observation_sha256": "observation_sha256",
                "safe_observation_sha256": "safe_observation_sha256",
                "candidate_digest": "candidate_digest",
                "candidate_replay_digest": "candidate_replay_digest",
                "support_digest": "support_digest", "macro_ticks": "macro_ticks"}
            for archived, actual in mapping.items():
                if expected.get(archived) != current[actual]:
                    raise RuntimeError(f"strict v2 replay parity mismatch at transcript[{index}].{archived}")
            action = expected.get("action")
            if type(action) is not int or action != current["teacher_action"]:
                raise RuntimeError("teacher prefix action changed during v2 replay")
            if expected.get("candidate_commitment") is not None:
                raise ValueError("v2 teacher transcript cannot contain candidate commitments")
            if type(expected.get("allow_action7_traversal")) is not bool:
                raise ValueError("v2 teacher transcript has invalid traversal semantics")
            reward, ended, timeout, trace, _, _, _ = _reward_step(env, action, ticks,
                goal_xy=expected.get("goal_xy_m"),
                goal_heading=expected.get("goal_heading_rad"),
                allow_action7_traversal=expected.get("allow_action7_traversal", False))
            if trace != expected.get("executed_action_trace"):
                raise RuntimeError("executed action trace changed during v2 prefix replay")
            if not math.isfinite(reward) or reward != expected.get("gross_macro_return"):
                raise RuntimeError("prefix gross return changed during v2 replay")
            if ended or timeout:
                raise RuntimeError("v2 prefix ended before its archived branch point")

        ticks = record.get("macro_ticks")
        if type(ticks) is not int or not 1 <= ticks <= action_ticks:
            raise ValueError("counterfactual branch has invalid macro ticks")
        current = _prefix_snapshot(env, graph, episode_id, action_ticks=ticks,
            candidate_k=candidate_k, prediction_mode=prediction_mode,
            max_c_candidates=max_c, max_a_pairs=max_a, materialize=True,
            require_hip_plan_a=require_hip_plan_a)
        if record.get("branch_eligibility") != current["branch_eligibility"] or not current["branch_eligibility"]["eligible"]:
            raise RuntimeError("strict v2 branch eligibility mismatch or ineligible branch point")
        summary = record.get("seed_branch_summary")
        if not isinstance(summary, dict) or summary.get("minimum_eligible_branches_per_seed") != MIN_ELIGIBLE_BRANCHES_PER_SEED:
            raise ValueError("counterfactual seed eligibility summary is missing")
        if manifest.get("seed_branch_summaries", {}).get(str(seed)) != summary:
            raise RuntimeError("row and manifest seed eligibility summaries differ")
        if decision_index == summary.get("coverage_through_decision_index"):
            statuses = _replay_episode_eligibility(record, manifest,
                require_hip_plan_a=require_hip_plan_a)
            eligible = [i for i, item in enumerate(statuses) if item["eligible"]]
            reasons = {}
            for item in statuses:
                for reason in item["skip_reasons"]:
                    reasons[reason] = reasons.get(reason, 0) + 1
            reconstructed = {"eligible_decision_indices": eligible,
                "coverage_through_decision_index": decision_index,
                "episode_decision_count": len(statuses),
                "covered_prefix_count": len(statuses),
                "skipped_count": sum(not item["eligible"] for item in statuses),
                "skipped_reason_counts": reasons,
                "eligibility_digest": _hash(statuses),
                "minimum_eligible_branches_per_seed": MIN_ELIGIBLE_BRANCHES_PER_SEED}
            reconstructed["summary_sha256"] = _hash(reconstructed)
            if summary != reconstructed or len(eligible) < MIN_ELIGIBLE_BRANCHES_PER_SEED:
                raise RuntimeError("strict v2 seed eligibility summary reconstruction failed")
        state = record.get("branch_state")
        mapping = {"snapshot_id": "snapshot_id",
            "simulator_state_sha256": "simulator_state_sha256",
            "observation_sha256": "observation_sha256",
            "safe_observation_sha256": "safe_observation_sha256",
            "candidate_digest": "candidate_digest",
            "candidate_replay_digest": "candidate_replay_digest",
            "support_digest": "support_digest"}
        if not isinstance(state, dict):
            raise ValueError("counterfactual branch state is missing")
        for archived, actual in mapping.items():
            top_level_value = record.get("predecision_" + archived)
            if (state.get(archived) != current[actual] or
                    (top_level_value is not None and top_level_value != current[actual])):
                raise RuntimeError(f"strict v2 branch-state parity mismatch at {archived}")
        for field in ("candidate_digest", "candidate_replay_digest", "support_digest"):
            if record.get(field) != current[field]:
                raise RuntimeError(f"strict v2 top-level branch parity mismatch at {field}")
        if current["macro_ticks"] != record.get("macro_ticks"):
            raise RuntimeError("strict v2 branch macro-tick mismatch")
        return {"env": env, "graph": graph, "snapshot": current,
                **current["_training_inputs"]}
    except BaseException:
        close = getattr(env, "close", None)
        if callable(close):
            close()
        raise


def _deterministic_sample(indices, limit, key):
    if limit < 1:
        return []
    return sorted(indices, key=lambda index: hashlib.sha256(
        f"{key}:{index}".encode()).digest())[:limit]


def _rebuild_and_check(env, graph, episode_id, expected, *, action_ticks,
                       candidate_k, prediction_mode, max_c_candidates,
                       max_a_pairs):
    actual = _prefix_snapshot(env, graph, episode_id, action_ticks=action_ticks,
        candidate_k=candidate_k, prediction_mode=prediction_mode,
        max_c_candidates=max_c_candidates, max_a_pairs=max_a_pairs,
        require_hip_plan_a=True)
    for field in ("snapshot_id", "macro_ticks", "simulator_state_sha256", "observation_sha256",
                  "safe_observation_sha256", "candidate_digest",
                  "candidate_replay_digest", "support_digest",
                  "candidate_descriptors", "plan_a_descriptors"):
        if actual[field] != expected[field]:
            raise RuntimeError(f"strict prefix replay parity mismatch at {field}")
    return actual


def _branch_return(*, scenario_id, category, seed, device, horizon_ticks,
                   action_ticks, decision_index, prefix, descriptor, kind,
                   candidate_k, prediction_mode, max_c_candidates,
                   max_a_pairs, gamma_per_macro, repeat_seed=None):
    """Evaluate one branch with isolated, reproducible RNG streams."""
    with _counterfactual_rng_scope(seed):
        return _branch_return_seeded(scenario_id=scenario_id, category=category,
            seed=seed, device=device, horizon_ticks=horizon_ticks,
            action_ticks=action_ticks, decision_index=decision_index,
            prefix=prefix, descriptor=descriptor, kind=kind,
            candidate_k=candidate_k, prediction_mode=prediction_mode,
            max_c_candidates=max_c_candidates, max_a_pairs=max_a_pairs,
            gamma_per_macro=gamma_per_macro, repeat_seed=repeat_seed)


def _branch_return_seeded(*, scenario_id, category, seed, device, horizon_ticks,
                   action_ticks, decision_index, prefix, descriptor, kind,
                   candidate_k, prediction_mode, max_c_candidates,
                   max_a_pairs, gamma_per_macro, repeat_seed=None):
    from .meco_track_training import _reset_episode
    env = make_curriculum_env(scenario_id, category, 1, device, seed,
                              horizon=horizon_ticks)
    branch_rng = None
    repeat_rng_provenance = None
    try:
        graph = build_legal_pose_graph(env, slot=3)
        episode_id = f"counterfactual-{scenario_id}-{category}-{seed}"
        _reset_episode(env, seed)
        for row in prefix[:decision_index]:
            current = _rebuild_and_check(env, graph, episode_id, row,
                action_ticks=row["macro_ticks"], candidate_k=candidate_k,
                prediction_mode=prediction_mode,
                max_c_candidates=max_c_candidates, max_a_pairs=max_a_pairs)
            if current["teacher_action"] != row["teacher_action"]:
                raise RuntimeError("teacher prefix action changed during replay")
            _, ended, timeout, trace, _, _, _ = _reward_step(env,
                row["teacher_action"], row["macro_ticks"],
                goal_xy=row["prefix_goal_xy_m"],
                goal_heading=row["prefix_goal_heading_rad"],
                allow_action7_traversal=row["prefix_action7_traversal"])
            if trace != row["executed_action_trace"]:
                raise RuntimeError("executed action trace changed during prefix replay")
            if ended or timeout:
                raise RuntimeError("prefix ended before its archived branch point")
        current = _rebuild_and_check(env, graph, episode_id, prefix[decision_index],
            action_ticks=prefix[decision_index]["macro_ticks"], candidate_k=candidate_k,
            prediction_mode=prediction_mode, max_c_candidates=max_c_candidates,
            max_a_pairs=max_a_pairs)
        branch_ticks = prefix[decision_index]["macro_ticks"]
        if kind == "C":
            choices = current["candidate_descriptors"]
            match = next((item for item in choices
                          if item["descriptor_sha256"] == descriptor["descriptor_sha256"]), None)
            if match is None:
                raise RuntimeError("archived Plan C descriptor is absent after strict replay")
            if repeat_seed is not None:
                branch_rng = _counterfactual_rng_scope(repeat_seed)
                branch_rng.__enter__()
                repeat_rng_provenance = _seed_plan_c_environment_generators(env, repeat_seed)
            reward, ended, timeout, _, _, _, _ = _reward_step(env, descriptor["action"],
                branch_ticks, goal_xy=descriptor["goal_xy_m"])
        else:
            if repeat_seed is not None:
                raise ValueError("matched repeat RNG scopes are only defined for Plan C branches")
            match = next((item for item in current["plan_a_descriptors"]
                          if item["descriptor_sha256"] == descriptor["descriptor_sha256"]), None)
            if match is None:
                raise RuntimeError("archived Plan A descriptor is absent after strict replay")
            reward, ended, timeout, _, _, _, _ = _reward_step(env, descriptor["action"],
                branch_ticks, goal_xy=descriptor["goal_xy_m"],
                goal_heading=descriptor["goal_heading_rad"],
                allow_action7_traversal=(descriptor["action"] == 7))
        discounted = reward
        total_macro_return = reward
        elapsed_ticks = sum(row["macro_ticks"] for row in prefix[:decision_index]) + branch_ticks
        continuation_decisions = 0
        while not (ended or timeout) and elapsed_ticks < horizon_ticks:
            action = int(teacher_actions(env, env._last_obs)[0, 3].item())
            macro_ticks = min(action_ticks, horizon_ticks - elapsed_ticks)
            step_reward, ended, timeout, trace, _, _, _ = _reward_step(
                env, action, macro_ticks)
            if len(trace) != macro_ticks and not (ended or timeout):
                raise RuntimeError("continuation macro ended before its allotted ticks")
            total_macro_return += step_reward
            continuation_decisions += 1
            discounted += (gamma_per_macro ** continuation_decisions) * step_reward
            elapsed_ticks += len(trace)
        return {"gross_macro_return": reward,
            "gross_horizon_return": total_macro_return,
            "discounted_horizon_return": discounted,
            "continuation_policy": CONTINUATION_POLICY,
            "continuation_macro_horizon": continuation_decisions,
            "discount_semantics": COUNTERFACTUAL_DISCOUNT_SEMANTICS,
            "terminal": bool(ended or timeout),
            **({"repeat_rng_provenance": repeat_rng_provenance}
               if repeat_rng_provenance is not None else {})}
    finally:
        try:
            if branch_rng is not None:
                branch_rng.__exit__(None, None, None)
        finally:
            close = getattr(env, "close", None)
            if callable(close):
                close()


def collect_counterfactual_v2(output: Path, *, scenario_id: str, category: str,
        seeds: list[int], qualification_protocol: Path, device=None,
        horizon_ticks=160, action_ticks=12, candidate_k=16,
        prediction_mode="cv", max_c_candidates=8, max_a_pairs=8,
        gamma_per_macro=DEFAULT_GAMMA):
    """Collect matched-prefix v2 labels; any parity mismatch aborts the file."""
    from .meco_supervised import _load_protocol, _protocol_train_seeds
    from .meco_track_training import _reset_episode
    from .curriculum_training import source_identity

    protocol, protocol_id, protocol_sha, protocol_path = _load_protocol(
        Path(qualification_protocol))
    if prediction_mode not in PREDICTION_MODES:
        raise ValueError("invalid prediction mode")
    definition = get_scenario(scenario_id)
    if category not in definition.required_categories:
        raise ValueError("category is not registered for scenario")
    if (not seeds or len(set(seeds)) != len(seeds) or
            any(type(seed) is not int or seed < 0 for seed in seeds) or
            not set(seeds) <= _protocol_train_seeds(protocol, scenario_id, category)):
        raise ValueError("all seeds must be unique members of the frozen train split")
    if (type(horizon_ticks) is not int or type(action_ticks) is not int or
            action_ticks < 1 or horizon_ticks < action_ticks or
            type(candidate_k) is not int or candidate_k not in (8, 16, 32) or
            type(max_c_candidates) is not int or max_c_candidates < 2 or
            type(max_a_pairs) is not int or max_a_pairs < 2 or
            not isinstance(gamma_per_macro, (int, float)) or
            not math.isfinite(gamma_per_macro) or not 0 < gamma_per_macro <= 1):
        raise ValueError("invalid counterfactual horizon, sample limits, or discount")
    from .meco_device import require_indexed_cuda_device
    device = str(require_indexed_cuda_device(device,
        purpose="MECO counterfactual collection"))
    from .fuel_physics_hip import prepare_native_fuel_extension
    prepare_native_fuel_extension(device)
    native_extension_identity = _native_fuel_extension_identity()
    device_metadata = _device_identity(device)
    if (device_metadata["device"].startswith("cuda") and
            native_extension_identity.get("status") != "present"):
        raise RuntimeError("GPU counterfactual collection did not prepare the native HIP extension")
    sources = source_identity()
    if sources.get("_native_fuel_extension") != native_extension_identity:
        raise RuntimeError("native HIP extension identity changed before collection snapshot")
    manifest = {"schema": DATASET_SCHEMA, "target_schema": TARGET_SCHEMA,
        "protocol_id": protocol_id, "qualification_protocol_sha256": protocol_sha,
        "qualification_protocol_file": protocol_path.name,
        "split": "train", "scenario_id": scenario_id, "category": category,
        "seeds": list(seeds), "sources": sources,
        "source_identity_sha256": _hash(sources), "device": device_metadata,
        "native_extension_identity_schema": 1,
        "native_extension_identity": native_extension_identity,
        "artifact_identity_sha256": _artifact_identity_sha256(
            _hash(sources), native_extension_identity),
        "state_digest_schema": STATE_DIGEST_SCHEMA,
        "prediction_mode": prediction_mode,
        "prediction_contract": prediction_contract(prediction_mode),
        "candidate_boundary": CANDIDATE_BOUNDARY,
        "candidate_k": candidate_k,
        "route_cost_points_per_second": ROUTE_COST_POINTS_PER_SECOND,
        "plan_a_support_order": SUPPORT_ORDER,
        "horizon_physics_ticks": horizon_ticks, "max_macro_ticks": action_ticks,
        "gamma_per_macro": gamma_per_macro,
        "discount_semantics": COUNTERFACTUAL_DISCOUNT_SEMANTICS,
        "continuation_policy": CONTINUATION_POLICY,
        "max_plan_c_descriptors_per_prefix": max_c_candidates,
        "max_plan_a_pairs_per_prefix": max_a_pairs,
        "minimum_eligible_branches_per_seed": MIN_ELIGIBLE_BRANCHES_PER_SEED,
        "eligibility_coverage_policy": "prefix through final eligible branch point; later teacher-only suffix excluded",
        "plan_c_target_semantics": "gross simulator reward; exact archived candidate; same deterministic teacher continuation",
        "plan_c_target_schema": PLAN_C_REPEAT_TARGET_SCHEMA,
        "plan_c_repeat_protocol": plan_c_repeat_protocol(protocol_id),
        "plan_a_target_semantics": "gross simulator reward per sampled valid node_behavior_descriptor; no P route-cost label",
        "plan_a_p_target": None,
        "plan_a_p_prior": 0.0,
        "plan_a_p_unlabelled_reason": "simulator executor does not execute P-modified routes",
        "source_runtime_device_bound": True}
    rows = []
    seed_branch_summaries = {}
    for seed in seeds:
        with _counterfactual_rng_scope(seed):
            env = make_curriculum_env(scenario_id, category, 1, device, seed,
                                      horizon=horizon_ticks)
            try:
                _verify_native_fuel_extension(manifest["native_extension_identity"])
                graph = build_legal_pose_graph(env, slot=3)
                episode_id = f"counterfactual-{scenario_id}-{category}-{seed}"
                _reset_episode(env, seed)
                prefix = []
                elapsed_ticks = 0
                while elapsed_ticks < horizon_ticks:
                    macro_ticks = min(action_ticks, horizon_ticks - elapsed_ticks)
                    item = _prefix_snapshot(env, graph, episode_id,
                        action_ticks=macro_ticks, candidate_k=candidate_k,
                        prediction_mode=prediction_mode,
                        max_c_candidates=max_c_candidates,
                        max_a_pairs=max_a_pairs, require_hip_plan_a=True)
                    prefix.append(item)
                    reward, ended, timeout, trace, heading_error, goal_distance, preempted = _reward_step(
                        env, item["teacher_action"], macro_ticks)
                    if len(trace) != macro_ticks and not (ended or timeout):
                        raise RuntimeError("teacher prefix macro ended before its allotted ticks")
                    item.update(prefix_goal_xy_m=None, prefix_goal_heading_rad=None,
                        prefix_action7_traversal=False, executed_action_trace=trace,
                        prefix_gross_return=reward,
                        prefix_terminal_heading_error=heading_error,
                        prefix_terminal_goal_distance_m=goal_distance,
                        prefix_native_goal_preempted=preempted,
                        prefix_candidate_commitment=None)
                    if ended or timeout:
                        break
                    elapsed_ticks += len(trace)
            finally:
                close = getattr(env, "close", None)
                if callable(close):
                    close()
            eligible_indexes = [i for i, snapshot in enumerate(prefix)
                                if snapshot["branch_eligibility"]["eligible"]]
            if len(eligible_indexes) < MIN_ELIGIBLE_BRANCHES_PER_SEED:
                raise RuntimeError(f"seed {seed} has {len(eligible_indexes)} eligible branch points; "
                    f"minimum is {MIN_ELIGIBLE_BRANCHES_PER_SEED}")
            coverage_end = eligible_indexes[-1]
            episode_transcript = [{
                "snapshot_id": row["snapshot_id"],
                "simulator_state_sha256": row["simulator_state_sha256"],
                "observation_sha256": row["observation_sha256"],
                "safe_observation_sha256": row["safe_observation_sha256"],
                "candidate_digest": row["candidate_digest"],
                "candidate_replay_digest": row["candidate_replay_digest"],
                "support_digest": row["support_digest"],
                "macro_ticks": row["macro_ticks"],
                "teacher_action": row["teacher_action"],
                "branch_eligibility": row["branch_eligibility"],
                "executed_action_trace": row["executed_action_trace"],
                "gross_macro_return": row["prefix_gross_return"]}
                for row in prefix]
            eligibility_list = [x["branch_eligibility"] for x in episode_transcript]
            reason_counts = {}
            for eligibility in eligibility_list:
                for reason in eligibility["skip_reasons"]:
                    reason_counts[reason] = reason_counts.get(reason, 0) + 1
            summary = {"eligible_decision_indices": eligible_indexes,
                "coverage_through_decision_index": coverage_end,
                "episode_decision_count": len(prefix),
                "covered_prefix_count": len(prefix),
                "skipped_count": sum(not x["eligible"] for x in eligibility_list),
                "skipped_reason_counts": reason_counts,
                "eligibility_digest": _hash(eligibility_list),
                "minimum_eligible_branches_per_seed": MIN_ELIGIBLE_BRANCHES_PER_SEED}
            summary["summary_sha256"] = _hash(summary)
            seed_branch_summaries[str(seed)] = summary
            for decision_index, item in enumerate(prefix):
                if decision_index not in eligible_indexes:
                    continue
                prefix_transcript = [{
                    "predecision_simulator_state_sha256": row["simulator_state_sha256"],
                    "predecision_observation_sha256": row["observation_sha256"],
                    "safe_observation_sha256": row["safe_observation_sha256"],
                    "candidate_digest": row["candidate_digest"],
                    "candidate_replay_digest": row["candidate_replay_digest"],
                    "support_digest": row["support_digest"],
                    "snapshot_id": row["snapshot_id"],
                    "macro_ticks": row["macro_ticks"],
                    "action": row["teacher_action"],
                    "goal_xy_m": row["prefix_goal_xy_m"],
                    "goal_heading_rad": row["prefix_goal_heading_rad"],
                    "allow_action7_traversal": row["prefix_action7_traversal"],
                    "candidate_commitment": row["prefix_candidate_commitment"],
                    "executed_action_trace": row["executed_action_trace"],
                    "gross_macro_return": row["prefix_gross_return"],
                    "branch_eligibility": row["branch_eligibility"]}
                    for row in prefix[:decision_index]]
                group_summary = summary
                group = {"schema": DATASET_SCHEMA, "target_schema": TARGET_SCHEMA,
                    "scenario_id": scenario_id, "category": category, "seed": seed,
                    "decision_index": decision_index,
                    "prefix_sha256": _hash(prefix_transcript),
                    "prefix_transcript_sha256": _hash(prefix_transcript),
                    "protocol_id": protocol_id,
                    "qualification_protocol_sha256": protocol_sha,
                    "source_identity_sha256": manifest["source_identity_sha256"],
                    "device": device_metadata,
                    "predecision_simulator_state_sha256": item["simulator_state_sha256"],
                    "predecision_observation_sha256": item["observation_sha256"],
                    "candidate_digest": item["candidate_digest"],
                    "candidate_replay_digest": item["candidate_replay_digest"],
                    "support_digest": item["support_digest"],
                    "macro_ticks": item["macro_ticks"],
                    "macro_start_tick": sum(row["macro_ticks"] for row in prefix[:decision_index]),
                    "max_macro_ticks": action_ticks,
                    "horizon_physics_ticks": horizon_ticks,
                    "gamma_per_macro": gamma_per_macro,
                    "discount_semantics": manifest["discount_semantics"],
                    "branch_state": {
                        "snapshot_id": item["snapshot_id"],
                        "simulator_state_sha256": item["simulator_state_sha256"],
                        "observation_sha256": item["observation_sha256"],
                        "safe_observation_sha256": item["safe_observation_sha256"],
                        "candidate_digest": item["candidate_digest"],
                        "candidate_replay_digest": item["candidate_replay_digest"],
                        "support_digest": item["support_digest"]},
                    "prefix_transcript": prefix_transcript,
                    "branch_eligibility": item["branch_eligibility"],
                    "plan_c_target_schema": manifest["plan_c_target_schema"],
                    "plan_c_repeat_protocol_id": manifest["plan_c_repeat_protocol"]["repeat_protocol_id"],
                    "seed_branch_summary": group_summary}
                if decision_index == coverage_end:
                    group["episode_transcript"] = episode_transcript
                    group["episode_transcript_sha256"] = _hash(episode_transcript)
                c_outcomes = []
                for descriptor in item["candidate_descriptors"]:
                    repeats = []
                    for repeat_index in range(PLAN_C_MATCHED_REPEAT_COUNT):
                        repeat_seed = plan_c_repeat_seed(protocol_id=protocol_id,
                            scenario_id=scenario_id, category=category, seed=seed,
                            decision_index=decision_index, repeat_index=repeat_index)
                        outcome = _branch_return(scenario_id=scenario_id,
                            category=category, seed=seed, device=device,
                            horizon_ticks=horizon_ticks, action_ticks=action_ticks,
                            decision_index=decision_index, prefix=prefix,
                            descriptor=descriptor, kind="C", candidate_k=candidate_k,
                            prediction_mode=prediction_mode,
                            max_c_candidates=max_c_candidates,
                            max_a_pairs=max_a_pairs,
                            gamma_per_macro=gamma_per_macro, repeat_seed=repeat_seed)
                        outcome.update(repeat_index=repeat_index, repeat_seed=repeat_seed)
                        repeats.append(outcome)
                        _verify_native_fuel_extension(manifest["native_extension_identity"])
                    c_outcomes.append(_plan_c_repeated_outcome(descriptor, repeats))
                c_pairs = _plan_c_pairwise_labels(c_outcomes)
                a_outcomes = []
                for descriptor in item["plan_a_descriptors"]:
                    outcome = _branch_return(scenario_id=scenario_id, category=category,
                        seed=seed, device=device, horizon_ticks=horizon_ticks,
                        action_ticks=action_ticks, decision_index=decision_index,
                        prefix=prefix, descriptor=descriptor, kind="A",
                        candidate_k=candidate_k, prediction_mode=prediction_mode,
                        max_c_candidates=max_c_candidates, max_a_pairs=max_a_pairs,
                        gamma_per_macro=gamma_per_macro)
                    _verify_native_fuel_extension(manifest["native_extension_identity"])
                    a_outcomes.append({"descriptor": descriptor, **outcome,
                        "p_target": None, "p_prior": 0.0,
                        "p_unlabelled_reason": manifest["plan_a_p_unlabelled_reason"]})
                a_pairs = [{"left_descriptor_sha256": left["descriptor"]["descriptor_sha256"],
                    "right_descriptor_sha256": right["descriptor"]["descriptor_sha256"],
                    "gross_horizon_return_difference": left["gross_horizon_return"] - right["gross_horizon_return"],
                    "preferred_descriptor_sha256": (left["descriptor"]["descriptor_sha256"]
                        if left["gross_horizon_return"] > right["gross_horizon_return"] else
                        right["descriptor"]["descriptor_sha256"]
                        if right["gross_horizon_return"] > left["gross_horizon_return"] else None)}
                    for index, left in enumerate(a_outcomes)
                    for right in a_outcomes[index + 1:]]
                rows.append({**group, "plan_c_counterfactuals": c_outcomes,
                    "plan_c_pairwise_labels": c_pairs,
                    "plan_a_sampled_node_behavior_labels": a_outcomes,
                    "plan_a_pairwise_preferences": a_pairs,
                    "branch_teacher_action_reference": item["teacher_action"],
                    "teacher_prefix_provenance": CONTINUATION_POLICY})
    _verify_native_fuel_extension(manifest["native_extension_identity"])
    if source_identity() != sources:
        raise RuntimeError("source/runtime changed during counterfactual collection")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n"
                      for row in rows).encode()
    output.write_bytes(payload)
    manifest["record_count"] = len(rows)
    manifest["seed_branch_summaries"] = seed_branch_summaries
    manifest["dataset_sha256"] = hashlib.sha256(payload).hexdigest()
    manifest["records_have_prefix_replay_parity"] = True
    output.with_suffix(output.suffix + ".manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    return {"path": str(output), "record_count": len(rows),
            "dataset_sha256": manifest["dataset_sha256"],
            "schema": DATASET_SCHEMA}


def _legal_node_pair_return(record, manifest, pair_index, repeat_index):
    """Replay a branch and execute one pair from the full legal Plan A support."""
    replay = replay_counterfactual_branch(record, manifest,
        require_hip_plan_a=True)
    env = replay["env"]
    repeat_seed = int.from_bytes(hashlib.sha256(
        f"legal-node-repeat:{record['seed']}:{repeat_index}".encode()).digest()[:4], "big")
    branch_rng = _counterfactual_rng_scope(repeat_seed)
    branch_rng.__enter__()
    try:
        support = replay["support"]
        if (support.support_digest != record["support_digest"] or
                pair_index < 0 or pair_index >= support.valid_mask.numel() or
                not bool(support.valid_mask[pair_index])):
            raise RuntimeError("legal-node pair is absent from full replayed support")
        node = int(support.pair_node_ids[pair_index].item())
        action = int(support.action_ids[pair_index].item())
        requested_action = action
        goal_xy = [float(x) for x in support.goal_xy[node].tolist()]
        goal_heading = float(support.goal_heading[node].item())
        goal_track_index = None
        goal_track_source_xy = None
        if int(support.behavior_ids[pair_index].item()) == 1:
            descriptor_id = int(support.descriptor_ids[pair_index].item())
            target_indices, target_available, _ = env._candidates()
            if (descriptor_id < 0 or descriptor_id >= target_indices.shape[-1] or
                    not bool(target_available[0, 3, descriptor_id].item())):
                return {"status": "unavailable", "unavailable_reason":
                    "selected intake descriptor is no longer sensed/available",
                    "requested_action": requested_action, "repeat_seed": repeat_seed}
            goal_track_index = int(target_indices[0, 3, descriptor_id].item())
            goal_track_source_xy = [float(x) for x in
                env.track_pos[0, 3, goal_track_index].tolist()]
        ticks = record["macro_ticks"]
        reward, ended, timeout, trace, _, _, preempted = _reward_step(env, action, ticks,
            goal_xy=goal_xy, goal_heading=goal_heading,
            goal_track_index=goal_track_index, goal_track_source_xy=goal_track_source_xy,
            allow_action7_traversal=(action == 7))
        target_valid = getattr(env, "_last_planner_collection_target_valid", None)
        action_match = getattr(env, "_last_planner_collection_action_match", None)
        target_lost = (isinstance(target_valid, torch.Tensor) and
                       not bool(target_valid.item()))
        action_remasked = (isinstance(action_match, torch.Tensor) and
                           not bool(action_match.item()))
        if (len(trace) != ticks or any(executed != action for executed in trace) or
                preempted or target_lost or action_remasked):
            return {"status": "unavailable", "unavailable_reason":
                "requested action or goal was preempted, remasked, or not executed for full macro",
                "requested_action": requested_action, "executed_action_trace": list(trace),
                "executed_branch_ticks": len(trace), "repeat_seed": repeat_seed,
                "goal_preempted": bool(preempted), "target_lost": bool(target_lost),
                "action_remasked": bool(action_remasked)}
        branch_trace = list(trace)
        branch_ended, branch_timeout = bool(ended), bool(timeout)
        gross = discounted = reward
        elapsed = record["macro_start_tick"] + ticks
        continuation = 0
        while not (ended or timeout) and elapsed < manifest["horizon_physics_ticks"]:
            action = int(teacher_actions(env, env._last_obs)[0, 3].item())
            macro_ticks = min(manifest["max_macro_ticks"],
                manifest["horizon_physics_ticks"] - elapsed)
            step_reward, ended, timeout, trace, _, _, _ = _reward_step(env, action, macro_ticks)
            if len(trace) != macro_ticks and not (ended or timeout):
                raise RuntimeError("continuation macro ended before allotted ticks")
            gross += step_reward
            continuation += 1
            discounted += manifest["gamma_per_macro"] ** continuation * step_reward
            elapsed += len(trace)
        return {"gross_horizon_return": gross,
            "discounted_horizon_return": discounted,
            "status": "available", "requested_action": requested_action,
            "executed_action_trace": branch_trace,
            "executed_branch_ticks": len(branch_trace),
            "branch_terminal": branch_ended, "branch_timeout": branch_timeout,
            "repeat_seed": repeat_seed,
            "continuation_macro_horizon": continuation,
            "continuation_policy": CONTINUATION_POLICY,
            "discount_semantics": COUNTERFACTUAL_DISCOUNT_SEMANTICS,
            "terminal": bool(ended or timeout)}
    finally:
        close = getattr(env, "close", None)
        try:
            if callable(close): close()
        finally:
            branch_rng.__exit__(None, None, None)


def collect_plan_a_legal_nodes_v1(output: Path, *, dataset: Path,
        qualification_protocol: Path, device: str, expert_goal_source: Path | None = None):
    """Collect simulator-only labels over a deterministic bounded legal-node subset."""
    from .curriculum_training import source_identity
    from .meco_supervised import _load_protocol, _read_counterfactual_dataset
    from .meco_track_training import _atomic_json
    from .meco_device import require_indexed_cuda_device

    device = str(require_indexed_cuda_device(device, purpose="Plan A legal-node collection"))
    sources = source_identity()
    rows, source_sha, source_manifest_sha, _, protocol_id, protocol_sha, source_manifest = \
        _read_counterfactual_dataset(Path(dataset), qualification_protocol=Path(qualification_protocol),
            expected_sources=sources)
    if source_manifest["device"]["device"] != device:
        raise ValueError("legal-node collection device must match frozen source dataset")
    _, loaded_id, loaded_sha, protocol_path = _load_protocol(Path(qualification_protocol))
    if (loaded_id != protocol_id or loaded_sha != protocol_sha or source_manifest["split"] != "train"):
        raise ValueError("legal-node source protocol/train split mismatch")
    expert_rows_by_key = None
    expert_source_sha = expert_manifest_sha = None
    expert_source_path = None
    if expert_goal_source is not None:
        expert_source_path = Path(expert_goal_source).resolve()
        expert_source_bytes = expert_source_path.read_bytes()
        expert_manifest_path = expert_source_path.with_suffix(expert_source_path.suffix + ".manifest.json")
        expert_manifest_bytes = expert_manifest_path.read_bytes()
        expert_manifest = json.loads(expert_manifest_bytes)
        expert_source_sha = hashlib.sha256(expert_source_bytes).hexdigest()
        expert_manifest_sha = hashlib.sha256(expert_manifest_bytes).hexdigest()
        if (expert_manifest.get("schema") != PLAN_A_EXPERT_GOAL_SOURCE_SCHEMA or
                expert_manifest.get("target_schema") != PLAN_A_EXPERT_GOAL_SOURCE_TARGET_SCHEMA or
                expert_manifest.get("base_dataset_sha256") != source_sha or
                expert_manifest.get("base_manifest_sha256") != source_manifest_sha or
                expert_manifest.get("protocol_id") != protocol_id or
                expert_manifest.get("qualification_protocol_sha256") != protocol_sha or
                expert_manifest.get("sources") != sources or
                expert_manifest.get("source_identity_sha256") != _hash(sources) or
                expert_manifest.get("device") != source_manifest.get("device") or
                expert_manifest.get("split") != "train" or
                expert_manifest.get("dataset_sha256") != expert_source_sha):
            raise ValueError("expert-goal source does not bind the exact legal-node source/protocol/device")
        expert_rows = [json.loads(line) for line in expert_source_bytes.decode().splitlines() if line]
        if len(expert_rows) != expert_manifest.get("record_count"):
            raise ValueError("expert-goal source record count mismatch")
        expert_rows_by_key = {}
        for goal_row in expert_rows:
            key = (goal_row.get("scenario_id"), goal_row.get("category"),
                goal_row.get("seed"), goal_row.get("decision_index"))
            if key in expert_rows_by_key:
                raise ValueError("expert-goal source contains duplicate branch identities")
            expert_rows_by_key[key] = goal_row
        source_keys = {(r["scenario_id"], r["category"], r["seed"], r["decision_index"])
            for r in rows}
        if set(expert_rows_by_key) != source_keys:
            raise ValueError("expert-goal source does not cover exact legal-node source rows")
    out_rows = []
    preflight_rollout_bound = len(rows) * PLAN_A_LEGAL_NODE_MAX_BRANCHES_PER_PREFIX
    if preflight_rollout_bound > PLAN_A_LEGAL_NODE_MAX_DATASET_BRANCHES:
        raise ValueError(f"legal-node deterministic preflight bound {preflight_rollout_bound} exceeds "
            f"dataset cap {PLAN_A_LEGAL_NODE_MAX_DATASET_BRANCHES}; partition the source dataset")
    actual_rollouts = 0
    for record in rows:
        if not record.get("plan_a_descriptors"): continue
        replay = replay_counterfactual_branch(record, source_manifest,
            require_hip_plan_a=True)
        env = replay["env"]
        graph = replay["graph"]
        try:
            support = replay["support"]
            nodes = [int(x) for x in support.node_ids.tolist()]
            pair_nodes = [int(x) for x in support.pair_node_ids.tolist()]
            pair_behaviors = [int(x) for x in support.behavior_ids.tolist()]
            pair_descriptors = [int(x) for x in support.descriptor_ids.tolist()]
            pair_actions = [int(x) for x in support.action_ids.tolist()]
            valid = [bool(x) for x in support.valid_mask.tolist()]
            reachable = [bool(math.isfinite(float(x))) for x in support.distances_m.tolist()]
            feasible_nodes = sorted({pair_nodes[i] for i, ok in enumerate(valid) if ok and reachable[pair_nodes[i]]})
            if not feasible_nodes:
                raise RuntimeError("legal-node branch has N=0 feasible reachable nodes")
            goal_row = None
            expert_node = None
            if expert_rows_by_key is not None:
                record_key = (record["scenario_id"], record["category"],
                    record["seed"], record["decision_index"])
                goal_row = expert_rows_by_key[record_key]
                if (goal_row.get("schema") != PLAN_A_EXPERT_GOAL_SOURCE_SCHEMA or
                        goal_row.get("base_record") != record or
                        goal_row.get("snapshot_id") != replay["snapshot"]["snapshot_id"] or
                        goal_row.get("state_sha256") != replay["snapshot"]["simulator_state_sha256"] or
                        goal_row.get("support_digest") != support.support_digest or
                        goal_row.get("execution_failure") is not False):
                    raise ValueError("expert-goal row is not bound to the exact replayed legal-node branch")
                selected_goal = goal_row.get("selected_goal")
                if not isinstance(selected_goal, dict) or type(selected_goal.get("node_id")) is not int:
                    raise ValueError("expert-goal row has no selected legal node")
                expert_node = selected_goal["node_id"]
                if expert_node not in feasible_nodes:
                    raise ValueError("expert selected goal is not feasible in exact legal support")
            behavior_count = max(1, len({pair_behaviors[i] for i, ok in enumerate(valid)
                if ok and pair_nodes[i] in feasible_nodes}))
            requested = max(1, len(feasible_nodes) // 16)
            # Preserve a complete feasible-behavior panel at each selected
            # node while bounding simulator interventions per prefix.
            budget_node_limit = max(1, PLAN_A_LEGAL_NODE_MAX_BRANCHES_PER_PREFIX //
                (behavior_count * PLAN_A_LEGAL_NODE_REPEATS))
            limit = min(requested, budget_node_limit)
            headings = [int(x) for x in graph.heading_bin.tolist()]
            regions = [int(x) for x in graph.region_ids.reshape(-1).tolist()]
            key = (f"legal-node-v1:{protocol_id}:{record['scenario_id']}:{record['category']}:"
                f"{record['seed']}:{record['decision_index']}:{support.support_digest}")
            strata = {}
            for node in feasible_nodes:
                strata.setdefault((headings[node], regions[node]), []).append(node)
            ordered_strata = sorted(strata, key=lambda cell: hashlib.sha256(
                f"{key}:stratum:{cell[0]}:{cell[1]}".encode()).digest())
            selected = []
            mandatory_nodes = set()
            if support.root_node in feasible_nodes:
                mandatory_nodes.add(support.root_node)
            if expert_node is not None:
                mandatory_nodes.add(expert_node)
            selected.extend(sorted(mandatory_nodes))
            limit = max(limit, len(selected))
            depth = 0
            while len(selected) < limit:
                added = False
                for cell in ordered_strata:
                    candidates = sorted(strata[cell], key=lambda node: hashlib.sha256(
                        f"{key}:node:{node}".encode()).digest())
                    if depth < len(candidates) and candidates[depth] not in selected:
                        selected.append(candidates[depth]); added = True
                        if len(selected) >= limit: break
                if not added: break
                depth += 1
            selected = sorted(selected)
            selected_set = set(selected)
            stratum_selected_counts = {}
            for selected_node in selected:
                cell = (headings[selected_node], regions[selected_node])
                stratum_selected_counts[cell] = stratum_selected_counts.get(cell, 0) + 1
            node_pi = {}
            for selected_node in selected:
                cell = (headings[selected_node], regions[selected_node])
                forced_in_stratum = sum(node in mandatory_nodes for node in strata[cell])
                if selected_node in mandatory_nodes:
                    pi = 1.0
                else:
                    population = len(strata[cell]) - forced_in_stratum
                    sample = stratum_selected_counts[cell] - forced_in_stratum
                    pi = sample / population if population else 1.0
                node_pi[selected_node] = pi
            if any(not (0. < pi <= 1.) for pi in node_pi.values()):
                raise RuntimeError("legal-node stratum inclusion probability is invalid")
            pairs = []
            for node in selected:
                behaviors_here = sorted({pair_behaviors[i] for i, ok in enumerate(valid)
                    if ok and pair_nodes[i] == node})
                for behavior in behaviors_here:
                    candidates = [i for i, ok in enumerate(valid)
                        if ok and pair_nodes[i] == node and pair_behaviors[i] == behavior]
                    pair_index = min(candidates, key=lambda i: hashlib.sha256(
                        f"{key}:descriptor:{node}:{behavior}:{pair_descriptors[i]}".encode()).digest())
                    pair = {"pair_index": pair_index, "node_id": node,
                        "behavior_id": behavior, "descriptor_id": pair_descriptors[pair_index],
                        "action": pair_actions[pair_index],
                        "goal_xy_m": [float(x) for x in support.goal_xy[node].tolist()],
                        "goal_heading_rad": float(support.goal_heading[node].item()),
                        "behavior_descriptor_support_count": len(candidates),
                        "descriptor_inclusion_probability": 1.0 / len(candidates),
                        "descriptor_inverse_probability_weight": len(candidates),
                        "descriptor_sampling": "uniform protocol-hash rank; one draw from D; inclusion probability 1/D; uniform-draw estimand, no D multiplier in fitting"}
                    pairs.append(pair)
            rollout_count = len(pairs) * PLAN_A_LEGAL_NODE_REPEATS
            if rollout_count > PLAN_A_LEGAL_NODE_MAX_BRANCHES_PER_PREFIX:
                raise RuntimeError("legal-node branch budget exceeded after deterministic selection")
            actual_rollouts += rollout_count
            if actual_rollouts > PLAN_A_LEGAL_NODE_MAX_DATASET_BRANCHES:
                raise RuntimeError("legal-node dataset rollout cap exceeded")
            layout = [{"pair_index": i, "node_id": pair_nodes[i],
                "behavior_id": pair_behaviors[i], "descriptor_id": pair_descriptors[i],
                "action": pair_actions[i], "feasible": valid[i]} for i in range(len(valid))]
            outcomes = []
            for pair in pairs:
                repeats = [_legal_node_pair_return(record, source_manifest,
                    pair["pair_index"], repeat) for repeat in range(PLAN_A_LEGAL_NODE_REPEATS)]
                if all(x.get("status") == "available" for x in repeats):
                    outcomes.append({**pair, "status": "available",
                        "repeat_count": PLAN_A_LEGAL_NODE_REPEATS,
                        "repeat_returns": repeats,
                        "gross_horizon_return": sum(x["gross_horizon_return"] for x in repeats) / len(repeats)})
                else:
                    outcomes.append({**pair, "status": "unavailable",
                        "unavailable_reason": next(x["unavailable_reason"] for x in repeats
                            if x.get("status") == "unavailable"), "repeat_outcomes": repeats})
            by_behavior = {}
            for outcome in outcomes:
                if outcome["status"] != "available": continue
                by_behavior.setdefault((outcome["node_id"], outcome["behavior_id"]), []).append(outcome)
            behavior_domain = list(range(5))
            node_targets, preferences = [], []
            for node in selected:
                feasible_behaviors = sorted({pair_behaviors[i] for i, ok in enumerate(valid)
                    if ok and pair_nodes[i] == node})
                repeat_means = {b: [sum(item["repeat_returns"][r]["gross_horizon_return"]
                    for item in values) / len(values) for r in range(PLAN_A_LEGAL_NODE_REPEATS)]
                    for (n, b), values in by_behavior.items() if n == node}
                means = {b: sum(values) / len(values) for b, values in repeat_means.items()}
                variances = {b: (sum((x-means[b])**2 for x in values) / len(values))
                    for b, values in repeat_means.items()}
                censored_behaviors = sorted(set(feasible_behaviors) - set(means))
                u_estimable = bool(feasible_behaviors) and not censored_behaviors
                # One uniform descriptor draw estimates each behavior's
                # descriptor-average return; feasible behaviors get equal weight.
                node_targets.append({"node_id": node,
                    "u_target": sum(means[b] for b in feasible_behaviors) / len(feasible_behaviors)
                        if u_estimable else None,
                    "status": "available" if u_estimable else "unavailable",
                    "node_inclusion_probability": node_pi[node],
                    "node_inverse_probability_weight": 1. / node_pi[node],
                    "feasible_behavior_ids": feasible_behaviors,
                    "available_behavior_ids": sorted(means),
                    "unavailable_behavior_ids": sorted(set(behavior_domain) - set(feasible_behaviors)),
                    "censored_feasible_behavior_ids": censored_behaviors,
                    "behavior_mean_returns": means, "behavior_repeat_return_variances": variances,
                    "behavior_repeat_mean_returns": repeat_means,
                    "behavior_weighting": "equal over feasible behaviors; one uniform hash-ranked descriptor estimates each behavior mean",
                    "u_oracle_version": PLAN_A_U_ORACLE_VERSION,
                    "u_evidence_sha256": None})
                node_targets[-1]["u_evidence_sha256"] = _plan_a_u_evidence_sha256(
                    support.support_digest, node_targets[-1], outcomes)
                available = sorted(means)
                for i, left in enumerate(available):
                    for right in available[i + 1:]:
                        repeat_diffs = [repeat_means[left][r] - repeat_means[right][r]
                            for r in range(PLAN_A_LEGAL_NODE_REPEATS)]
                        diff = means[left] - means[right]
                        signs = [0 if value == 0 else 1 if value > 0 else -1
                            for value in repeat_diffs]
                        stable = len(set(signs)) == 1 and signs[0] != 0
                        status = ("comparable" if stable else
                            "tie" if all(value == 0 for value in repeat_diffs) else "uncertain")
                        preferences.append({"node_id": node, "left_behavior_id": left,
                            "right_behavior_id": right, "left_mean_return": means[left],
                            "right_mean_return": means[right],
                            "paired_repeat_differences": repeat_diffs,
                            "paired_repeat_difference_variance": sum((x-diff)**2 for x in repeat_diffs)/len(repeat_diffs),
                            "preferred_behavior_id": (left if signs[0] > 0 else right) if stable else None,
                            "status": status,
                            "stability_rule": "both paired repeats have the same nonzero sign; no calibrated noise margin; otherwise uncertain"})
            out_rows.append({"schema": PLAN_A_LEGAL_NODE_DATASET_SCHEMA,
                "target_schema": PLAN_A_LEGAL_NODE_TARGET_SCHEMA,
                "u_oracle_version": PLAN_A_U_ORACLE_VERSION,
                "scenario_id": record["scenario_id"], "category": record["category"],
                "seed": record["seed"], "decision_index": record["decision_index"],
                "macro_start_tick": record["macro_start_tick"],
                "snapshot_id": replay["snapshot"]["snapshot_id"],
                "state_sha256": replay["snapshot"]["simulator_state_sha256"],
                "support_digest": support.support_digest, "canonical_node_ids": nodes,
                "canonical_poses": [{"node_id": i,
                    "x_m": float(graph.xy[i, 0].item()), "y_m": float(graph.xy[i, 1].item()),
                    "heading_bin": headings[i], "heading_rad": float(graph.heading[i].item()),
                    "region_id": regions[i]} for i in nodes],
                "reachable_mask": reachable, "full_pair_layout": layout,
                "full_pair_layout_sha256": _hash(layout), "feasible_node_ids": feasible_nodes,
                "pair_feasibility_digest": _hash(valid),
                "selected_node_ids": selected, "selected_node_coverage_fraction": len(selected)/len(feasible_nodes),
                "selected_heading_region_coverage": sorted({f"{headings[n]}:{regions[n]}" for n in selected}),
                "stratum_population_counts": {f"{h}:{r}": len(strata[(h, r)]) for h, r in strata},
                "stratum_selected_counts": {f"{h}:{r}": stratum_selected_counts.get((h, r), 0)
                    for h, r in strata},
                "root_node": support.root_node,
                "root_node_behavior_supported": support.root_node in feasible_nodes,
                "root_node_unavailable_reason": None if support.root_node in feasible_nodes else "root has no feasible legal behavior pair",
                "selection_rule": "min(max(1,floor(N/16)),floor(512/(2*feasible_behavior_count))); deterministic heading-bin x graph-region stratification, feasible reachable root included",
                "node_inclusion_probabilities": {str(n): node_pi[n] for n in selected},
                "selection_digest": _hash({"protocol_id": protocol_id, "key": key,
                    "support_digest": support.support_digest, "selected_node_ids": selected,
                    "selected_pair_indices": [p["pair_index"] for p in pairs]}),
                "maximum_branch_rollouts": PLAN_A_LEGAL_NODE_MAX_BRANCHES_PER_PREFIX,
                "pairs": outcomes, "node_u_targets": node_targets,
                "behavior_preferences": preferences, "p_target": None, "p_prior": 0.0})
        finally:
            close = getattr(env, "close", None)
            if callable(close): close()
    if not out_rows: raise ValueError("source contains no replayable Plan A legal-node support")
    import os, tempfile
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    raw = "".join(json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False)+"\n"
        for row in out_rows).encode()
    fd, temp_name = tempfile.mkstemp(prefix=f".{output.name}.", suffix=".tmp", dir=output.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(raw); stream.flush(); os.fsync(stream.fileno())
        os.replace(temp_name, output)
    except BaseException:
        try: os.unlink(temp_name)
        except FileNotFoundError: pass
        raise
    manifest = {"schema": PLAN_A_LEGAL_NODE_DATASET_SCHEMA,
        "target_schema": PLAN_A_LEGAL_NODE_TARGET_SCHEMA, "split": "train",
        "u_oracle_version": PLAN_A_U_ORACLE_VERSION,
        "u_oracle_semantics": PLAN_A_U_ORACLE_SEMANTICS,
        "protocol_id": protocol_id, "qualification_protocol_sha256": protocol_sha,
        "qualification_protocol_file": protocol_path.name, "sources": sources,
        "source_identity_sha256": _hash(sources), "source_dataset_sha256": source_sha,
        "source_manifest_sha256": source_manifest_sha, "source_schema": DATASET_SCHEMA,
        "source_dataset": Path(dataset).name, "device": source_manifest["device"],
        "prediction_mode": source_manifest["prediction_mode"],
        "prediction_contract": source_manifest["prediction_contract"],
        "horizon_physics_ticks": source_manifest["horizon_physics_ticks"],
        "max_macro_ticks": source_manifest["max_macro_ticks"],
        "gamma_per_macro": source_manifest["gamma_per_macro"],
        "discount_semantics": COUNTERFACTUAL_DISCOUNT_SEMANTICS,
        "continuation_policy": CONTINUATION_POLICY, "support_order": SUPPORT_ORDER,
        "preference_stability_rule": "both paired repeats have same nonzero sign; exact all-repeat zero is tie; all other pairs uncertain; zero noise threshold because no calibrated margin exists",
        "coverage_rule": ("min(max(1,floor(N/16)),floor(512/(2*feasible_behavior_count))) nodes; one uniform hash-ranked descriptor per feasible behavior and two paired repeat outcomes; descriptor inclusion probabilities/weights are provenance for uniform-draw U/B estimands, with no descriptor reweighting; heading-bin x graph-region stratification; feasible reachable root included when available" + ("; replay-bound expert selected goal included as a certainty unit" if expert_source_path else "")),
        "branch_rollout_cap": PLAN_A_LEGAL_NODE_MAX_BRANCHES_PER_PREFIX,
        "paired_repeats": PLAN_A_LEGAL_NODE_REPEATS,
        "dataset_rollout_cap": PLAN_A_LEGAL_NODE_MAX_DATASET_BRANCHES,
        "preflight_rollout_upper_bound": preflight_rollout_bound,
        "actual_branch_rollouts": actual_rollouts,
        "expert_goal_source": (os.path.relpath(expert_source_path, Path(output).resolve().parent)
            if expert_source_path else None),
        "expert_goal_source_sha256": expert_source_sha,
        "expert_goal_source_manifest_sha256": expert_manifest_sha,
        "p_target": None, "p_prior": 0.0,
        "p_unlabelled_reason": "independent P-v3 oracle labels are not supplied by this dataset",
        "record_count": len(out_rows), "dataset_sha256": hashlib.sha256(raw).hexdigest()}
    _atomic_json(output.with_suffix(output.suffix + ".manifest.json"), manifest)
    return {"path": str(output), "record_count": len(out_rows),
        "dataset_sha256": manifest["dataset_sha256"], "coverage_rule": manifest["coverage_rule"]}


def _select_plan_a_expert_goal_v1(support):
    """Choose the nearest reachable feasible pair; ties follow support order.

    This deterministic rule records a feasible behavior preference only. It is
    not an optimality claim and does not supply an expert U target.
    """
    pairs = torch.nonzero(support.valid_mask, as_tuple=False).reshape(-1).tolist()
    ranked = [(float(support.route_cost[int(support.pair_node_ids[pair])].item()), pair)
        for pair in pairs]
    ranked = [(distance, pair) for distance, pair in ranked if math.isfinite(distance)]
    if not ranked:
        raise RuntimeError("Plan A expert-goal selector found no feasible supported pair")
    _, pair = min(ranked, key=lambda item: (item[0], item[1]))
    node = int(support.pair_node_ids[pair].item())
    selected = {"node_id": node,
        "behavior_id": int(support.behavior_ids[pair].item()),
        "descriptor_id": int(support.descriptor_ids[pair].item()),
        "action": int(support.action_ids[pair].item()),
        "goal_xy_m": [float(x) for x in support.goal_xy[node].tolist()],
        "goal_heading_rad": float(support.goal_heading[node].item()),
        "support_digest": support.support_digest,
        "selection_rule": "minimum finite route_cost; canonical pair-index tie break"}
    selected["descriptor_sha256"] = _hash(selected)
    return pair, selected


def _replay_plan_a_expert_goal_source_row(record, base_manifest):
    """Rebuild source support, rerun selection, and verify explicit execution."""
    base = record.get("base_record")
    if (not isinstance(base, dict) or
            record.get("snapshot_id") != base.get("branch_state", {}).get("snapshot_id") or
            record.get("state_sha256") != base.get("predecision_simulator_state_sha256") or
            record.get("macro_ticks") != base.get("macro_ticks")):
        raise RuntimeError("expert-goal source state identity differs from its v3 base row")
    replay = replay_counterfactual_branch(record["base_record"], base_manifest,
        require_hip_plan_a=True)
    env = replay["env"]
    try:
        support = replay["support"]
        if support.support_digest != record.get("support_digest"):
            raise RuntimeError("expert-goal source support digest changed on replay")
        pair, selected = _select_plan_a_expert_goal_v1(support)
        if (selected != record.get("selected_goal") or
                selected["descriptor_sha256"] !=
                    record.get("teacher_selected_goal_descriptor_sha256")):
            raise RuntimeError("expert-goal selector result changed on replay")
        (reward, ended, timeout, trace, _, _, preempted,
         selected_target_lost, action_remasked,
         selected_target_action_remasked) = _reward_step(env,
            selected["action"], record["macro_ticks"], goal_xy=selected["goal_xy_m"],
            goal_heading=selected["goal_heading_rad"],
            allow_action7_traversal=(selected["action"] == 7),
            planner_force_goal_replan=True, return_execution_flags=True)
        if (preempted or selected_target_lost or action_remasked or
                selected_target_action_remasked):
            raise RuntimeError("explicit expert-goal execution was preempted, lost, or remasked")
        if (trace != record.get("executed_action_trace") or
                record.get("executed_action") != selected["action"] or
                reward != record.get("gross_macro_return") or
                bool(ended or timeout) != record.get("terminal") or
                record.get("execution_failure") is not False or
                record.get("selected_target_lost") is not False or
                record.get("action_remasked") is not False or
                record.get("selected_target_action_remasked") is not False or
                record.get("native_goal_preempted") is not False):
            raise RuntimeError("expert-goal execution trace/outcome changed on replay")
        return {"teacher_selected_goal_descriptor_sha256": selected["descriptor_sha256"],
            "selected_goal": selected, "selected_pair_index": pair,
            "teacher_selected_action": selected["action"],
            "teacher_selected_behavior": selected["behavior_id"],
            "support_digest": support.support_digest,
            "snapshot_id": record["snapshot_id"],
            "simulator_state_sha256": record["state_sha256"]}
    finally:
        close = getattr(env, "close", None)
        if callable(close): close()


def collect_plan_a_expert_goal_source_v1(output: Path, *, dataset: Path,
        qualification_protocol: Path, device: str):
    """Collect separately versioned, explicit rule-selected Plan A goal traces."""
    from .curriculum_training import source_identity
    from .meco_supervised import _read_counterfactual_dataset
    from .meco_device import require_indexed_cuda_device
    sources = source_identity()
    rows, source_sha, source_manifest_sha, _, protocol_id, protocol_sha, base_manifest = \
        _read_counterfactual_dataset(Path(dataset),
            qualification_protocol=Path(qualification_protocol), expected_sources=sources)
    device = str(require_indexed_cuda_device(device,
        purpose="Plan A explicit-goal source collection"))
    if base_manifest["device"]["device"] != device or base_manifest.get("split") != "train":
        raise ValueError("expert-goal source device/train split must match frozen v3 source")
    output_rows = []
    for base in rows:
        replay = replay_counterfactual_branch(base, base_manifest,
            require_hip_plan_a=True)
        env = replay["env"]
        try:
            pair, selected = _select_plan_a_expert_goal_v1(replay["support"])
            (reward, ended, timeout, trace, _, _, preempted,
             selected_target_lost, action_remasked,
             selected_target_action_remasked) = _reward_step(env,
                selected["action"], base["macro_ticks"], goal_xy=selected["goal_xy_m"],
                goal_heading=selected["goal_heading_rad"],
                allow_action7_traversal=(selected["action"] == 7),
                planner_force_goal_replan=True, return_execution_flags=True)
            if (preempted or selected_target_lost or action_remasked or
                    selected_target_action_remasked):
                raise RuntimeError("selected explicit expert goal was preempted, lost, or remasked")
            output_rows.append({"schema": PLAN_A_EXPERT_GOAL_SOURCE_SCHEMA,
                "target_schema": PLAN_A_EXPERT_GOAL_SOURCE_TARGET_SCHEMA,
                "scenario_id": base["scenario_id"], "category": base["category"],
                "seed": base["seed"], "decision_index": base["decision_index"],
                "snapshot_id": base["branch_state"]["snapshot_id"],
                "state_sha256": base["predecision_simulator_state_sha256"],
                "support_digest": replay["support"].support_digest,
                "macro_ticks": base["macro_ticks"], "selected_goal": selected,
                "teacher_selected_goal_descriptor_sha256": selected["descriptor_sha256"],
                "selected_pair_index": pair, "executed_action": selected["action"],
                "executed_action_trace": trace, "gross_macro_return": reward,
                "execution_failure": bool(preempted or selected_target_lost or
                    action_remasked or selected_target_action_remasked),
                "native_goal_preempted": preempted,
                "selected_target_lost": selected_target_lost,
                "action_remasked": action_remasked,
                "selected_target_action_remasked": selected_target_action_remasked,
                "terminal": bool(ended or timeout), "u_status": "unavailable",
                "u_target": None,
                "u_unavailable_reason": "no independent expert goal-value oracle",
                "base_record": base})
        finally:
            close = getattr(env, "close", None)
            if callable(close): close()
    payload = "".join(json.dumps(row, sort_keys=True, separators=(",", ":"),
        allow_nan=False) + "\n" for row in output_rows).encode()
    output = Path(output).resolve(); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(payload)
    manifest = {"schema": PLAN_A_EXPERT_GOAL_SOURCE_SCHEMA,
        "target_schema": PLAN_A_EXPERT_GOAL_SOURCE_TARGET_SCHEMA,
        "base_dataset_schema": DATASET_SCHEMA, "base_dataset_sha256": source_sha,
        "base_manifest_sha256": source_manifest_sha, "protocol_id": protocol_id,
        "base_dataset": os.path.relpath(
            Path(dataset).resolve(), Path(output).resolve().parent),
        "qualification_protocol_sha256": protocol_sha, "sources": sources,
        "source_identity_sha256": _hash(sources), "device": base_manifest["device"],
        "split": "train", "support_order": SUPPORT_ORDER,
        "selection_rule": "minimum finite route_cost; canonical pair-index tie break",
        "selection_semantics": "demonstrated feasible B preference; no U optimality claim; U unavailable",
        "execution_semantics": "explicit selected-goal override; preemption and action remask rejected",
        "records_have_replay_bound_selected_goal": True,
        "record_count": len(output_rows), "dataset_sha256": hashlib.sha256(payload).hexdigest()}
    output.with_suffix(output.suffix + ".manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    return {"path": str(output), "record_count": len(output_rows),
        "dataset_sha256": manifest["dataset_sha256"],
        "schema": PLAN_A_EXPERT_GOAL_SOURCE_SCHEMA}


def _execute_plan_a_p_route_branch(record, manifest, descriptor, intervention_seed,
                                   *, p_cost_scale=1.0):
    """Strictly replay one branch with a deterministic P intervention map."""
    if not math.isfinite(float(p_cost_scale)) or p_cost_scale <= 0.:
        raise ValueError("P route intervention scale must be positive and finite")
    replay = replay_counterfactual_branch(record, manifest,
        require_hip_plan_a=True)
    env = replay["env"]
    try:
        expected = next((item for item in replay["snapshot"]["plan_a_descriptors"]
                         if item["descriptor_sha256"] ==
                            descriptor.get("descriptor_sha256")), None)
        if expected != descriptor:
            raise RuntimeError("P route descriptor differs from exact replay support")
        intervention_map = (None if intervention_seed is None else
            _plan_a_p_intervention_map(env, intervention_seed))
        ticks = record["macro_ticks"]
        action = int(descriptor["action"])
        goal = tuple(float(value) for value in descriptor["goal_xy_m"])
        route_trace = []
        reward, ended, timeout, trace, _, _, _ = _reward_step(env, action, ticks,
            goal_xy=goal, goal_heading=descriptor["goal_heading_rad"],
            allow_action7_traversal=(action == 7),
            planner_traversal_cost_override=intervention_map,
            planner_traversal_cost_scale=p_cost_scale,
            planner_force_goal_replan=True,
            planner_route_trace=route_trace)
        if any(executed != action for executed in trace):
            raise RuntimeError("P route branch action was remasked or preempted")
        route = _plan_a_p_route_record(env, intervention_map=intervention_map,
            intervention_seed=intervention_seed, selected_action=action,
            selected_goal=goal,
            selected_goal_descriptor_sha256=descriptor["descriptor_sha256"],
            scale=p_cost_scale, route_trace=route_trace)
        total_return, discounted = reward, reward
        elapsed = record["macro_start_tick"] + len(trace)
        continuation_decisions = 0
        while not (ended or timeout) and elapsed < manifest["horizon_physics_ticks"]:
            next_action = int(teacher_actions(env, env._last_obs)[0, 3].item())
            macro_ticks = min(manifest["max_macro_ticks"],
                manifest["horizon_physics_ticks"] - elapsed)
            step_reward, ended, timeout, continuation_trace, _, _, _ = _reward_step(
                env, next_action, macro_ticks)
            if (len(continuation_trace) != macro_ticks and not (ended or timeout)):
                raise RuntimeError("P route continuation macro ended early")
            total_return += step_reward
            continuation_decisions += 1
            discounted += manifest["gamma_per_macro"] ** continuation_decisions * step_reward
            elapsed += len(continuation_trace)
        return {"gross_macro_return": reward,
            "gross_horizon_return": total_return,
            "discounted_horizon_return": discounted,
            "continuation_policy": CONTINUATION_POLICY,
            "continuation_macro_horizon": continuation_decisions,
            "discount_semantics": manifest["discount_semantics"],
            "terminal": bool(ended or timeout), "native_route": route,
            "executed_action_trace": trace}
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()


def collect_plan_a_p_routes_v1(output: Path, *, dataset: Path,
        qualification_protocol: Path, p_cost_scale=1.0,
        intervention_maps=3, device=None):
    """Collect P0 and model-independent weighted-route simulator outcomes.

    `dataset` must be an already collected strict counterfactual v2 training
    dataset. This separate schema deliberately leaves the v2 P-null contract
    unchanged.
    """
    from .curriculum_training import source_identity
    from .meco_supervised import _read_counterfactual_dataset
    dataset_path = Path(dataset).resolve()
    base_manifest_path = dataset_path.with_suffix(dataset_path.suffix + ".manifest.json")
    base_manifest_hint = json.loads(base_manifest_path.read_text())
    from .meco_device import require_indexed_cuda_device
    base_device = str(require_indexed_cuda_device(device,
        purpose="Plan A P route collection"))
    if base_device != base_manifest_hint.get("device", {}).get("device"):
        raise ValueError("Plan A P route collection device must match the source dataset device")
    from .fuel_physics_hip import prepare_native_fuel_extension
    prepare_native_fuel_extension(base_device)
    sources = source_identity()
    base_rows, base_sha, base_manifest_sha, _, protocol_id, protocol_sha, base_manifest = (
        _read_counterfactual_dataset(dataset_path,
            qualification_protocol=Path(qualification_protocol),
            expected_sources=sources))
    if (not math.isfinite(float(p_cost_scale)) or p_cost_scale <= 0. or
            type(intervention_maps) is not int or not 1 <= intervention_maps <= 8):
        raise ValueError("invalid P route scale or intervention-map count")
    manifest = {**base_manifest,
        "schema": PLAN_A_P_ROUTE_DATASET_SCHEMA,
        "target_schema": PLAN_A_P_ROUTE_TARGET_SCHEMA,
        "source_counterfactual_dataset_sha256": base_sha,
        "source_counterfactual_manifest_sha256": base_manifest_sha,
        "source_counterfactual_schema": DATASET_SCHEMA,
        "p_intervention_generator": "seeded_smooth_positive_xy_maps_v1",
        "p_intervention_maps_per_descriptor": intervention_maps,
        "p_intervention_scale": float(p_cost_scale),
        "plan_a_p_oracle_target": "exact_simulator_gross_horizon_return_for_executed_native_adstar_route",
        "plan_a_p_supervision_semantics": "within_prefix_route_pairwise_preference; no per-cell oracle target",
        "plan_a_p_target": "within-prefix executed-route preference from simulator gross horizon return",
        "plan_a_p_prior": 0.0,
        "plan_a_p_unlabelled_reason": None,
        "records_have_exact_weighted_route_replay": True}
    result_rows = []
    for source_row in base_rows:
        route_groups = []
        for descriptor_index, descriptor in enumerate(
                source_row["plan_a_sampled_node_behavior_labels"]):
            descriptor = descriptor["descriptor"]
            descriptor_seed_base = int.from_bytes(hashlib.sha256(
                f"{source_row['seed']}:{source_row['decision_index']}:{descriptor_index}:p-route-v1".encode()
                ).digest()[:8], "little") & ((1 << 63) - 1)
            variants = [("P0", None)] + [(f"probe-{index}",
                (descriptor_seed_base + index) & ((1 << 63) - 1))
                for index in range(1, intervention_maps + 1)]
            outcomes = []
            for intervention_id, seed in variants:
                outcome = _execute_plan_a_p_route_branch(source_row,
                    base_manifest, descriptor, seed, p_cost_scale=p_cost_scale)
                _verify_native_fuel_extension(manifest.get("native_extension_identity"))
                outcomes.append({"intervention_id": intervention_id,
                    "intervention_seed": seed, **outcome})
            route_groups.append({"descriptor": descriptor, "outcomes": outcomes})
        core = {key: value for key, value in source_row.items()
                if key not in ("plan_c_counterfactuals", "plan_c_pairwise_labels",
                    "plan_a_sampled_node_behavior_labels", "plan_a_pairwise_preferences")}
        result_rows.append({**core,
            "schema": PLAN_A_P_ROUTE_DATASET_SCHEMA,
            "target_schema": PLAN_A_P_ROUTE_TARGET_SCHEMA,
            "plan_a_p_route_groups": route_groups})
    _verify_native_fuel_extension(manifest.get("native_extension_identity"))
    if source_identity() != sources:
        raise RuntimeError("source/runtime changed during Plan A P route collection")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n"
                      for row in result_rows).encode()
    output.write_bytes(payload)
    manifest["record_count"] = len(result_rows)
    manifest["dataset_sha256"] = hashlib.sha256(payload).hexdigest()
    manifest["route_group_count"] = sum(len(row["plan_a_p_route_groups"])
                                        for row in result_rows)
    output.with_suffix(output.suffix + ".manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    return {"path": str(output), "record_count": len(result_rows),
        "route_group_count": manifest["route_group_count"],
        "dataset_sha256": manifest["dataset_sha256"],
            "schema": PLAN_A_P_ROUTE_DATASET_SCHEMA}


def _plan_a_p_oracle_integral(cost_grid, points, resolution_m):
    """Independent CPU reference for grid-cell traversal cost in edge units."""
    nx, ny = len(cost_grid), len(cost_grid[0])
    total = 0.0
    for first, second in zip(points, points[1:]):
        x0, y0 = map(float, first)
        x1, y1 = map(float, second)
        dx, dy = x1 - x0, y1 - y0
        length = math.hypot(dx, dy)
        if length == 0.:
            continue
        cuts = [0., 1.]
        if dx:
            low, high = sorted((x0, x1))
            cuts.extend((cell * resolution_m - x0) / dx
                for cell in range(math.floor(low / resolution_m) + 1,
                                  math.ceil(high / resolution_m))
                if 0. < (cell * resolution_m - x0) / dx < 1.)
        if dy:
            low, high = sorted((y0, y1))
            cuts.extend((cell * resolution_m - y0) / dy
                for cell in range(math.floor(low / resolution_m) + 1,
                                  math.ceil(high / resolution_m))
                if 0. < (cell * resolution_m - y0) / dy < 1.)
        cuts.sort()
        for start, end in zip(cuts, cuts[1:]):
            middle = (start + end) * .5
            ix = min(nx - 1, max(0, math.floor((x0 + dx * middle) / resolution_m)))
            iy = min(ny - 1, max(0, math.floor((y0 + dy * middle) / resolution_m)))
            total += float(cost_grid[ix][iy]) * length * (end - start) / resolution_m
    return total


def _plan_a_p_oracle_grid(env, intervention_map):
    base = env.planner._bump_cost.detach().cpu().to(torch.float64).tolist()
    traversability = [[1.0 for _ in column] for column in base]
    if intervention_map is None:
        effective = [[float(value) for value in column] for column in base]
    else:
        p_map = intervention_map.detach().cpu().to(torch.float64).tolist()
        scale = float(getattr(env, "_meco_oracle_p_scale", 1.0))
        effective = [[float(base[ix][iy]) + scale * float(p_map[ix][iy])
                      for iy in range(len(base[ix]))] for ix in range(len(base))]
    return traversability, base, effective


def collect_plan_a_p_routes_v3(output: Path, *, dataset: Path,
        qualification_protocol: Path, p_cost_scale=1.0,
        intervention_maps=3, device=None):
    """Collect executed P0/probe routes with independent frozen-grid cost labels.

    The v2 reward-ranked collector remains the source of the matched replay
    branches. This additive schema records a pure-Python route-cost oracle.
    """
    temporary = Path(output).with_suffix(Path(output).suffix + ".routes-v2")
    collect_plan_a_p_routes_v1(temporary, dataset=dataset,
        qualification_protocol=qualification_protocol, p_cost_scale=p_cost_scale,
        intervention_maps=intervention_maps, device=device)
    manifest_path = temporary.with_suffix(temporary.suffix + ".manifest.json")
    manifest = json.loads(manifest_path.read_text())
    rows = [json.loads(line) for line in temporary.read_text().splitlines() if line]
    from .curriculum_training import source_identity
    manifest.update(schema=PLAN_A_P_ROUTE_ORACLE_DATASET_SCHEMA,
        target_schema=PLAN_A_P_ROUTE_ORACLE_TARGET_SCHEMA,
        p_oracle_version=PLAN_A_P_ROUTE_ORACLE_VERSION,
        p_oracle_implementation="frc_defense.meco_counterfactual._plan_a_p_oracle_integral",
        p_oracle_inputs="archived planner bump grid plus exact regenerated effective P map; xy resolution; executed route trace",
        plan_a_p_supervision_semantics="non_tied sampled executed-path cost preference; lower independent oracle cost is preferred",
        plan_a_p_oracle_target="mean per-snapshot integral of base bump cost plus scaled effective P map along executed path in normalized grid-edge units",
        plan_a_p_target="mean per-snapshot integral of base bump cost plus scaled effective P map along executed path in normalized grid-edge units",
        plan_a_p_unlabelled_reason=None)
    for row in rows:
        for group in row["plan_a_p_route_groups"]:
            for outcome in group["outcomes"]:
                route = outcome["native_route"]
                # The original branch data contains the exact intervention seed
                # and scale; regenerate the map only after replaying that branch.
                replay = replay_counterfactual_branch(row, manifest,
                    require_hip_plan_a=True)
                try:
                    env = replay["env"]
                    seed = outcome["intervention_seed"]
                    p_map = None if seed is None else _plan_a_p_intervention_map(env, seed)
                    scale = float(manifest["p_intervention_scale"])
                    env._meco_oracle_p_scale = scale
                    traversability_grid, base_grid, effective_grid = _plan_a_p_oracle_grid(env, p_map)
                    resolution = float(env.planner.resolution)
                    per_snapshot = [_plan_a_p_oracle_integral(effective_grid,
                        snap["route_points_m"], resolution)
                        for snap in route["route_trace"]]
                finally:
                    close = getattr(replay["env"], "close", None)
                    if callable(close):
                        close()
                evidence = {"version": PLAN_A_P_ROUTE_ORACLE_VERSION,
                    "implementation": "frc_defense.meco_counterfactual._plan_a_p_oracle_integral",
                    "grid_shape": [len(base_grid), len(base_grid[0])],
                    "resolution_m": resolution,
                    "base_traversability_cost_grid": traversability_grid,
                    "base_bump_cost_grid": base_grid,
                    "effective_cost_grid": effective_grid,
                    "grid_inputs_sha256": _hash({"traversability": traversability_grid,
                        "base": base_grid,
                        "effective": effective_grid, "resolution_m": resolution}),
                    "route_snapshot_costs_grid_edges": per_snapshot,
                    "route_cost_grid_edges": sum(per_snapshot) / len(per_snapshot),
                    "route_trace_sha256": route["route_trace_sha256"],
                    "selected_goal_descriptor_sha256": route["selected_goal_descriptor_sha256"],
                    "intervention_map_sha256": route["map_sha256"],
                    "intervention_scale": scale if seed is not None else 0.0}
                route["independent_cost_oracle"] = evidence
    if source_identity() != manifest["sources"]:
        raise RuntimeError("source/runtime changed during Plan A P oracle collection")
    for row in rows:
        row["schema"] = PLAN_A_P_ROUTE_ORACLE_DATASET_SCHEMA
        row["target_schema"] = PLAN_A_P_ROUTE_ORACLE_TARGET_SCHEMA
    payload = "".join(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n"
                      for row in rows).encode()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(payload)
    manifest["dataset_sha256"] = hashlib.sha256(payload).hexdigest()
    manifest["record_count"] = len(rows)
    manifest["oracle_label_count"] = sum(len(group["outcomes"])
        for row in rows for group in row["plan_a_p_route_groups"])
    output.with_suffix(output.suffix + ".manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    temporary.unlink(missing_ok=True)
    manifest_path.unlink(missing_ok=True)
    return {"path": str(output), "record_count": len(rows),
        "schema": PLAN_A_P_ROUTE_ORACLE_DATASET_SCHEMA,
        "oracle_label_count": manifest["oracle_label_count"],
        "dataset_sha256": manifest["dataset_sha256"]}
