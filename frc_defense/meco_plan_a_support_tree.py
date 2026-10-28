"""Versioned Plan A support evidence without altering pinned route sources."""
import torch

from .meco_plan_a_policy import _plan_a_support_tensors
from .meco_tree_digest import tree_digest_cpu, tree_digest_hip_bytes

DOMAIN = b"meco-plan-a-support-tree-sha256-v2\0"
SCHEME = "tree-sha256-v2"


def _support_digest(support, graph, *, require_hip=True, return_device=False, legacy=False):
    """Bind the identical v1 metadata and ordered tensor bytes via a v2 tree."""
    device = graph.xy.device
    on_hip = device.type == "cuda" and torch.version.hip is not None
    if require_hip and not on_hip:
        raise RuntimeError("Plan A support tree digest requires a HIP device")
    if device.type != "cpu" and not on_hip:
        raise RuntimeError("Plan A support tree digest requires a HIP device")
    if not torch.is_tensor(support.root_node) or not torch.is_tensor(support.root_connector_m):
        raise RuntimeError("Plan A support tree requires tensor root provenance")
    if not torch.is_tensor(support.fallback_pair_tensor):
        raise RuntimeError("Plan A support tree requires tensor fallback provenance")
    tensors = {"root_node": support.root_node.reshape(()),
               "root_connector_m": support.root_connector_m.reshape(()),
               "fallback_pair_tensor": support.fallback_pair_tensor,
               **_plan_a_support_tensors(support, graph)}
    header = ["plan-a-support-sha256-v1", support.snapshot_id,
              support.order_digest, f"legacy_fallback={support.fallback_pair}"]
    values = []
    for name, tensor in tensors.items():
        if tensor is None:
            header.append(f"{name}=none")
            continue
        if tensor.device != device:
            raise RuntimeError(f"Plan A support tensor {name} is on a different device")
        header.append(f"{name}:{tensor.dtype}:{tuple(tensor.shape)}")
        values.append(tensor.detach().contiguous().reshape(-1).view(torch.uint8))
    header_bytes = ("\n".join(header) + "\n").encode()
    if on_hip:
        prefix = torch.frombuffer(bytearray(header_bytes), dtype=torch.uint8).to(device)
        payload = torch.cat([prefix, *values])
        if legacy:
            from .meco_plan_a_shortest_path_hip import extension
            digest = extension().sha256_bytes(payload)
        else:
            digest = tree_digest_hip_bytes(payload, domain=DOMAIN)
        return digest if return_device else bytes(digest.cpu().tolist()).hex()
    if return_device:
        raise RuntimeError("Plan A support device digest requires HIP")
    payload = header_bytes + b"".join(tensor.numpy().tobytes() for tensor in values)
    return tree_digest_cpu(payload, domain=DOMAIN)


def support_tree_digest(support, graph, *, require_hip=True):
    return _support_digest(support, graph, require_hip=require_hip)


def support_tree_digest_device(support, graph):
    """Return the v2 digest as device uint8[32] for batched readback."""
    return _support_digest(support, graph, return_device=True)


def support_digest_v1_compatible(support, graph):
    """Preserve the v1 byte stream while safely flattening scalar tensors."""
    return _support_digest(support, graph, legacy=True)
