"""Experimental fixed-shape device row selection for sequential PPO.

The caller supplies only the tensors and shared metadata needed for evaluation.
Per-row provenance must be validated separately before constructing the bank.
"""
from dataclasses import fields, is_dataclass

import torch


class _TensorColumn:
    def __init__(self, values):
        first = values[0]
        if any(v.shape != first.shape or v.dtype != first.dtype or
               v.device != first.device for v in values):
            raise ValueError("row tensor shape, dtype, or device differs")
        self.values = torch.stack(values)

    def select(self, index):
        return self.values.index_select(0, index).squeeze(0)


class _Node:
    def __init__(self, values, device):
        first = values[0]
        if isinstance(first, torch.Tensor):
            if any(not isinstance(v, torch.Tensor) for v in values):
                raise ValueError("row types differ")
            if first.device != device:
                raise ValueError("row tensor is on the wrong device")
            self.kind, self.payload = "tensor", _TensorColumn(values)
        elif is_dataclass(first) and not isinstance(first, type):
            if any(type(v) is not type(first) for v in values):
                raise ValueError("row dataclass types differ")
            self.kind = "dataclass"
            self.payload = (type(first), {f.name: _Node([getattr(v, f.name) for v in values], device)
                                          for f in fields(first) if f.init})
        elif isinstance(first, dict):
            if any(not isinstance(v, dict) or v.keys() != first.keys() for v in values):
                raise ValueError("row mapping keys differ")
            self.kind, self.payload = "dict", {
                k: _Node([v[k] for v in values], device) for k in first}
        elif type(first) in (tuple, list):
            if any(type(v) is not type(first) or len(v) != len(first) for v in values):
                raise ValueError("row sequence shapes differ")
            self.kind, self.payload = "sequence", (type(first), [
                _Node([v[i] for v in values], device) for i in range(len(first))])
        elif type(first) in (str, int, float, bool, type(None)):
            if any(type(v) is not type(first) or v != first for v in values):
                raise ValueError("row static metadata differs")
            self.kind, self.payload = "static", first
        else:
            raise TypeError(f"unsupported row value: {type(first).__name__}")

    def select(self, index):
        if self.kind == "tensor":
            return self.payload.select(index)
        if self.kind == "static":
            return self.payload
        if self.kind == "dict":
            return {k: node.select(index) for k, node in self.payload.items()}
        if self.kind == "sequence":
            typ, nodes = self.payload
            return typ(node.select(index) for node in nodes)
        typ, nodes = self.payload
        return typ(**{k: node.select(index) for k, node in nodes.items()})


class DeviceRowBank:
    """Select homogeneous rows using a device index without reading it on host.

    Invalid indices are clamped for memory safety and returned as a device flag.
    The caller must combine that flag with other validation and prevent publication
    of all optimizer and RNG state on failure. This class does not do so itself.
    """
    def __init__(self, rows, *, device):
        if not rows:
            raise ValueError("row bank cannot be empty")
        self.device = torch.device(device)
        self.count = len(rows)
        self.root = _Node(rows, self.device)

    def select(self, index):
        if not isinstance(index, torch.Tensor) or index.device != self.device or \
                index.dtype != torch.long or index.numel() != 1:
            raise ValueError("index must be one int64 value on the bank device")
        index = index.reshape(1)
        valid = (index >= 0) & (index < self.count)
        safe = index.clamp(0, self.count - 1)
        return self.root.select(safe), valid.reshape(())


def build_ppo_row_bank(rows, *, device):
    """Build after PPO has validated every original support and route digest.

    Provenance bytes remain attached to their selected data. The support strings
    are blank inside the evaluation-only object because they are not consumed by
    scoring; the original objects remain available to the validation pass.
    """
    from dataclasses import replace

    if any(row.get("support") is None or row.get("execution_failure", False)
           for row in rows):
        raise ValueError("row bank requires successful Plan A rows")
    planner = rows[0]["planner_object"]
    if any(row["planner_object"] is not planner for row in rows):
        raise ValueError("row bank requires a shared planner")
    keys = ("raster", "globals", "graph", "route_cost_nodes", "risk_cost_nodes",
            "action", "logprob", "value", "reward", "hidden_before")
    metadata = {
        "snapshot": [r["support"].snapshot_id for r in rows],
        "order": [r["support"].order_digest for r in rows],
        "support_digest": [r["support_digest"] for r in rows],
        "route_digest": [r["route_evidence_sha256"] for r in rows],
        "source_digest": [r["route_evidence_source_identity_sha256"] for r in rows],
        "support_scheme": [r["support_digest_scheme"] for r in rows],
        "route_scheme": [r["route_evidence_scheme"] for r in rows],
    }
    encoded = {name: [value.encode("utf-8") for value in values]
               for name, values in metadata.items() if name != "route_digest"}
    widths = {name: max(map(len, values)) for name, values in encoded.items()}
    prepared = []
    for row_index, row in enumerate(rows):
        selected = {key: row[key] for key in keys}
        support = row["support"]
        selected["support"] = replace(support, snapshot_id="", support_digest="", order_digest="")
        selected["graph_object"] = row["graph_object"]
        selected["provenance"] = {
            name: torch.tensor(list(values[row_index].ljust(widths[name], b"\0")),
                               device=device, dtype=torch.uint8)
            for name, values in encoded.items()}
        route_digest = metadata["route_digest"][row_index]
        if isinstance(route_digest, torch.Tensor):
            selected["provenance"]["route_digest"] = route_digest
        else:
            selected["provenance"]["route_digest"] = torch.tensor(
                list(route_digest.encode("utf-8")), device=device, dtype=torch.uint8)
        prepared.append(selected)
    return DeviceRowBank(prepared, device=device)
