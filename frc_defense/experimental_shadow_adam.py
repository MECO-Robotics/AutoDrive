"""Experimental device-flagged Adam transaction.

All reads and writes must be ordered on one stream and no observer may access the
originals until commit finishes. Device errors during commit are not recoverable.
This does not validate PPO replay, ancestor metadata, or checkpoint publication.
"""
import copy

import torch


class ShadowAdam:
    """Run Adam speculatively and publish state only after validation."""

    def __init__(self, optimizer):
        if type(optimizer) is not torch.optim.Adam:
            raise TypeError("requires torch.optim.Adam")
        self.original = optimizer
        self.pairs = []
        groups = []
        devices = set()
        storage_addresses = set()
        parameter_ids = set()
        for group in optimizer.param_groups:
            if group.get("differentiable") or group.get("fused"):
                raise ValueError("differentiable and fused Adam are unsupported")
            shadows = []
            for original in group["params"]:
                if id(original) in parameter_ids:
                    raise ValueError("duplicate parameters are unsupported")
                parameter_ids.add(id(original))
                if not original.requires_grad:
                    raise ValueError("all parameters must require gradients")
                devices.add(original.device)
                state = optimizer.state.get(original)
                expected = {"step", "exp_avg", "exp_avg_sq"}
                if group.get("amsgrad"):
                    expected.add("max_exp_avg_sq")
                if state is None:
                    state = {}
                if state and set(state) != expected:
                    raise ValueError("Adam state is incomplete")
                if any(not isinstance(value, torch.Tensor) for value in state.values()):
                    raise ValueError("all Adam state must be tensors")
                if any(value.device != original.device for value in state.values()):
                    raise ValueError("Adam state and parameter must share a device")
                if original.device.type != "cpu" and not group.get("capturable"):
                    raise ValueError("GPU Adam must be capturable")
                for tensor in (original, *(state[k] for k in sorted(state))):
                    if tensor.numel():
                        address = (tensor.device, tensor.untyped_storage().data_ptr())
                        if address in storage_addresses:
                            raise ValueError("aliased optimizer tensors are unsupported")
                        storage_addresses.add(address)
                shadow = torch.nn.Parameter(original.detach().clone())
                if original.grad is not None:
                    if (original.grad.layout != torch.strided or
                            original.grad.shape != original.shape or
                            original.grad.device != original.device or
                            original.grad.dtype != original.dtype):
                        raise ValueError("unsupported original gradient")
                    shadow.grad = original.grad.detach().clone()
                shadows.append(shadow)
                self.pairs.append((original, shadow))
            copied = {k: copy.deepcopy(v) for k, v in group.items() if k != "params"}
            copied["params"] = shadows
            groups.append(copied)
        if len(devices) != 1 or not self.pairs:
            raise ValueError("requires parameters on exactly one device")
        self.device = next(iter(devices))
        self.shadow = torch.optim.Adam(groups)
        for original, shadow in self.pairs:
            self.shadow.state[shadow] = {
                k: v.detach().clone() for k, v in optimizer.state.get(original, {}).items()
            }
        self.valid = torch.ones((), dtype=torch.bool, device=self.device)
        self.finished = False

    def require(self, condition):
        if self.finished:
            raise RuntimeError("transaction is finished")
        if condition.device != self.device or condition.dtype != torch.bool or condition.shape != ():
            raise ValueError("condition must be a scalar boolean on the optimizer device")
        self.valid.logical_and_(condition.detach())

    def clone_modules(self, *modules):
        """Bind independent module copies to this transaction's parameters.

        The copies must be used for every forward in the transaction. Buffers
        are rejected because this transaction does not publish buffer changes.
        Optimizer and module parameter coverage must agree exactly.
        """
        if self.finished:
            raise RuntimeError("transaction is finished")
        if not modules or any(not isinstance(module, torch.nn.Module) for module in modules):
            raise TypeError("requires one or more modules")
        originals = {id(original): shadow for original, shadow in self.pairs}
        used = {}
        for module in modules:
            if any(True for _ in module.buffers()):
                raise ValueError("modules with buffers are unsupported")
            for parameter in module.parameters():
                if parameter.requires_grad:
                    if id(parameter) not in originals:
                        raise ValueError("trainable module parameter missing from optimizer")
                    used[id(parameter)] = parameter
        if set(used) != set(originals):
            raise ValueError("optimizer contains parameter missing from modules")
        memo = dict(originals)
        return tuple(copy.deepcopy(module, memo) for module in modules)

    def commit(self):
        return self._commit(allow_gradient_presence_change=False)

    def commit_after_validation(self):
        """Publish after a host validity check, including gradient presence.

        This method synchronizes and is only for a transaction that already
        requires a host decision. An invalid transaction leaves live state
        untouched. It does not make publication crash-atomic.
        """
        if self.finished:
            raise RuntimeError("transaction is finished")
        if not bool(self.valid):
            self.finished = True
            return False
        self._commit(allow_gradient_presence_change=True)
        return True

    def _commit(self, *, allow_gradient_presence_change):
        if self.finished:
            raise RuntimeError("transaction is finished")
        # Gradient presence is Python metadata. Changes are only allowed
        # after commit_after_validation has accepted the transaction.
        destinations = set()
        sources = set()
        writes = []
        gradient_changes = []
        new_states = []
        for original, shadow in self.pairs:
            if (original.grad is None) != (shadow.grad is None):
                if not allow_gradient_presence_change:
                    raise ValueError("gradient presence changed during transaction")
                if shadow.grad is not None:
                    if (shadow.grad.layout != torch.strided or
                            shadow.grad.shape != original.shape or
                            shadow.grad.device != original.device or
                            shadow.grad.dtype != original.dtype):
                        raise ValueError("transaction gradient metadata changed")
                    replacement = shadow.grad.detach().clone()
                else:
                    replacement = None
                gradient_changes.append((original, replacement))
            original_state = self.original.state.get(original)
            shadow_state = self.shadow.state.get(shadow)
            if original_state is None:
                original_state = {}
            if not isinstance(original_state, dict) or not isinstance(shadow_state, dict):
                raise ValueError("optimizer state changed during transaction")
            if not original_state and shadow_state and allow_gradient_presence_change:
                group = next(g for g in self.original.param_groups
                             if any(p is original for p in g["params"]))
                expected = {"step", "exp_avg", "exp_avg_sq"}
                if group.get("amsgrad"):
                    expected.add("max_exp_avg_sq")
                if (set(shadow_state) != expected or any(
                        not isinstance(v, torch.Tensor) or v.device != original.device
                        for v in shadow_state.values())):
                    raise ValueError("new Adam state is invalid")
                if (shadow_state["step"].numel() != 1 or any(
                        shadow_state[k].shape != original.shape or
                        shadow_state[k].dtype != original.dtype
                        for k in expected - {"step"})):
                    raise ValueError("new Adam state metadata is invalid")
                new_states.append((original, {k: v.detach().clone()
                                               for k, v in shadow_state.items()}))
                shadow_state = {}
            if original_state.keys() != shadow_state.keys():
                raise ValueError("optimizer state changed during transaction")
            pairs = [(original, shadow)]
            pairs.extend((original_state[key], shadow_state[key]) for key in original_state)
            if original.grad is not None and shadow.grad is not None:
                pairs.append((original.grad, shadow.grad))
            for destination, source in pairs:
                if (not isinstance(destination, torch.Tensor) or
                        not isinstance(source, torch.Tensor) or
                        destination.layout != torch.strided or
                        source.layout != torch.strided or
                        destination.shape != source.shape or
                        destination.device != source.device or
                        destination.dtype != source.dtype or
                        torch._debug_has_internal_overlap(destination) != 0):
                    raise ValueError("transaction tensor metadata changed")
                if destination.numel():
                    address = (destination.device, destination.untyped_storage().data_ptr())
                    if address in destinations:
                        raise ValueError("aliased transaction destinations are unsupported")
                    destinations.add(address)
                if source.numel():
                    sources.add((source.device, source.untyped_storage().data_ptr()))
                writes.append((destination, source))
        if sources & destinations:
            raise ValueError("transaction sources alias destinations")
        if (self.valid.device != self.device or self.valid.dtype != torch.bool or
                self.valid.shape != () or
                (self.valid.device, self.valid.untyped_storage().data_ptr()) in destinations):
            raise ValueError("transaction validity metadata changed")
        self.finished = True
        with torch.no_grad():
            for destination, source in writes:
                destination.copy_(torch.where(self.valid, source, destination))
            for original, state in new_states:
                self.original.state[original] = state
            for original, replacement in gradient_changes:
                original.grad = replacement
        return self.valid
