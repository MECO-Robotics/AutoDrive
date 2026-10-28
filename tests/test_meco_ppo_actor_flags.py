import inspect
import torch
from frc_defense.meco_track_training import _ppo_update


def test_actor_row_host_flags_match_tensor_reduction():
    for failures in ([], [False], [True], [True, False, True]):
        rows = [{'execution_failure': value} for value in failures]
        flags = [not row.get('execution_failure', False) for row in rows]
        assert any(flags) == bool(torch.tensor(flags, dtype=torch.bool).any())


def test_ppo_update_avoids_device_reduction_readback_for_actor_branch():
    source = inspect.getsource(_ppo_update)
    assert 'if any(actor_row_flags):' in source
    assert 'bool(actor_rows.any())' not in source
