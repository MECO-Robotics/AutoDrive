"""Architecture adapters for the curriculum's categorical PPO interface."""
from torch import nn

from .architecture_cnn import FullFieldCNN, FullFieldResNet
from .architecture_dense import ResidualMLP, FieldFNOPolicy
from .architecture_entities import DeepSetsPolicy, EntityAttentionPolicy
from .tensor_training_runtime import ActorCritic


ARCHITECTURES = {
    'baseline_mlp': None,
    'residual_mlp': ResidualMLP,
    'deep_sets': DeepSetsPolicy,
    'entity_attention': EntityAttentionPolicy,
    'full_field_cnn': FullFieldCNN,
    'full_field_resnet': FullFieldResNet,
    'field_fno': FieldFNOPolicy,
}


class PerceptionActorCritic(ActorCritic):
    """Reuse native masking, sampling and scoring with perception dictionaries."""

    def __init__(self, architecture):
        nn.Module.__init__(self)
        self.action_kind = 'categorical'
        self.architecture = architecture
        self.network = ARCHITECTURES[architecture]()

    def forward(self, inputs):
        logits, value = self.network(inputs)
        return logits, None, value


def make_policy(architecture='baseline_mlp'):
    if architecture not in ARCHITECTURES:
        raise ValueError(f'unknown architecture: {architecture}')
    if architecture == 'baseline_mlp':
        return ActorCritic(137, 8, action_kind='categorical')
    return PerceptionActorCritic(architecture)
