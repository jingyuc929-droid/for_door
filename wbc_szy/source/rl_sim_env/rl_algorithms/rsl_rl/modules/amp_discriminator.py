import torch
import torch.nn as nn
from rl_algorithms.rsl_rl.utils.log_print import (
    print_placeholder_end,
    print_placeholder_start,
)
from torch import autograd


class AMPDiscriminator(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_layer_sizes,
        device,
    ):
        super().__init__()

        self.device = device
        self.input_dim = input_dim

        amp_layers = []
        curr_in_dim = input_dim
        for hidden_dim in hidden_layer_sizes:
            amp_layers.append(nn.Linear(curr_in_dim, hidden_dim))
            amp_layers.append(nn.LeakyReLU())
            curr_in_dim = hidden_dim
        self.trunk = nn.Sequential(*amp_layers).to(device)
        self.amp_linear = nn.Linear(hidden_layer_sizes[-1], 1).to(device)

        self.trunk.train()
        self.amp_linear.train()

        print_placeholder_start("AMP Discriminator")
        print(f"AMP Discriminator: {self.trunk}")
        print(f"AMP Linear: {self.amp_linear}")
        print_placeholder_end()

    def forward(self, x):
        h = self.trunk(x)
        d = self.amp_linear(h)
        return d

    def compute_gradient_penalty(self, expert_state, expert_next_state, lambda_=10):
        expert_data = torch.cat([expert_state, expert_next_state], dim=-1)
        expert_data = expert_data.clone().requires_grad_()

        disc = self.amp_linear(self.trunk(expert_data))
        ones = torch.ones(disc.size(), device=disc.device)
        grad = autograd.grad(
            outputs=disc,
            inputs=expert_data,
            grad_outputs=ones,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        # Enforce that the grad norm approaches 0.
        gradient_penalty = lambda_ * (grad.norm(2, dim=1) - 0).pow(2).mean()
        return gradient_penalty

    def discriminator_out(self, state, next_state, normalizer=None):
        with torch.no_grad():
            self.eval()
            if normalizer is not None:
                state = normalizer.normalize_torch(state, self.device)
                next_state = normalizer.normalize_torch(next_state, self.device)

            d = self.amp_linear(self.trunk(torch.cat([state, next_state], dim=-1)))
            self.train()
        return d

    def load_state_dict(self, state_dict, strict=True):
        """Load the parameters of the actor-critic model.

        Args:
            state_dict (dict): State dictionary of the model.
            strict (bool): Whether to strictly enforce that the keys in state_dict match the keys returned by this
                           module's state_dict() function.

        Returns:
            bool: Whether this training resumes a previous training. This flag is used by the `load()` function of
                  `OnPolicyRunner` to determine how to load further parameters (relevant for, e.g., distillation).
        """

        super().load_state_dict(state_dict, strict=strict)
        return True
