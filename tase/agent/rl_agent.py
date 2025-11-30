import torch
import torch.nn as nn
from torch.distributions import Normal


class PPOAgent(nn.Module):
    """Simple PPO actor-critic.

    Inputs:
      - num_inputs: observation dimensionality
      - num_actions: action dimensionality

    Outputs (methods):
      - forward(x): returns (mu, std) for the Gaussian policy
      - get_value(x): scalar value estimate
      - get_action_and_value(x, action=None): sample/evaluate actions + logprob, entropy, value
    """

    def __init__(self, num_inputs: int, num_actions: int):
        super(PPOAgent, self).__init__()

        # Deterministic policy mean (mu). Final Tanh bounds outputs to [-1, 1].
        self.actor_mu = nn.Sequential(
            nn.Linear(num_inputs, 512),
            nn.Tanh(),
            nn.Linear(512, 512),
            nn.Tanh(),
            nn.Linear(512, 512),
            nn.Tanh(),
            nn.Linear(512, num_actions),
            nn.Tanh()
        )

        # Learnable log-std per action (diagonal covariance).
        self.actor_logstd = nn.Parameter(torch.ones(1, num_actions) * -0.5)

        # State-value function (critic).
        self.critic = nn.Sequential(
            nn.Linear(num_inputs, 512),
            nn.Tanh(),
            nn.Linear(512, 512),
            nn.Tanh(),
            nn.Linear(512, 512),
            nn.Tanh(),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        mu = self.actor_mu(x)
        std = torch.exp(self.actor_logstd).expand_as(mu)
        return mu, std

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        mu, std = self.forward(x)
        dist = Normal(mu, std)
        # If no action provided, sample using reparameterization (for backprop).
        if action is None:
            action = dist.rsample()
        # Aggregate per-dimension metrics to scalars across action dims.
        log_prob = dist.log_prob(action).sum(-1)
        entropy = dist.entropy().sum(-1)
        return action, log_prob, entropy, self.get_value(x)
