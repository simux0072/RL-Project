import torch


class PPO_Network(torch.nn.Module):
    def __init__(self, input_size, output_size) -> None:
        super().__init__()

        self.input_size: int = input_size
        self.output_size: int = output_size

        self.policy = torch.nn.Sequential(
            torch.nn.Linear(in_features=self.input_size, out_features=64),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=64, out_features=32),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=32, out_features=self.output_size),
        )

        self.critic = torch.nn.Sequential(
            torch.nn.Linear(in_features=self.input_size, out_features=64),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=64, out_features=32),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=32, out_features=1),
        )

    def forward(self, obs_input: torch.Tensor):
        obs_input = torch.flatten(obs_input, start_dim=1)
        policy_probs = self.policy(obs_input)
        dist = torch.distributions.Categorical(logits=policy_probs)
        value = self.critic(obs_input).flatten(start_dim=0)
        return value, dist
