import torch


class Actor(torch.nn.Module):
    def __init__(self, input_size, output_size) -> None:
        super().__init__()

        self.input_size: int = input_size
        self.output_size: int = output_size

        self.policy = torch.nn.Sequential(
            torch.nn.BatchNorm2d(1),
            torch.nn.Conv2d(1, 3, kernel_size=3),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(3),
            torch.nn.Conv2d(3, 6, kernel_size=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(6, 6, kernel_size=1),
            torch.nn.Flatten(start_dim=1),
            torch.nn.ReLU(),
            torch.nn.Linear(294, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, self.output_size),
        )

    def forward(self, obs_input: torch.Tensor):
        policy_probs = self.policy(obs_input)
        dist = torch.distributions.Categorical(logits=policy_probs)
        return dist


class Critic(torch.nn.Module):
    def __init__(self, input_size, output_size) -> None:
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size

        self.critic = torch.nn.Sequential(
            torch.nn.BatchNorm2d(1),
            torch.nn.Conv2d(1, 3, kernel_size=3),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(3),
            torch.nn.Conv2d(3, 6, kernel_size=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(6, 6, kernel_size=1),
            torch.nn.Flatten(start_dim=1),
            torch.nn.ReLU(),
            torch.nn.Linear(294, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1),
        )

    def forward(self, obs_input: torch.Tensor):
        value = self.critic(obs_input).flatten(start_dim=0)
        return value
