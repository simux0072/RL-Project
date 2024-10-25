import torch


class Actor(torch.nn.Module):
    def __init__(self, input_size, output_size) -> None:
        super().__init__()

        self.input_size: int = input_size
        self.output_size: int = output_size

        self.policy = torch.nn.Sequential(
            torch.nn.Conv2d(1, 8, kernel_size=3),
            torch.nn.ELU(),
            torch.nn.Conv2d(8, 8, kernel_size=3),
            torch.nn.Flatten(start_dim=1),
            torch.nn.ELU(),
            torch.nn.Linear(288, 128),
            torch.nn.ELU(),
            torch.nn.Linear(128, self.output_size),
        )

    def forward(self, obs_input: torch.Tensor, masks):
        policy_probs = self.policy(obs_input)
        # policy_probs[~masks] = float("-inf")
        return policy_probs

    def forward_eval(self, obs_input, masks):
        policy_probs = self.policy(obs_input)
        # policy_probs[~masks] = float("-inf")
        probs = torch.nn.functional.softmax(policy_probs, dim=-1)
        return probs.argmax(), probs


class Critic(torch.nn.Module):
    def __init__(self, input_size) -> None:
        super().__init__()

        self.input_size: int = input_size

        self.critic = torch.nn.Sequential(
            torch.nn.Conv2d(1, 8, kernel_size=3),
            torch.nn.ELU(),
            torch.nn.Conv2d(8, 8, kernel_size=3),
            torch.nn.Flatten(start_dim=1),
            torch.nn.ELU(),
            torch.nn.Linear(288, 128),
            torch.nn.ELU(),
            torch.nn.Linear(128, 1),
        )

    def forward(self, obs_input: torch.Tensor):
        value = self.critic(obs_input).flatten(start_dim=0)
        return value


class RND(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.random_state = torch.nn.Sequential(
            torch.nn.Conv2d(1, 8, kernel_size=3),
            torch.nn.ELU(),
            torch.nn.Conv2d(8, 8, kernel_size=3),
            torch.nn.Flatten(start_dim=1),
            torch.nn.ELU(),
            torch.nn.Linear(288, 128),
            torch.nn.ELU(),
            torch.nn.Linear(128, 1),
        )

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, obs_input: torch.Tensor):
        value = self.random_state(obs_input).flatten(start_dim=0)
        return value


class Adverserial_Net(torch.nn.Module):
    def __init__(self, input_size, output_size) -> None:
        super().__init__()

        self.input_size: int = input_size
        self.output_size: int = output_size

        self.policy = torch.nn.Sequential(
            torch.nn.Conv2d(1, 8, kernel_size=3),
            torch.nn.ELU(),
            torch.nn.Conv2d(8, 8, kernel_size=3),
            torch.nn.Flatten(start_dim=1),
            torch.nn.ELU(),
            torch.nn.Linear(288, 128),
            torch.nn.ELU(),
            torch.nn.Linear(128, self.output_size),
        )

    def forward(self, obs_input: torch.Tensor, masks):
        policy_probs = self.policy(obs_input)
        # policy_probs[~masks] = float("-inf")
        dist = torch.distributions.Categorical(logits=policy_probs)
        return dist
