import torch
import gae


class Memory:
    def __init__(self, size: int) -> None:
        self.size = size
        self.restart()

    def restart(self):
        self.states = []
        self.actions = []
        self.probs = []
        self.rewards = []
        self.ends = []
        self.values = []

    def set_target_values(self, gamma, lambda_):
        self.returns, self.advantages = gae.GAE(
            self.ends,
            self.rewards,
            torch.cat(self.values),
            gamma,
            lambda_,
        )

    def update_memory(self, state, action, value, prob, end, reward):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(prob)
        self.ends.append(end)
        self.rewards.append(reward)
        self.values.append(value)

    def get_memory(self, device):
        return (
            torch.cat(self.states).to(device).detach(),
            torch.tensor(self.actions).to(device).detach(),
            torch.cat(self.probs).to(device).detach(),
            self.advantages.to(device).detach(),
            self.returns.to(device).detach(),
        )
