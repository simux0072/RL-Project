import network
import torch
import numpy


class Agent:
    def __init__(
        self,
        memory,
        device: torch.device,
        input_size: int,
        actions: int,
        lr: float,
        epochs: int,
        mini_batch_size: int,
        epsilon: float,
    ) -> None:
        self.memory = memory
        self.device: torch.device = device

        self.network = (
            network.PPO_Network(input_size, actions).to(self.device).to(torch.float64)
        )
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)

        self.mini_batch_size = mini_batch_size
        self.epochs = epochs
        self.epsilon = epsilon
        self.MSELoss = torch.nn.MSELoss()

    def get_action(self, obs_state: torch.Tensor):
        value, dist = self.network(obs_state)
        action = dist.sample()
        return value, action.item(), dist.log_prob(action)

    def get_minibatch(self):
        batch_size = len(self.memory.states)
        states, actions, old_log_probs, advantages, returns = self.memory.get_memory(
            self.device
        )
        if batch_size // self.mini_batch_size == 0:
            rand_idx = numpy.random.randint(0, batch_size, batch_size)
            yield states[rand_idx], actions[rand_idx], old_log_probs[rand_idx], returns[
                rand_idx
            ], advantages[rand_idx]
        else:
            for _ in range(batch_size // self.mini_batch_size):
                rand_idx = numpy.random.randint(0, batch_size, self.mini_batch_size)
                yield states[rand_idx], actions[rand_idx], old_log_probs[
                    rand_idx
                ], returns[rand_idx], advantages[rand_idx]

    def train(self):
        loss_val = 0
        for i in range(self.epochs):
            l = 0
            for (
                state,
                action,
                old_log_probs,
                return_,
                advantage,
            ) in self.get_minibatch():
                value, dist = self.network(state)
                new_log_probs = dist.log_prob(action)
                entropy = dist.entropy().mean()
                ratio = (new_log_probs - old_log_probs).exp()
                surr_1 = ratio * advantage
                surr_2 = (
                    torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantage
                )
                actor_loss = -torch.min(surr_1, surr_2).mean()
                critic_loss = self.MSELoss(return_, value)
                # if critic_loss >= 10 or actor_loss >= 10:
                #    print(f"returns: {return_.shape}")
                #    print(f"advantages: {advantage}")
                #    print(f"actor loss: {actor_loss}")
                #    print(f"critic_loss: {critic_loss}")
                #    print(f"entropy: {entropy}")
                loss = 0.5 * critic_loss + actor_loss - 0.1 * entropy
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                loss_val += loss.item()
                l += 1
        return loss_val / self.epochs
