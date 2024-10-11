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

        self.actor = (
            network.Actor(input_size, actions).to(self.device).to(torch.float64)
        )
        self.critic = (
            network.Critic(input_size, actions).to(self.device).to(torch.float64)
        )
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)
        self.mini_batch_size = mini_batch_size
        self.epochs = epochs
        self.epsilon = epsilon
        self.MSELoss = torch.nn.MSELoss()

    def get_action(self, obs_state: torch.Tensor):
        dist = self.actor(obs_state)
        value = self.critic(obs_state)
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
        actor_loss_val = 0
        critic_loss_val = 0
        for i in range(self.epochs):
            l = 0
            for (
                state,
                action,
                old_log_probs,
                return_,
                advantage,
            ) in self.get_minibatch():
                dist = self.actor(state)
                value = self.critic(state)
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
                actor_loss = actor_loss - 0.1 * entropy
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                actor_loss.backward()
                critic_loss.backward()
                self.actor_optimizer.step()
                self.critic_optimizer.step()
                actor_loss_val += actor_loss.item()
                critic_loss_val += critic_loss.item()
                l += 1
        return actor_loss_val / self.epochs, critic_loss_val / self.epochs
