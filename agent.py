import network
import torch
import numpy


class Agent:
    def __init__(
        self,
        device: torch.device,
        input_size: int,
        actions: int,
        lr: float,
        adver_lr: float,
        epochs: int,
        mini_batch_size: int,
        epsilon: float,
        c: float,
        annealing_rate: float,
    ) -> None:
        self.device: torch.device = device
        self.mini_batch_size = mini_batch_size
        self.epochs = epochs
        self.epsilon = epsilon

        self.c = c
        self.annealing_rate = annealing_rate

        self.actor = (
            network.Actor(input_size, actions).to(self.device).to(torch.float64)
        )
        self.critic = network.Critic(input_size).to(self.device).to(torch.float64)
        self.rnd = network.RND().to(torch.float64).to(self.device)
        self.adverserial_net = (
            network.Adverserial_Net(input_size, actions)
            .to(self.device)
            .to(torch.float64)
        )

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)
        self.adverserial_optimizer = torch.optim.Adam(
            self.adverserial_net.parameters(), lr=adver_lr
        )

    @torch.no_grad()
    def action(self, obs_states: torch.Tensor, masks):
        logits = self.actor(obs_states, masks)
        value = self.critic(obs_states)
        dist = torch.distributions.Categorical(logits=logits)
        actions = dist.sample()
        return value, actions.tolist(), logits

    @torch.no_grad()
    def eval_action(self, obs_states, masks):
        actions, probabilities = self.actor.forward_eval(obs_states, masks)
        return actions.tolist(), probabilities

    @torch.no_grad()
    def get_intrinsic_reward(self, obs_states, values):
        values_int = self.rnd(obs_states)
        r_int = ((values_int - values) ** 2).cpu().numpy()
        # r_int = (r_int - r_int.mean()) / (r_int.std() + 1e-8)
        return r_int

    def anneal_constant(self):
        self.c += self.annealing_rate
        if self.c <= 0:
            self.c -= self.annealing_rate
            self.annealing_rate = 0

    def get_minibatch(self, states, actions, logits, advantages, returns, masks):
        batch_size = states.size(0)

        old_dist = torch.distributions.Categorical(logits=logits)
        old_log_probs = old_dist.log_prob(actions)

        if batch_size // self.mini_batch_size == 0:
            rand_idx = numpy.random.randint(0, batch_size, batch_size)
            yield states[rand_idx], actions[rand_idx], old_log_probs[rand_idx], logits[
                rand_idx
            ], returns[rand_idx], advantages[rand_idx], masks[rand_idx]
        else:
            for _ in range(batch_size // self.mini_batch_size):
                rand_idx = numpy.random.randint(0, batch_size, self.mini_batch_size)
                yield states[rand_idx], actions[rand_idx], old_log_probs[
                    rand_idx
                ], logits[rand_idx], returns[rand_idx], advantages[rand_idx], masks[
                    rand_idx
                ]

    def get_minibatch_agac(self, states, actions, logits, advantages, returns, masks):
        batch_size = states.size(0)

        old_dist = torch.distributions.Categorical(logits=logits)
        old_log_probs = old_dist.log_prob(actions)
        old_adver_dist = self.adverserial_net(states, masks)
        old_adver_log_probs = old_adver_dist.log_prob(actions)

        agac_advantages = (
            advantages + self.c * (old_log_probs - old_adver_log_probs)
        ).detach()
        agac_returns = (
            returns
            + self.c * torch.distributions.kl_divergence(old_dist, old_adver_dist)
        ).detach()

        self.kl_div = torch.distributions.kl_divergence(old_dist, old_adver_dist).mean()
        self.adv_diff = (agac_advantages - advantages).mean()
        self.returns_diff = (agac_returns - returns).mean()

        if batch_size // self.mini_batch_size == 0:
            rand_idx = numpy.random.randint(0, batch_size, batch_size)
            yield states[rand_idx], actions[rand_idx], old_log_probs[rand_idx], logits[
                rand_idx
            ], agac_returns[rand_idx], agac_advantages[rand_idx], masks[rand_idx]
        else:
            for _ in range(batch_size // self.mini_batch_size):
                rand_idx = numpy.random.randint(0, batch_size, self.mini_batch_size)
                yield states[rand_idx], actions[rand_idx], old_log_probs[
                    rand_idx
                ], logits[rand_idx], agac_returns[rand_idx], agac_advantages[
                    rand_idx
                ], masks[
                    rand_idx
                ]

    def train(self, states, actions, logits, advantages, returns, masks):
        actor_loss_val = 0
        critic_loss_val = 0
        entropy_loss_val = 0
        for _ in range(self.epochs):
            for (
                state,
                action,
                old_log_prob,
                _,
                return_,
                advantage,
                mask,
            ) in self.get_minibatch(
                states, actions, logits, advantages, returns, masks
            ):
                new_logits = self.actor(state, mask)
                values = self.critic(state)

                dist = torch.distributions.Categorical(logits=new_logits)
                new_log_probs = dist.log_prob(action)

                entropy = dist.entropy().mean()
                ratio = (new_log_probs - old_log_prob).exp()
                surr_1 = ratio * advantage
                surr_2 = (
                    torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantage
                )
                critic_loss = ((values - return_) ** 2).mean()
                actor_loss = -torch.min(surr_1, surr_2).mean()
                loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy

                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()

                loss.backward()

                self.actor_optimizer.step()
                self.critic_optimizer.step()

                actor_loss_val += actor_loss.item()
                critic_loss_val += critic_loss.item() * 0.5
                entropy_loss_val += entropy.item() * 0.01

        division_factor = self.epochs * self.mini_batch_size
        return (
            actor_loss_val / division_factor,
            critic_loss_val / division_factor,
            entropy_loss_val / division_factor,
        )

    def train_agac(self, states, actions, logits, advantages, returns, masks):
        actor_loss_val = 0
        critic_loss_val = 0
        adver_loss_val = 0
        for _ in range(self.epochs):
            for (
                state,
                action,
                old_log_prob,
                logit,
                agac_return,
                agac_advantage,
                mask,
            ) in self.get_minibatch_agac(
                states, actions, logits, advantages, returns, masks
            ):
                new_logits = self.actor(state, mask)
                values = self.critic(state)

                dist = torch.distributions.Categorical(logits=new_logits)
                adver_dist = self.adverserial_net(state, mask)
                old_dist = torch.distributions.Categorical(logits=logit)

                new_log_probs = dist.log_prob(action)

                ratio = (new_log_probs - old_log_prob).exp()
                surr_1 = ratio * agac_advantage
                surr_2 = (
                    torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon)
                    * agac_advantage
                )

                agac_kl_div = torch.distributions.kl_divergence(old_dist, adver_dist)

                entropy = dist.entropy().mean()

                actor_loss = -torch.min(surr_1, surr_2).mean()
                critic_loss = ((values - agac_return) ** 2).mean()
                adver_loss = agac_kl_div.mean()
                loss = (
                    actor_loss + 0.5 * critic_loss + 0.05 * entropy + 4e-5 * adver_loss
                )
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                self.adverserial_optimizer.zero_grad()

                loss.backward()

                self.actor_optimizer.step()
                self.critic_optimizer.step()
                self.adverserial_optimizer.step()

                actor_loss_val += actor_loss.item()
                critic_loss_val += critic_loss.item() * 0.5
                adver_loss_val += adver_loss.item()

        division_factor = self.epochs
        self.anneal_constant()
        return (
            actor_loss_val / division_factor,
            critic_loss_val / division_factor,
            adver_loss_val / division_factor,
        )
