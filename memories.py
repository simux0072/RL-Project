import numpy
import torch
import gae


class Memory:
    def __init__(self, num_games, max_memory_size, dim_size) -> None:
        self.restart(num_games, max_memory_size, dim_size)

    def restart(self, num_games, max_memory_size, dim_size):
        self.idx = 0
        self.dim_size = dim_size
        self.max_memory_size = max_memory_size
        self.num_games = num_games

        self.states = []
        self.actions = []
        self.logits = []
        self.rewards = []
        self.ends = []
        self.values = []
        self.masks = []

        self.state_buf = numpy.zeros(
            (
                self.num_games,
                self.max_memory_size,
                self.dim_size[0],
                self.dim_size[1],
            )
        )
        self.action_buf = numpy.zeros((self.num_games, self.max_memory_size))
        self.logits_buf = torch.zeros(
            (self.num_games, self.max_memory_size, 4), dtype=torch.float64
        )
        self.reward_buf = numpy.zeros((self.num_games, self.max_memory_size))
        self.end_buf = numpy.zeros((self.num_games, self.max_memory_size))
        self.value_buf = torch.zeros(
            (self.num_games, self.max_memory_size), dtype=torch.float64
        )
        self.masks_buf = torch.zeros(
            (num_games, self.max_memory_size, 4), dtype=torch.bool
        )

    def add_new_steps(self, state, action, logits, reward, end, value, mask):
        self.state_buf[:, self.idx] = state
        self.action_buf[:, self.idx] = action
        self.logits_buf[:, self.idx] = logits
        self.reward_buf[:, self.idx] = reward
        self.end_buf[:, self.idx] = end
        self.value_buf[:, self.idx] = value
        self.masks_buf[:, self.idx] = mask

        return self.check_for_end()

    def set_target_values(self, gamma, lambda_):
        self.returns, self.advantages = gae.GAE(
            numpy.concatenate(self.ends),
            numpy.concatenate(self.rewards),
            torch.cat(self.values),
            gamma,
            lambda_,
        )

    def reset_on_steps_reached(self):
        self.states.append(
            self.state_buf[:].reshape(
                -1, self.state_buf.shape[-2], self.state_buf.shape[-1]
            )
        )
        self.actions.append(self.action_buf[:].reshape(-1))

        self.logits.append(self.logits_buf[:].reshape(-1, 4))
        self.rewards.append(self.reward_buf[:].reshape(-1))
        self.ends.append(self.end_buf[:].reshape(-1))
        self.values.append(self.value_buf[:].reshape(-1))
        self.masks.append(self.masks_buf[:].reshape(-1, 4))

    def check_for_end(self):
        end_idx = numpy.where(self.end_buf)[0]
        if len(end_idx) != 0:
            self.states.append(
                self.state_buf[end_idx, : self.idx + 1].reshape(
                    -1, self.state_buf.shape[-2], self.state_buf.shape[-1]
                )
            )
            self.actions.append(self.action_buf[end_idx, : self.idx + 1].reshape(-1))
            self.logits.append(self.logits_buf[end_idx, : self.idx + 1].reshape(-1, 4))
            self.rewards.append(self.reward_buf[end_idx, : self.idx + 1].reshape(-1))
            self.ends.append(self.end_buf[end_idx, : self.idx + 1].reshape(-1))
            self.values.append(self.value_buf[end_idx, : self.idx + 1].reshape(-1))
            self.masks.append(self.masks_buf[end_idx, : self.idx + 1].reshape(-1, 4))

            self.state_buf = numpy.delete(self.state_buf, end_idx, axis=0)
            self.action_buf = numpy.delete(self.action_buf, end_idx, axis=0)
            self.reward_buf = numpy.delete(self.reward_buf, end_idx, axis=0)
            self.end_buf = numpy.delete(self.end_buf, end_idx, axis=0)
            masks = torch.ones(self.value_buf.shape[0], dtype=torch.bool)
            masks[end_idx] = False

            self.logits_buf = self.logits_buf[masks]
            self.value_buf = self.value_buf[masks]
            self.masks_buf = self.masks_buf[masks]

            self.idx += 1
            return True, end_idx
        self.idx += 1
        return False, None

    def get_memory(self, device):
        return (
            torch.from_numpy(numpy.concatenate(self.states))
            .unsqueeze(dim=1)
            .to(device),
            torch.from_numpy(numpy.concatenate(self.actions)).to(device),
            torch.cat(self.logits).to(device),
            self.advantages.to(device),
            self.returns.to(device),
            torch.cat(self.masks).to(device),
        )
