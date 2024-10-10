import torch


def GAE(ends, rewards, values, gamma, lambda_):
    returns = []
    values = torch.cat((values, torch.zeros((1,))))
    gae = 0
    for i in reversed(range(len(ends))):
        delta = rewards[i] + gamma * values[i + 1] * (1 - ends[i]) - values[i]
        gae = delta + gamma * lambda_ * (1 - ends[i]) * gae
        returns.insert(0, gae + values[i])
    returns = torch.tensor(returns)
    advantages = returns - values[:-1]
    return returns, (advantages - torch.mean(advantages)) / (
        torch.std(advantages) + 1e-10
    )
