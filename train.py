import torch
from lib.utils import get_score_fn

def train_step(config, batch, model, sde, optim, train = True):
    if train == True:
        model.train()
    else:
        model.eval()

    score_fn = get_score_fn(sde, model) 
    t = torch.rand(batch.shape[0], device=batch.device) * (sde.T - config.eps) + config.eps
    z = torch.randn_like(batch)
    mean, std = sde.marginal_prob(batch, t)
    perturbed_data = mean + std[:, None] * z
    score = score_fn(perturbed_data, t)

    losses = torch.square(score * std[:, None] + z)
    losses = torch.mean(losses.reshape(losses.shape[0], -1), dim=-1)
    loss = torch.mean(losses)

    optim.zero_grad()
    loss.backward()
    optim.step()

    return loss