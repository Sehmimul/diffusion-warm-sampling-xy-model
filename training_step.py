import torch
from lib.utils import get_score_fn

def train_step(config, batch, model, sde, optim, beta = None, train = True):
    if train == True:
        model.train()
    else:
        model.eval()

    score_fn = get_score_fn(sde, model)

    t = torch.rand(batch.shape[0], device=batch.device) * (sde.T - config.eps) + config.eps
    z = torch.randn_like(batch)
    mean, std = sde.marginal_prob(batch, t)
    perturbed_data = mean + std[:, None] * z
    if beta == None:       
        score = score_fn(perturbed_data, t)
    else:
        p_drop = 0.1
        beta = torch.ones(batch.shape[0], device = batch.device) * beta
        drop_mask = (torch.rand(beta.shape[0], device=batch.device) < p_drop)
        beta[drop_mask] = torch.tensor(0.0, device = batch.device)
        score = score_fn(perturbed_data, t, beta = beta)

    losses = torch.square(score * std[:, None] + z)
    losses = torch.mean(losses.reshape(losses.shape[0], -1), dim=-1)
    loss = torch.mean(losses)

    if train == True:    
        optim.zero_grad()
        loss.backward()
        optim.step()

    return loss