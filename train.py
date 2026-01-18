import torch
from lib.utils import get_score_fn
import numpy as np

def p(x, sigma, N=5):
    p_ = 0
    for i in range(-N, N + 1):
        p_ += np.exp(-(x + 2 * np.pi * i) ** 2 / 2 / sigma ** 2)
    return p_


def grad(x, sigma, N=5):
    p_ = 0
    for i in range(-N, N + 1):
        p_ += (x + 2 * np.pi * i) / sigma ** 2 * np.exp(-(x + 2 * np.pi * i) ** 2 / 2 / sigma ** 2)
    return p_

def data_score(x, sigma):
    score = grad(x, sigma)/p(x, sigma)
    return score

torsional = False

def train_step(config, batch, model, sde, optim, beta = None, train = True):
    if train == True:
        model.train()
    else:
        model.eval()

    if torsional == False:
        score_fn = get_score_fn(sde, model)


        # mean_per_sample = batch.mean(dim=1, keepdim=True)   # shape: (B, 1, 1)
        # # Subtract
        # batch = batch - mean_per_sample



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
    if torsional == True:
        score_fn = get_score_fn(sde, model) 
        batch = (batch + np.pi) % (2*np.pi) - np.pi
        t = torch.rand(batch.shape[0], device=batch.device) * (sde.T - config.eps) + config.eps
        z = torch.randn_like(batch)
        mean, std = sde.marginal_prob(batch, t)
        # z = ((z+np.pi) % (2*np.pi)) - np.pi
        sigma_z = (std[:, None] * z + np.pi) % (2*np.pi) - np.pi
        perturbed_data = mean - sigma_z
        sigma = sde.sigma_min * (sde.sigma_max / sde.sigma_min) ** t
        sigma = sigma.unsqueeze(1)

        perturbed_data = (perturbed_data + np.pi) % (2*np.pi) - np.pi
        diff = sigma_z
        diff = diff.cpu().detach().numpy()
        sigma = sigma.cpu().detach().numpy()
        data_score_value = data_score(x = diff, sigma = sigma)
        data_score_value = torch.from_numpy(data_score_value).to(batch.device)
        score = score_fn(perturbed_data, t)
        losses = std[:, None]*torch.square(score - data_score_value)


    losses = torch.mean(losses.reshape(losses.shape[0], -1), dim=-1)
    loss = torch.mean(losses)

    if train == True:    
        optim.zero_grad()
        loss.backward()
        optim.step()

    return loss