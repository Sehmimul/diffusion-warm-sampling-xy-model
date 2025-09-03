import torch
from utils import get_score_fn


def get_loss_fn(sde, eps=1e-5):
    def loss_fn(model, batch):
        score_fn = get_score_fn(sde, model)

        t = torch.rand(batch.shape[0], device=batch.device) * (sde.T - eps) + eps
        z = torch.randn_like(batch)
        mean, std = sde.marginal_prob(batch, t)
        perturbed_data = mean + std[:, None, None, None] * z
        score = score_fn(perturbed_data, t)

        losses = torch.square(score * std[:, None, None, None] + z)
        losses = torch.mean(losses.reshape(losses.shape[0], -1), dim=-1)
        return torch.mean(losses)
    
    return loss_fn