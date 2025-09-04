import torch
import numpy as np
from lib.utils import get_score_fn
import tqdm

class EulerMaruyamaPredictor:
  def __init__(self, sde, score_fn):
    super().__init__()
    self.sde = sde
    self.score_fn = score_fn

  def update_fn(self, x, t):
    dt = -1. / self.sde.N
    z = torch.randn_like(x)
    drift, diffusion = self.sde.reverse_sde(x, t, self.score_fn)
    x_mean = x + drift * dt
    x = x_mean + diffusion[:, None] * np.sqrt(-dt) * z
    return x, x_mean


def sample(sde, model, num_samples = 1000, eps=1e-5, device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    score_fn = get_score_fn(sde, model)
    sampler = EulerMaruyamaPredictor(sde = sde, score_fn = score_fn)
    shape = [num_samples, 1]
    x = sde.prior_sampling(shape).to(device)
    timesteps = torch.linspace(sde.T, eps, sde.N, device=device)
    for i in tqdm.tqdm(range(sde.N)):
        t = torch.ones(shape[0], device=device) * timesteps[i]
        x, x_mean = sampler.update_fn(x, t)
    return x_mean