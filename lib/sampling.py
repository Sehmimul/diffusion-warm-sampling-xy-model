import torch
import numpy as np

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