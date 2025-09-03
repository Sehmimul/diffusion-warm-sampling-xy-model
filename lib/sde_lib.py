import torch
import numpy as np


class VESDE:
  def __init__(self, sigma_min=0.01, sigma_max=50, N=1000):
    """Construct a Variance Exploding SDE.

    Args:
      sigma_min: smallest sigma.
      sigma_max: largest sigma.
      N: number of discretization steps
    """
    super().__init__()
    self.sigma_min = sigma_min
    self.sigma_max = sigma_max
    self.discrete_sigmas = torch.exp(torch.linspace(np.log(self.sigma_min), np.log(self.sigma_max), N))
    self.N = N
    self.probability_flow = False

  @property
  def T(self):
    return 1

  def sde(self, x, t):
    sigma = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
    drift = torch.zeros_like(x)
    diffusion = sigma * torch.sqrt(torch.tensor(2 * (np.log(self.sigma_max) - np.log(self.sigma_min)),
                                                device=t.device))
    return drift, diffusion

  def marginal_prob(self, x, t):
    # understand
    std = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
    mean = x
    return mean, std

  def prior_sampling(self, shape):
    # understand why not standard gaussian
    return torch.randn(*shape) * self.sigma_max

  def prior_logp(self, z):
    # understand
    shape = z.shape
    N = np.prod(shape[1:])
    return -N / 2. * np.log(2 * np.pi * self.sigma_max ** 2) - torch.sum(z ** 2, dim=(1, 2, 3)) / (2 * self.sigma_max ** 2)

  def discretize(self, x, t):
    # understand
    """SMLD(NCSN) discretization."""
    timestep = (t * (self.N - 1) / self.T).long()
    sigma = self.discrete_sigmas.to(t.device)[timestep]
    adjacent_sigma = torch.where(timestep == 0, torch.zeros_like(t),
                                 self.discrete_sigmas[timestep - 1].to(t.device).long()) # added this long
    f = torch.zeros_like(x)
    G = torch.sqrt(sigma ** 2 - adjacent_sigma ** 2)
    return f, G

  def reverse_sde(self, x, t, score_fn):
    """Create the drift and diffusion functions for the reverse SDE/ODE."""
    drift, diffusion = self.sde(x, t)
    # t = torch.ones(x.shape[0]).to(x.device) * t
    score = score_fn(x, t)
    drift = drift - diffusion[:, None] ** 2 * score * (0.5 if self.probability_flow else 1.)
    # Set the diffusion function to zero for ODEs.
    diffusion = 0. if self.probability_flow else diffusion
    return drift, diffusion

  def reverse_discretize(self, x, t):
    """Create discretized iteration rules for the reverse diffusion sampler."""
    f, G = self.discretize(x, t)
    rev_f = f - G[:, None] ** 2 * score_fn(x, t) * (0.5 if self.probability_flow else 1.)
    rev_G = torch.zeros_like(G) if self.probability_flow else G
    return rev_f, rev_G