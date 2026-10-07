import numpy
import torch
import numpy as np
from scipy import integrate
from lib.utils import get_score_fn

def to_flattened_numpy(x):
  """Flatten a torch tensor `x` and convert it to numpy."""
  return x.detach().cpu().numpy().reshape((-1,))


def from_flattened_numpy(x, shape):
  """Form a torch tensor with the given `shape` from a flattened numpy array `x`."""
  return torch.from_numpy(x.reshape(shape))


def get_div_fn(fn):
  """Create the divergence function of `fn` using the Hutchinson-Skilling trace estimator."""

  def div_fn(x, t, eps):
    with torch.enable_grad():
      x.requires_grad_(True)
      fn_eps = torch.sum(fn(x, t) * eps)
      grad_fn_eps = torch.autograd.grad(fn_eps, x)[0]
    x.requires_grad_(False)
    return torch.sum(grad_fn_eps * eps, dim=tuple(range(1, len(x.shape))))

  return div_fn


def get_likelihood_fn(sde, hutchinson_type='Rademacher',
                      rtol=1e-2, atol=1e-2, method='LSODA', eps=1e-5): # all were 1e-5
  """Create a function to compute the unbiased log-likelihood estimate of a given data point.

  Args:
    sde: A `sde_lib.SDE` object that represents the forward SDE.
    hutchinson_type: "Rademacher" or "Gaussian". The type of noise for Hutchinson-Skilling trace estimator.
    rtol: A `float` number. The relative tolerance level of the black-box ODE solver.
    atol: A `float` number. The absolute tolerance level of the black-box ODE solver.
    method: A `str`. The algorithm for the black-box ODE solver.
      See documentation for `scipy.integrate.solve_ivp`.
    eps: A `float` number. The probability flow ODE is integrated to `eps` for numerical stability.

  Returns:
    A function that a batch of data points and returns the log-likelihoods in bits/dim,
      the latent code, and the number of function evaluations cost by computation.
  """

  def drift_fn(model, x, t):
    """The drift function of the reverse-time SDE."""
    # score_fn = get_score_fn(sde, model, train=False, continuous=True)
    score_fn = get_score_fn(sde, model)
    # Probability flow ODE is a special case of Reverse SDE
    rsde = sde.reverse_sde(x, t, score_fn, probability_flow=True)
    return rsde[0]

  def div_fn(model, x, t, noise):
    return get_div_fn(lambda xx, tt: drift_fn(model, xx, tt))(x, t, noise)

  def likelihood_fn(model, data):
    from rich import print as pprint
    """Compute an unbiased estimate to the log-likelihood in bits/dim.

    Args:
      model: A score model.
      data: A PyTorch tensor.

    Returns:
      bpd: A PyTorch tensor of shape [batch size]. The log-likelihoods on `data` in bits/dim.
      z: A PyTorch tensor of the same shape as `data`. The latent representation of `data` under the
        probability flow ODE.
      nfe: An integer. The number of function evaluations used for running the black-box ODE solver.
    """
    with torch.no_grad():
      shape = data.shape
      if hutchinson_type == 'Gaussian':
        epsilon = torch.randn_like(data)
      elif hutchinson_type == 'Rademacher':
        epsilon = torch.randint_like(data, low=0, high=2).float() * 2 - 1.
      else:
        raise NotImplementedError(f"Hutchinson type {hutchinson_type} unknown.")

      def ode_func(t, x):
        # pprint("shape is {}".format(shape))
        # pprint("x is {}".format(x.shape))
        sample = from_flattened_numpy(x[:-shape[0]], shape).to(data.device).type(torch.float32)
        # pprint("sample is {}".format(sample.shape))
        vec_t = torch.ones(sample.shape[0], device=sample.device) * t
        drift = to_flattened_numpy(drift_fn(model, sample, vec_t))
        logp_grad = to_flattened_numpy(div_fn(model, sample, vec_t, epsilon))
        return np.concatenate([drift, logp_grad], axis=0)

      init = np.concatenate([to_flattened_numpy(data), np.zeros((shape[0],))], axis=0)
      solution = integrate.solve_ivp(ode_func, (eps, sde.T), init, rtol=rtol, atol=atol, method=method)
      nfe = solution.nfev
      zp = solution.y[:, -1]
      # pprint(zp)
      z = from_flattened_numpy(zp[:-shape[0]], shape).to(data.device).type(torch.float32)
      delta_logp = from_flattened_numpy(zp[-shape[0]:], (shape[0],)).to(data.device).type(torch.float32)
      prior_logp = sde.prior_logp(z)
      # bpd = -(prior_logp + delta_logp) / np.log(2)
      prob = prior_logp + delta_logp
      bpd = torch.exp(prob)
      N = np.prod(shape[1:])
      # bpd = bpd / N
      # bpd = torch.exp(bpd)
      # A hack to convert log-likelihoods to bits/dim
      return bpd, z, nfe

  return likelihood_fn

if __name__ == "__main__":
  from sde_lib import VESDE
  import matplotlib.pyplot as plt 
  from model import Model
  from rich import print as pprint
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  import wandb
  import plotly.graph_objects as go

  wandb.init(project="prob-viz")

  sde = VESDE()
  likelihood_function = get_likelihood_fn(sde = sde)

  
  beta = 2
  # data = torch.from_numpy(np.load('sandbox/assets/samples_beta_{}.npy'.format(beta))).to(device)[0:5000]
  data1 = torch.from_numpy(np.load("sandbox/assets/diffusion_samples_1.npy")).to(device).reshape((-1,1))
  data2 = torch.from_numpy(np.load("sandbox/assets/diffusion_samples_2.npy")).to(device).reshape((-1,1))
  data3 = torch.from_numpy(np.load("sandbox/assets/diffusion_samples_3.npy")).to(device).reshape((-1,1))
  data4 = torch.from_numpy(np.load("sandbox/assets/diffusion_samples_4.npy")).to(device).reshape((-1,1))
  data5 = torch.from_numpy(np.load("sandbox/assets/diffusion_samples_5.npy")).to(device).reshape((-1,1))
  data6 = torch.from_numpy(np.load("sandbox/assets/diffusion_samples_6.npy")).to(device).reshape((-1,1))
  data7 = torch.from_numpy(np.load("sandbox/assets/diffusion_samples_7.npy")).to(device).reshape((-1,1))

  data = torch.cat((data1,data2,data3,data4,data5))
  # model = Model().to(device)
  # data = torch.from_numpy(np.load("sandbox/assets/diffusion_samples.npy")).to(device).reshape((-1,1))
  model = Model().to(device)            # initialize architecture
  model.load_state_dict(torch.load("sandbox/assets/model.pth"))

  """
  likelihoods = likelihood_function(model = model, data = data)
  prob = likelihoods[0]
  prob = prob/sum(prob)
  pprint(sum(prob))
  probs = prob.cpu().detach().numpy()
  """
  from scipy.stats import gaussian_kde
  kde = gaussian_kde(data.detach().cpu().numpy().reshape((-1)))
  probs = kde(data.detach().cpu().numpy().reshape((-1)))


  values = data.detach().cpu().numpy().reshape((-1)) # replace with your numbers

    # Plot histogram
  n_bins = 200
  bin_edges = np.linspace(values.min(), values.max(), n_bins + 1)

  mu = torch.tensor(3)
  sigma = torch.tensor(1)
  z = torch.randn(10000)
  z = mu + sigma * z
  z = z.cpu().numpy()

  plt.hist(values, bins=bin_edges, alpha = 0.5, density = True, label = "samples")

  from scipy.stats import norm
  mu1, sigma1, w1 = -10, 1.0, 0.15
  mu2, sigma2, w2 = 7, 1.0, 0.85
  # Range of x values
  x = np.linspace(-15, 15, 1000)
  # PDFs for the two Gaussians
  pdf1 = w1 * norm.pdf(x, mu1, sigma1)
  pdf2 = w2 * norm.pdf(x, mu2, sigma2)
  # Mixture PDF
  pdf = pdf1 + pdf2
  N = len(values)

  components = np.random.choice([0, 1], size=N, p=[w1, w2])

  # Sample from chosen Gaussian
  samples = np.where(
      components == 0,
      np.random.normal(mu1, sigma1, N),
      np.random.normal(mu2, sigma2, N)
  )
  plt.plot(x, pdf, 'k', linewidth=1, label="Mixture (2 peaks)")
  plt.hist(samples, bins=bin_edges, density=True, color='red', histtype="step", label="Gaussian Samples")

  ############

  plt.xlabel("Value")
  plt.ylabel("Frequency")
  plt.title("Histogram")
  plt.legend()
  plt.savefig('sandbox/assets/density_hist.png')
  wandb.log({"prob_distribution": wandb.Image(plt)})
  plt.close()

  fig2 = go.Figure(data=[
      go.Scatter(x=values, y=probs, mode="markers", line=dict(color="green"))
  ])
  fig2.update_layout(
      title="Probability Distribution Function",
      xaxis_title="Values",
      yaxis_title="Probability",
  )



  # log histogram
  data = data.detach().cpu().numpy()
  table = wandb.Table(data=data, columns=["bird_scores"])
  wandb.log({'my_histogram': wandb.plot.histogram(table, "bird_scores",
            title="Bird Confidence Scores")})





  wandb.log({"cdf": wandb.Plotly(fig2)})

  wandb.finish()
  

