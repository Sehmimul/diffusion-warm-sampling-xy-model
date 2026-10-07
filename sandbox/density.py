import torch
from rich import print as pprint

def hutchinson_divergence(f_fn, x, t, n_probes=1, probe='rademacher'):
    """
    Estimate div_x f(x,t) for a batch x using Hutchinson.
    - f_fn(x, t) -> tensor same shape as x, requires_grad=True
    - x: tensor (batch, D) with requires_grad=True
    - t: scalar or tensor (batch,) time argument passed to f_fn
    Returns: div_est of shape (batch,)
    """
    batch = x.shape[0]
    D = x.numel() // batch

    div_est = x.new_zeros(batch)

    # accumulate probes
    for _ in range(n_probes):
        if probe == 'rademacher':
            v = x.new_empty(x.shape).random_(0, 2) * 2 - 1  # ±1
        else:
            v = torch.randn_like(x)

        f = f_fn(x, t)            # (batch, D)
        # make sure we compute gradients wrt x
        # compute scalar S = sum_{i,dim} f_i * v_i  (a single scalar)
        S = (f * v).sum()
        grads = torch.autograd.grad(S, x, create_graph=True)[0]  # (batch, D)
        # per-sample v^T J_f v  = sum_dim (grads * v)
        per_sample = (grads * v).view(batch, -1).sum(dim=1)      # (batch,)
        div_est += per_sample.detach()  # detach to avoid building huge graph

    div_est = div_est / float(n_probes)
    return div_est  # shape (batch,)


# --------- ODE integrator to compute log p(x0) ----------
def estimate_log_prob_samples(x0,
                              f_fn,
                              prior_logprob_fn,
                              T=1.0,
                              n_steps=100,
                              n_hutchinson_probes=1,
                              device=None):
    """
    Estimate log p(x0) for a batch x0 (shape [N, D]) using probability-flow ODE:
      dx/dt = f(x,t)
    We integrate x(t) forward from t=0 to t=T and accumulate -div f along the trajectory.
    Arguments:
      x0: tensor (N, D) data points (at t=0)
      f_fn: callable f_fn(x, t) -> (N, D), deterministic drift
      prior_logprob_fn: callable log p_T(x_T) that returns (N,) tensor
      T: final time where prior is known
      n_steps: Euler steps (increase for accuracy)
      n_hutchinson_probes: number of Hutchinson probes per step
    Returns:
      log_prob_est: tensor (N,) estimated log p(x0)
    """
    if device is None:
        device = x0.device
    x = x0.clone().detach().to(device)
    x.requires_grad_(True)

    dt = T / n_steps
    t = 0.0
    acc = x.new_zeros(x.shape[0])  # accumulate integral of div

    for step in range(n_steps):
        t_tensor = x.new_full((x.shape[0],), t)  # if f_fn expects per-sample t
        # compute f at current x,t
        with torch.set_grad_enabled(True):
            f = f_fn(x, t_tensor)  # (N, D) should be differentiable w.r.t x

        # estimate divergence at current (x,t)
        div_est = hutchinson_divergence(f_fn, x, t_tensor,
                                        n_probes=n_hutchinson_probes,
                                        probe='rademacher')  # (N,)

        # accumulate -div * dt (since d/dt log p = -div f)
        acc = acc - div_est * dt

        # Euler step x <- x + f * dt
        x = (x + f * dt).detach()
        x.requires_grad_(True)
        t += dt

    # now x is x_T, evaluate prior log-prob
    logp_T = prior_logprob_fn(x)  # (N,)

    logp0_est = logp_T + acc  # because log p(x0) = log p_T(x_T) + accumulated ( -int div dt )
    return logp0_est
def gaussian_logprob(x, mean=0.0, sigma=1.0):
    D = x.shape[1]
    var = torch.tensor(sigma ** 2)
    logZ = 0.5 * D * torch.log(2 * torch.pi * var)
    sq = ((x - mean) ** 2).view(x.shape[0], -1).sum(dim=1)
    return -0.5 * sq / var - logZ

if __name__ == "__main__":
    # toy f_fn example - replace with your model's probability-flow drift f(x,t)
    def f_fn(x, t):
        # example: simple linear drift. Replace with actual model drift.
        return -0.5 * x * 0

    # create random samples [N,9] (replace with your samples)
    N = 128
    D = 9
    x0 = torch.randn(N, D)

    # estimate log p(x0)
    logp_est = estimate_log_prob_samples(x0,
                                         f_fn=f_fn,
                                         prior_logprob_fn=lambda x: gaussian_logprob(x, mean=0.0, sigma=1.0),
                                         T=1.0,
                                         n_steps=200,
                                         n_hutchinson_probes=1)
    probs = torch.exp(logp_est)
    pprint("log p estimates:", probs)