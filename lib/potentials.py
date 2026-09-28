import plotly.graph_objects as go
import torch
class DoubleWellPotential():
    """
    This class needs to contain code for gradient and the energy
    """
    def __init__(self, theta, beta):
        self.theta = theta
        self.beta = beta

    def energy(self, x):
        vec = x - self.theta
        return 0.5 * torch.sum(vec ** 2, dim=1)

class GaussianMixturePotential:
    """
    Potential defined as a Gaussian mixture.
    Energy = -β log p(x), where p(x) is a weighted Gaussian mixture.
    """
    def __init__(self, mus, sigmas, weights=None, beta=1.0):
        """
        Args:
            mus (list or tensor): list of means (each mean can be scalar or vector)
            sigmas (list): list of standard deviations
            weights (list): list of mixture weights (if None, assume uniform)
            beta (float): inverse temperature scaling
        """
        self.mus = mus
        self.sigmas = sigmas
        self.n_components = mus.size(0)
        self.beta = beta

        if weights is None:
            weights = [1.0 / self.n_components] * self.n_components
        self.weights = weights

    def energy(self, x):
        """
        Compute energy for input x.
        Args:
            x (Tensor): shape (batch, dim)
        Returns:
            Tensor: energy values for each x
        """
        mixture = 0.0
        for mu, sigma, w in zip(self.mus, self.sigmas, self.weights):
            diff = x - mu
            exponent = torch.exp(-0.5 * torch.sum(diff**2, dim=1) / (sigma**2))
            mixture = mixture + w * exponent

        energy = -self.beta * torch.log(mixture + 1e-8)
        return energy


    def grad(self, x):
        """Return gradient of energy wrt x."""
        x = x.clone().detach().requires_grad_(True)
        E = self.energy(x).sum()
        grad = torch.autograd.grad(E, x)[0]
        return grad


def langevin_dynamics(potential, x0, n_steps=1000, step_size=1e-2):
    """
    Run Langevin dynamics.

    Args:
        potential: instance of GaussianMixturePotential
        x0: initial position (batch, dim)
        n_steps: number of steps
        step_size: Langevin step size η
    Returns:
        trajectory: list of samples
    """
    x = x0.clone().detach()
    traj = [x]

    for t in range(n_steps):
        grad = potential.grad(x)
        noise = torch.randn_like(x)
        x = x - step_size * grad + torch.sqrt(torch.tensor(2.0 * step_size)) * noise
        traj.append(x)

    return torch.stack(traj)


def mala(potential, x0, n_steps=1000, step_size=1e-2):
    """
    Run Metropolis-Adjusted Langevin Algorithm (MALA).
    """
    x = x0.clone().detach()
    traj = [x]

    for t in range(n_steps):
        grad = potential.grad(x)
        noise = torch.randn_like(x)
        proposal = x - step_size * grad + torch.sqrt(torch.tensor(2.0 * step_size)) * noise

        # Compute energies
        E_x = potential.energy(x)
        E_y = potential.energy(proposal)

        # Compute proposal densities q(y|x) and q(x|y)
        grad_y = potential.grad(proposal)
        mean_x = x - step_size * grad
        mean_y = proposal - step_size * grad_y

        log_q_y_given_x = -torch.sum((proposal - mean_x)**2, dim=1) / (4 * step_size)
        log_q_x_given_y = -torch.sum((x - mean_y)**2, dim=1) / (4 * step_size)

        # Acceptance probability
        log_alpha = -E_y + E_x + log_q_x_given_y - log_q_y_given_x
        alpha = torch.exp(torch.clamp(log_alpha, max=0.0))

        # Accept/reject
        u = torch.rand_like(alpha)
        accept = u < alpha
        x = torch.where(accept.unsqueeze(1), proposal, x)

        traj.append(x)

    return torch.stack(traj)




def get_weights(x, data_config):
    device = x.device
    beta = data_config.beta
    beta = torch.tensor(beta).to(device)

    if type(data_config).__name__ == "GaussianDataConfig":
        theta = torch.tensor(data_config.mu).to(device)
        potential = DoubleWellPotential(theta = theta, beta = beta)
        energies = potential.energy(x).to(device)

    if type(data_config).__name__ == "GaussianMixtureDataConfig":
        mus = torch.tensor(data_config.mu).to(device)
        sigmas = torch.tensor(data_config.sigma).to(device)
        weights = torch.tensor(data_config.weights).to(device)

        potential = GaussianMixturePotential(mus = mus, sigmas = sigmas, weights = weights, beta = beta)
        energies = potential.energy(x).to(device)

    if type(data_config).__name__ == "XYDataConfig":
        beta = torch.tensor(beta).to(device)
        from sandbox.mala import get_energy
        L = int(x.size(1)**(1/2))
        x = x.cpu().detach().numpy()
        _, energies = get_energy(samples = x, L = L)
        energies = torch.tensor(energies, dtype=torch.float32).to(device)

    logweights = -beta * energies
    max_logweight = torch.max(logweights)
    weights = torch.exp(logweights - max_logweight)

    return energies, weights

if __name__ == "__main__":

    # Example usage:
    # torch.manual_seed(0)
    mus = torch.tensor([-10,4,10])   # two 1D Gaussians
    sigmas = torch.tensor([1,1,1])
    weights = torch.tensor([1/3,1/3,1/3])

    potential = GaussianMixturePotential(mus, sigmas, weights, beta=1.0)

    x0 = torch.randn(2000, 1) # 100 walkers
    trajectory = mala(potential, x0, n_steps=100000, step_size=1e-2)

    print("Final samples shape:", trajectory.shape)  # (steps+1, batch, dim)

    import matplotlib.pyplot as plt

    # Take the last set of samples (shape: batch, dim)
    final_samples = trajectory[-1].detach().cpu().numpy().flatten()

    plt.figure(figsize=(6,4))
    plt.hist(final_samples, bins=50, density=True, alpha=0.5, color="blue", histtype = 'step')
    plt.xlabel("x")
    plt.ylabel("Density (frequency normalized)")
    plt.title("Histogram of Langevin Samples")
    plt.savefig('sandbox/assets/la_gaussian.png')