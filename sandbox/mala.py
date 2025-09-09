import numpy as np
from rich import print as pprint
import tqdm
import matplotlib.pyplot as plt

def xy_energy(theta, L, J=1.0):
    theta = theta.reshape((L, L))
    energy = 0.0
    for i in range(L):
        for j in range(L):
            theta_ij = theta[i, j]
            theta_right = theta[i, (j+1) % L]
            theta_down = theta[(i+1) % L, j]
            energy -= J * (np.cos(theta_ij - theta_right) + np.cos(theta_ij - theta_down))
    return energy

def xy_grad(theta, L, J=1.0):
    theta = theta.reshape((L, L))
    grad = np.zeros_like(theta)
    for i in range(L):
        for j in range(L):
            θ = theta[i, j]
            neighbors = [
                theta[i, (j+1)%L], theta[i, (j-1)%L],
                theta[(i+1)%L, j], theta[(i-1)%L, j]
            ]
            grad[i, j] = J * sum(np.sin(θ - n) for n in neighbors)
    return grad.flatten()

def mala_xy(L=4, steps=10000, epsilon=0.01, beta=1.0, eq_cutoff = 200):
    N = L * L
    theta = np.random.uniform(0, 2*np.pi, size=N)
    samples = []

    for step in tqdm.tqdm(range(steps)):
        grad_E = xy_grad(theta, L)
        noise = np.random.randn(N)
        proposal = theta - beta*epsilon * grad_E + np.sqrt(2*epsilon) * noise
        proposal = np.mod(proposal, 2*np.pi)

        E_curr = xy_energy(theta, L)
        E_prop = xy_energy(proposal, L)
        grad_E_prop = xy_grad(proposal, L)

        def log_q(x, y, grad_y):
            diff = x - y + beta*epsilon * grad_y
            return -0.25 * np.sum((diff)**2)/epsilon

        log_p_accept = (
            -beta * E_prop + log_q(theta, proposal, grad_E_prop)
            + beta * E_curr - log_q(proposal, theta, grad_E)
        )

        if np.log(np.random.rand()) < log_p_accept:
            theta = proposal

        if step >= eq_cutoff:
            samples.append(theta)

    return samples

def get_energy(L = 3, beta = 15, num_samples = 4000, steps=20000, eq_cutoff = 200):
    samples = []
    energies = []
    samples = mala_xy(L=L, steps=steps, epsilon=0.15, beta = beta, eq_cutoff = eq_cutoff)
    for i in tqdm.tqdm(range(len(samples))):
        energy =  xy_energy(samples[i], L = L)
        energies.append(energy)
    return samples, energies

if __name__ == "__main__":
    beta = 5
    samples, energies = get_energy(L = 3, beta = beta, num_samples = 10000, steps = 200000, eq_cutoff = 200)
    np.save('sandbox/assets/samples_beta_{}.npy'.format(beta), samples)
    np.save('sandbox/assets/energies_beta_{}.npy'.format(beta), energies)
    pprint("samples shape is {}".format(len(samples)))
    pprint("energies length is {}".format(len(energies)))

    plt.hist(energies, bins = 200, alpha = 0.5)
    plt.xlabel("Energies")
    plt.ylabel("Frequency")
    plt.title("Energy histogram")
    plt.savefig("sandbox/assets/mala_energy_xy_beta_{}.png".format(beta))
    plt.close()
