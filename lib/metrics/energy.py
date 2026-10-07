import torch

def get_energy(L = 3, samples = None):
    theta = torch.from_numpy(samples)
    right = torch.roll(theta, shifts=-1, dims=-1)
    down  = torch.roll(theta, shifts=-1, dims=-2)
    energies =  -(torch.cos(theta - right) + torch.cos(theta - down))
    energies = energies.sum(dim=(-2, -1))
    energies = energies.cpu().numpy()
    return samples, energies