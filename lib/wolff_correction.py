import torch
import math, os
import numpy as np
from rich import print as pprint
import matplotlib.pyplot as plt
from lib.metrics.energy import get_energy
from lib.metrics.helicity_modulus import ps
from lib.metrics.correlation import correlation_function_Cr
from lib.metrics.vortex import vortex_density_torch
from lib.metrics.magnetization import xy_magnetization_susceptibility
from scipy.stats import wasserstein_distance

import torch
import math
import tqdm
from joblib import Parallel, delayed

import torch, wandb

import time

# from sandbox.wasserstein import ensemble_circular_wasserstein
import numpy as np
from scipy.stats import wasserstein_distance

import numpy as np



def percentage_overlap(real_samples, model_samples, bins = 80):
    """
    Computes % overlap using histograms with range determined by min/max of both datasets.
    """

    # Determine shared range from BOTH datasets
    data_min = min(np.min(real_samples), np.min(model_samples))
    data_max = max(np.max(real_samples), np.max(model_samples))

    # Compute histograms with SAME bin edges
    real_hist, bin_edges = np.histogram(real_samples, bins=bins, range=(data_min, data_max))
    model_hist, _ = np.histogram(model_samples, bins=bins, range=(data_min, data_max))

    # Normalize to PMFs
    real_pmf = real_hist / np.sum(real_hist)
    model_pmf = model_hist / np.sum(model_hist)

    # Compute overlap
    overlap = np.sum(np.minimum(real_pmf, model_pmf))

    return overlap

def compute_emd_from_histograms(real_samples, model_samples, num_bins=80):
    """
    Computes EMD (Wasserstein-1) between two empirical distributions
    using histogram-based PMFs.
    
    Args:
        real_samples: array-like, ground truth samples
        model_samples: array-like, generated samples
        num_bins: int, number of histogram bins
        
    Returns:
        emd: float
    """
    
    # 1. Define common bin edges (VERY IMPORTANT)
    min_val = min(real_samples.min(), model_samples.min())
    max_val = max(real_samples.max(), model_samples.max())
    
    bin_edges = np.linspace(min_val, max_val, num_bins + 1)
    
    # 2. Compute histograms
    real_hist, _ = np.histogram(real_samples, bins=bin_edges, density=False)
    model_hist, _ = np.histogram(model_samples, bins=bin_edges, density=False)
    
    # 3. Convert to PMFs
    real_pmf = real_hist / real_hist.sum()
    model_pmf = model_hist / model_hist.sum()
    
    # 4. Compute bin centers (locations of probability mass)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    
    # 5. Compute EMD (weighted version!)
    emd = wasserstein_distance(
        u_values=bin_centers,
        v_values=bin_centers,
        u_weights=real_pmf,
        v_weights=model_pmf
    )
    
    return emd

def get_metrics(theta_cpu, L, beta, reference_samples = None):
    # mean energy and C_v
    _, energies = get_energy(samples = theta_cpu, L = L)

    energies_np = np.array(energies)
    E_mean = np.mean(energies_np)
    E2_mean = np.mean(energies_np**2)
    Cv = beta**2 * (E2_mean - E_mean**2)

    Cv = Cv / (L*L)

    # helicity_modulus
    mean_ps, ps_x, ps_y = ps(samples = theta_cpu, beta = beta)

    # vortex density
    rho_v, w, rho_v_batch = vortex_density_torch(theta = torch.from_numpy(theta_cpu))

    # correlation
    C_r = correlation_function_Cr(theta_cpu)

    # magnetization and susceptibility
    magnetization, susceptibility = xy_magnetization_susceptibility(theta = theta_cpu, beta = beta)
    mean_magnetization = magnetization.mean()

    if reference_samples is not None:
        print(theta_cpu.shape)
        print(reference_samples.shape)
        _, reference_energies = get_energy(samples = reference_samples, L = L)
        reference_magnetization, _ = xy_magnetization_susceptibility(theta = reference_samples, beta = beta)
        print(magnetization.shape)
        print(reference_magnetization.shape)
        energy_emd = wasserstein_distance(energies / (L*L), reference_energies / (L*L))
        magnetization_emd = wasserstein_distance(magnetization, reference_magnetization)
    else:
        energy_emd = 0
        magnetization_emd = 0

    return energies_np, E_mean, Cv, mean_ps, rho_v, C_r, mean_magnetization, susceptibility, energy_emd, magnetization_emd


def get_eps(result_first, result_second):
    if len(result_first) != len(result_second):
        raise ValueError("Result tuples must have the same length")

    mse_list = []

    for a, b in zip(result_first, result_second):
        a = np.asarray(a)
        b = np.asarray(b)

        if a.shape != b.shape:
            a = np.mean(a)
            b = np.mean(b)

        mse = np.mean((a - b) ** 2)
        mse_list.append(mse)


    return mse_list


def xy_energy_torch(theta):
    # theta: (L,L) or (B,L,L) but used per-sample below
    right = torch.roll(theta, shifts=-1, dims=-1)
    down  = torch.roll(theta, shifts=-1, dims=-2)
    return -(torch.cos(theta - right) + torch.cos(theta - down)).sum()

@torch.no_grad()
def wolff_xy_update_single(theta, beta):
    """
    One Wolff cluster update for a single XY configuration.
    theta: torch.Tensor (L,L) on some device
    beta: float
    returns: (theta_new, cluster_size)
    """
    device = theta.device
    L = theta.shape[0]
    assert theta.shape == (L, L)

    # random reflection axis r = (cos phi, sin phi)
    phi = 2.0 * math.pi * torch.rand((), device=device, dtype=theta.dtype)
    rx = torch.cos(phi)
    ry = torch.sin(phi)

    # spins and projections p_i = s_i · r
    sx = torch.cos(theta)
    sy = torch.sin(theta)
    p  = sx * rx + sy * ry  # (L,L)

    # seed
    i0 = torch.randint(0, L, (1,), device=device).item()
    j0 = torch.randint(0, L, (1,), device=device).item()

    in_cluster = torch.zeros((L, L), dtype=torch.bool, device=device)
    stack = [(i0, j0)]
    in_cluster[i0, j0] = True

    # grow cluster (Python loop; OK for L up to ~128 typically)
    while stack:
        i, j = stack.pop()
        pij = p[i, j]

        # 4 neighbors with periodic BC
        neigh = [((i + 1) % L, j),
                 ((i - 1) % L, j),
                 (i, (j + 1) % L),
                 (i, (j - 1) % L)]

        for ni, nj in neigh:
            if in_cluster[ni, nj]:
                continue

            prod = pij * p[ni, nj]
            if prod <= 0:
                continue

            # add-bond probability: 1 - exp(-2 beta prod)
            padd = 1.0 - torch.exp(-2.0 * beta * prod)

            if torch.rand((), device=device) < padd:
                in_cluster[ni, nj] = True
                stack.append((ni, nj))

    sx_new = sx.clone()
    sy_new = sy.clone()
    pc = p[in_cluster]
    sx_new[in_cluster] = sx[in_cluster] - 2.0 * pc * rx
    sy_new[in_cluster] = sy[in_cluster] - 2.0 * pc * ry

    theta_new = torch.atan2(sy_new, sx_new)
    return theta_new, int(in_cluster.sum().item())

@torch.no_grad()
def wolff_cluster_correct_xy(theta0, beta, n_steps=200, device=None, show_progress=True, project = None, diffusion_results = None, mala_results = None, run = None):
    """
    theta0:
      - numpy array (B,L,L) or (L,L)
      - or torch tensor (B,L,L) or (L,L)
    Returns:
      - theta: torch tensor (B,L,L) (or (1,L,L) if input was (L,L))
      - avg_cluster_size: float
      - cluster_sizes: list[int]
    """
    if run == None:
        raise ValueError("Wandb run not initialized")
    if isinstance(theta0, np.ndarray):
        theta = torch.from_numpy(theta0)
    elif torch.is_tensor(theta0):
        theta = theta0
    else:
        raise TypeError("theta0 must be a numpy array or torch tensor")

    if theta.ndim == 2:
        theta = theta.unsqueeze(0)  # (1,L,L)
    assert theta.ndim == 3, "theta must be (B,L,L) or (L,L)"

    if device is not None:
        theta = theta.to(device)

    B, L, _ = theta.shape
    cluster_sizes = []

    iterator = range(n_steps)
    if show_progress:
        iterator = tqdm.tqdm(iterator, desc="Wolff updates")
    C_r_prev = []
    mse_list = []

    print("Staring Wolff")
    project = "wolff benchmarking"
    theta_list = []
    reference_samples = None
    img = False
    previous_configuration = None
    cluster_size = 0
    wolff_total_time = 0

    beta_key = f"beta={beta:.3f}"   # fixed format so keys are stable

    run.define_metric(f"{beta_key}/wolff_steps")
    run.define_metric(f"{beta_key}/*", step_metric=f"{beta_key}/wolff_steps")

    for step in iterator:
        theta_cpu = theta.detach().cpu().numpy()       
        theta_list.append(theta_cpu)

        correction_results = get_metrics(theta_cpu = theta_cpu, L = L, beta = beta)
        expected_energy = correction_results[1]
        heat_capacity = correction_results[2]
        helicity_modulus = correction_results[3]
        vortex_density = correction_results[4]
        correlation_list = correction_results[5]
        magnetization = correction_results[6]
        susceptibility = correction_results[7]
        energy_emd = correction_results[8]
        magnetization_emd = correction_results[9]

        run.log({
            f"{beta_key}/MCMC Energy": expected_energy,
            f"{beta_key}/MCMC Energy per spin": expected_energy / (L*L),
            f"{beta_key}/magnetization": magnetization,
            f"{beta_key}/helicity modulus": helicity_modulus,
            f"{beta_key}/heat capacity": heat_capacity,
            f"{beta_key}/vortex density": vortex_density,
            f"{beta_key}/susceptibility": susceptibility,
            f"{beta_key}/wolff_steps": step,
            f"{beta_key}/L": L,
            f"{beta_key}/beta": beta,
            f"{beta_key}/wolff time": wolff_total_time,
            f"{beta_key}/energy emd": energy_emd,
            f"{beta_key}/magnetization emd": magnetization_emd,
            f"{beta_key}/cluster size": cluster_size,
        })
        if step == 0:
            print("initial correlations is {}".format(correlation_list.tolist()))

        previous_configuration = theta_cpu.copy()

        wolff_step_start = time.time()
        results = Parallel(n_jobs=-1, prefer="processes")(
            delayed(wolff_xy_update_single)(theta[b].clone(), beta)
            for b in tqdm.tqdm(range(B))
        )
        wolff_step_end = time.time()
        wolff_step_time = wolff_step_end - wolff_step_start
        wolff_total_time = wolff_total_time + wolff_step_time

        for b, (theta_b, sz) in enumerate(results):
            theta[b].copy_(theta_b)   # in-place torch update
            cluster_sizes.append(sz)

        theta_cpu = theta.detach().cpu().numpy()
        theta_list.append(theta_cpu) ## error is here
        cluster_size = np.mean(cluster_sizes)

        print("theta cpu is {}".format(theta_cpu.shape))

        if step % 10 == 0:
            print("step {} correlations is {}".format(step, correlation_list.tolist()))
    avg_cluster_size = float(np.mean(cluster_sizes)) if len(cluster_sizes) else 0.0
    return theta, avg_cluster_size, cluster_sizes, mse_list

