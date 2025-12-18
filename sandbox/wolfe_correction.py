import torch
import math, os
import numpy as np
from rich import print as pprint
import matplotlib.pyplot as plt

from mala import get_energy, mala_xy
from helicity_modulus import ps
from correlation import correlation_function_Cr
from vortex import vortex_density_torch
from magnetization import xy_magnetization_susceptibility

import torch
import math
import tqdm
from joblib import Parallel, delayed

import torch, wandb


def get_metrics(theta_cpu, L, beta):
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

    return energies_np, E_mean, Cv, mean_ps, rho_v, C_r, magnetization, susceptibility

# --- XY energy (optional, for debugging) ---
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

    # reflect spins in cluster: s' = s - 2 (s·r) r = s - 2 p r
    sx_new = sx.clone()
    sy_new = sy.clone()
    pc = p[in_cluster]
    sx_new[in_cluster] = sx[in_cluster] - 2.0 * pc * rx
    sy_new[in_cluster] = sy[in_cluster] - 2.0 * pc * ry

    theta_new = torch.atan2(sy_new, sx_new)
    return theta_new, int(in_cluster.sum().item())

@torch.no_grad()
def wolff_cluster_correct_xy(theta0, beta, n_steps=200, device=None, show_progress=True, project = None, diffusion_results = None, mala_results = None):
    """
    Replace your mala_local_correct_xy with Wolff cluster correction.

    theta0:
      - numpy array (B,L,L) or (L,L)
      - or torch tensor (B,L,L) or (L,L)
    Returns:
      - theta: torch tensor (B,L,L) (or (1,L,L) if input was (L,L))
      - avg_cluster_size: float
      - cluster_sizes: list[int]
    """
    # convert input to torch (batch-first)
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
    energies_diff, E_mean_diff, Cv_diff, mean_ps_diff, rho_v_diff, C_r_diff, magnetization_diff, susceptibility_diff = diffusion_results
    energies_mala, E_mean_mala, Cv_mala, mean_ps_mala, rho_v_mala, C_r_mala, magnetization_mala, susceptibility_mala = mala_results

    print("Staring Wolff")
    with wandb.init(project=project) as run:
        for step in iterator:
            # apply one Wolff update per batch element
            # for b in tqdm.tqdm(range(B)):
            #     theta_b, sz = wolff_xy_update_single(theta[b], beta)
            #     theta[b] = theta_b
            #     cluster_sizes.append(sz)

            results = Parallel(n_jobs=-1, prefer="processes")(
                delayed(wolff_xy_update_single)(theta[b].clone(), beta)
                for b in tqdm.tqdm(range(B))
            )

            for b, (theta_b, sz) in enumerate(results):
                theta[b].copy_(theta_b)   # in-place torch update
                cluster_sizes.append(sz)

            theta_cpu = theta.detach().cpu().numpy()

            if step % 10 == 0:
                energies, E_mean, Cv, mean_ps, rho_v, C_r, magnetization, susceptibility = get_metrics(theta_cpu = theta_cpu, L = L, beta = beta)

                run.log({
                    "Expected Energy/corrected": E_mean,
                    "Expected Energy/diffusion": E_mean_diff,
                    "Expected Energy/mala": E_mean_mala,
                    "Heat Capacity/corrected": Cv, 
                    "Heat Capacity/diffusion": Cv_diff,
                    "Heat Capacity/mala": Cv_mala,
                    "Helicity Modulus/corrected": mean_ps,
                    "Helicity Modulus/diffusion": mean_ps_diff,
                    "Helicity Modulus/mala": mean_ps_mala,                
                    "Vortex Density/corrected": rho_v,
                    "Vortex Density/diffusion": rho_v_diff,
                    "Vortex Density/mala": rho_v_mala,
                    "magnetization/corrected": magnetization,
                    "magnetization/diffusion": magnetization_diff,
                    "magnetization/mala": magnetization_mala,
                    "susceptibility/corrected": susceptibility,
                    "susceptibility/diffusion": susceptibility_diff,
                    "susceptibility/mala": susceptibility_mala,
                    "beta": beta,
                    "L": L
                })


                # C(r) plot
                R = len(C_r_mala)
                assert len(C_r_diff) == R
                assert len(C_r) == R

                r = np.arange(R)

                fig, ax = plt.subplots(figsize=(6, 4))
                colors = {
                    "mala": "#0072B2",       # blue
                    "corrected": "#D55E00",  # vermillion
                    "diffusion": "#009E73",  # green
                }

                ax.plot(r, C_r_mala, label="MALA", linewidth=1, alpha = 0.5, marker = 's', linestyle = '--', color=colors["mala"])
                ax.plot(r, C_r, label="Corrected", linewidth=1, alpha = 0.5, marker="x", linestyle = '-', color=colors["corrected"])
                ax.plot(r, C_r_diff, label="Diffusion", linewidth=1, alpha = 0.5, marker = "o", linestyle = '--', color=colors["diffusion"])

                ax.set_xlabel("r")
                ax.set_ylabel("C(r)")
                ax.set_title(f"C(r) vs r (step = {step})")
                ax.legend()
                ax.grid(True)

                wandb.log({
                    "C_r_plot": wandb.Image(fig),
                })

                plt.close(fig)

                # C(r) log plot
                R = len(C_r_mala)
                assert len(C_r_diff) == R
                assert len(C_r) == R

                r = np.arange(R)

                fig, ax = plt.subplots(figsize=(6, 4))
                colors = {
                    "mala": "#0072B2",       # blue
                    "corrected": "#D55E00",  # vermillion
                    "diffusion": "#009E73",  # green
                }

                ax.plot(r, C_r_mala, label="MALA", linewidth=1, alpha = 0.5, marker = 's', linestyle = '--', color=colors["mala"])
                ax.plot(r, C_r, label="Corrected", linewidth=1, alpha = 0.5, marker="x", linestyle = '-', color=colors["corrected"])
                ax.plot(r, C_r_diff, label="Diffusion", linewidth=1, alpha = 0.5, marker = "o", linestyle = '--', color=colors["diffusion"])
                ax.set_yscale('log')
                ax.set_xlabel("r")
                ax.set_ylabel("C(r)")
                ax.set_title(f"C(r) vs r [log plot] (step = {step})")
                ax.legend()
                ax.grid(True)

                wandb.log({
                    "C_r_log_plot": wandb.Image(fig),
                })

                plt.close(fig)


                # plot energy diagram
                # plot histograms
                fig, ax = plt.subplots(figsize=(6, 4))


                ax.hist(
                    energies_mala,
                    bins=50,
                    density=True,
                    histtype="step",
                    linewidth=2,
                    label="MALA",
                    alpha = 0.5
                )
                ax.hist(
                    energies,
                    bins=50,
                    density=True,
                    histtype="step",
                    linewidth=2,
                    label="Corrected",
                    alpha = 0.5
                )
                ax.hist(
                    energies_diff,
                    bins=50,
                    density=True,
                    histtype="step",
                    linewidth=1,
                    label="Diffusion",
                    alpha = 0.5
                )
                ax.set_xlabel("Energy")
                ax.set_ylabel("Density")
                ax.set_title(f"Energy distribution (step = {step})")
                ax.grid(alpha=0.3)
                ax.legend()

                wandb.log({
                    "Energy plot": wandb.Image(fig),
                })
                plt.close(fig)     


    avg_cluster_size = float(np.mean(cluster_sizes)) if len(cluster_sizes) else 0.0
    return theta, avg_cluster_size, cluster_sizes, mse_list


if __name__ == "__main__":
    from rich import print as pprint

    project =  "inference-correction"

    wandb.init(project=project)

    beta = 1.3
    L = 40

    diffusion_out = np.load('sandbox/assets/diffusion_samples_beta_{}_L_{}_g1.8_allT.npy'.format(beta, L))

    pprint("L: {}; beta: {}".format(L, beta))

    diffusion_out = diffusion_out.reshape((diffusion_out.shape[0], L,L))

    diffusion_results = get_metrics(theta_cpu = diffusion_out, L = L, beta = beta)

    # diffusion_results = [E_mean_diff, Cv_diff, mean_ps_diff, rho_v_diff]

    mala_out = np.load('sandbox/assets/wolfe_samples_beta_{}_L_{}.npy'.format(beta, L))
    
    mala_out = mala_out.reshape((mala_out.shape[0], L,L))

    mala_results = get_metrics(theta_cpu = mala_out, L = L, beta = beta)

    # mala_results = [E_mean_mala, Cv_mala, mean_ps_mala, rho_v_mala]
    ###########
    # B, L, _ = diffusion_out.shape

    # diffusion_out = np.random.uniform(
    #     low=0.0,
    #     high=2.0 * np.pi,
    #     size=(B, L, L)
    # )
    # in 10 steps I could correct more or less for [beta = 1, 32] but in 10 steps I cannot generate mcmc samples using the wolfe!
    ###########
    theta_corr, avg_sz, sizes, mse_list = wolff_cluster_correct_xy(diffusion_out, beta = beta, n_steps=100, project = project, diffusion_results = diffusion_results, mala_results = mala_results)
    print("MALA acceptance:", avg_sz)
    pprint("Mse list: {}".format(mse_list))
