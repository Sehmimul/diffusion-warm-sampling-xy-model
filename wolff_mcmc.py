import numpy as np
import pandas as pd
import tqdm
import wandb
import torch
import time
import matplotlib.pyplot as plt
from lib.metrics.helicity_modulus import ps
from lib.metrics.vortex import vortex_density_torch
from lib.metrics.magnetization import xy_magnetization_susceptibility
from lib.wolff_correction import get_metrics
from rich import print as pprint

# The code is based on the Julia implementation of the Wolff algorithm for the XY model which can be found at https://github.com/mschuylermoss/705Project/tree/main.

def build_lattice_df(Lx, Ly):
    """
    interactions is a pandas DataFrame where each column is named like "site,x" or "site,y",
    and each cell at row (site-1) holds a Python list of neighbor indices (1-based).
    """
    N = Lx * Ly
    sites = list(range(1, N + 1))
    dirs = ['x', 'y']

    # Create empty DataFrame with columns "1,x", "1,y", "2,x", ...
    columns = [f"{site},{d}" for site in sites for d in dirs]
    interactions = pd.DataFrame({col: [None] * N for col in columns})

    for site in range(1, N + 1):
        # periodic neighbors
        up = (site - 1 - Lx) % N
        down = (site - 1 + Lx) % N

        x_mod = (site - 1) % Lx
        row_start = (site - 1) - x_mod

        right = row_start + ((x_mod + 1) % Lx)
        left = row_start + ((x_mod - 1) % Lx)

        interactions.loc[site - 1, f"{site},x"] = [left + 1, right + 1]
        interactions.loc[site - 1, f"{site},y"] = [up + 1, down + 1]

    return interactions


def rows_and_columns(Lx, Ly):
    """
    Returns:
      rows: DataFrame with columns 'row1', 'row2', ..., each a Series of site indices.
      cols: DataFrame with columns 'col1', 'col2', ..., each a Series of site indices.
    """
    N = Lx * Ly

    row_cols = {f"row{r}": [] for r in range(1, Ly + 1)}
    col_cols = {f"col{c}": [] for c in range(1, Lx + 1)}

    for site in range(1, N + 1):
        row = int((((site - 1) - ((site - 1) % Lx)) / Ly) + 1)
        col = int(((site - 1) % Lx) + 1)

        row_cols[f"row{row}"].append(site)
        col_cols[f"col{col}"].append(site)

    rows_df = pd.DataFrame({k: pd.Series(v) for k, v in row_cols.items()})
    cols_df = pd.DataFrame({k: pd.Series(v) for k, v in col_cols.items()})

    return rows_df, cols_df


# ---------------------------
# XY energy and updates
# ---------------------------

def Calculate_XY_Energy(config, interactions):
    """
    config: 1D numpy array of angles (length N)
    interactions: DataFrame from build_lattice_df
    """
    N = len(config)
    total_energy = 0.0

    for site in range(1, N + 1):
        neighbors_x = interactions.loc[site - 1, f"{site},x"]  # list of 1-based indices
        neighbors_y = interactions.loc[site - 1, f"{site},y"]

        neighbors_x_idx = np.array(neighbors_x) - 1  # to 0-based
        neighbors_y_idx = np.array(neighbors_y) - 1

        energy_x = 0.5 * -1.0 * np.cos(config[site - 1] - config[neighbors_x_idx])
        energy_y = 0.5 * -1.0 * np.cos(config[site - 1] - config[neighbors_y_idx])

        total_energy += energy_x.sum() + energy_y.sum()

    return total_energy


def XY_update(config, energy, interactions, beta, energy_func):
    """
    Single-site global random update (actually global random re-draw in original code).
    """
    # propose random angles
    random_angles = np.random.rand(len(config)) * 2 * np.pi

    new_energy = energy_func(random_angles, interactions)
    deltaE = new_energy - energy

    accept_prob = np.exp(-beta * deltaE)
    rand_value = np.random.rand()

    delta_accept_count = 0

    if accept_prob > rand_value:
        config = random_angles
        delta_accept_count = 1
    else:
        deltaE = 0.0

    return config, deltaE, delta_accept_count


# ---------------------------
# Wolff cluster step (standalone)
# ---------------------------

def Wolff_step(L, neighbours, spins_angles, T):
    """
    neighbours: 2D array-like where neighbours[spin-1, :] gives neighbor indices (1-based).
    spins_angles: 1D numpy array of angles, length L^2.
    """
    N = L ** 2
    alpha = np.random.uniform(0.0, 2.0 * np.pi)

    cluster = []
    spins_to_check = []

    initial_spin = np.random.randint(1, N + 1)  # 1..N
    cluster.append(initial_spin)
    spins_to_check.append(initial_spin)

    while len(spins_to_check) > 0:
        spin_being_checked = spins_to_check[0]
        theta = spins_angles[spin_being_checked - 1]

        all_neighbours = np.atleast_1d(neighbours[spin_being_checked - 1])
        neigh_not_in_cluster = [int(i) for i in all_neighbours if int(i) not in cluster]

        for i in neigh_not_in_cluster:
            theta_neigh = spins_angles[i - 1]
            prob = 1 - np.exp(
                min(0.0, -(2.0 / T) * np.cos(theta - alpha) * np.cos(theta_neigh - alpha))
            )
            if np.random.rand() < prob:
                cluster.append(i)
                spins_to_check.append(i)

        # remove the one we just checked
        spins_to_check.pop(0)

    for i in cluster:
        spins_angles[i - 1] = (np.pi + 2 * alpha - spins_angles[i - 1]) % (2 * np.pi)

    return spins_angles


# ---------------------------
# XY Wolff-like cluster update using interactions DataFrame
# ---------------------------

def XY_cluster_update(config, energy, interactions, beta, energy_func):
    """
    Uses the interactions DataFrame from build_lattice_df.
    """
    N = len(config)

    alpha = np.random.uniform(0.0, 2.0 * np.pi)

    cluster = []
    spins_to_check = []

    initial_spin = np.random.randint(1, N + 1)  # 1..N
    cluster.append(initial_spin)
    spins_to_check.append(initial_spin)

    while len(spins_to_check) > 0:
        site = spins_to_check[0]
        theta = config[site - 1]

        neighbors_x = interactions.loc[site - 1, f"{site},x"]  # list of 1-based
        neighbors_y = interactions.loc[site - 1, f"{site},y"]
        neighbors = list(neighbors_x) + list(neighbors_y)

        neighbors_not_in_cluster = [int(i) for i in neighbors if int(i) not in cluster]

        for i in neighbors_not_in_cluster:
            theta_neighbor = config[i - 1]
            prob = 1 - np.exp(
                min(
                    0.0,
                    -(2.0 * beta)
                    * np.cos(theta - alpha)
                    * np.cos(theta_neighbor - alpha),
                )
            )
            if prob > np.random.rand():
                cluster.append(i)
                spins_to_check.append(i)

        # remove the one we just checked
        spins_to_check.pop(0)

    for i in cluster:
        config[i - 1] = (np.pi + 2 * alpha - config[i - 1]) % (2 * np.pi)

    new_energy = energy_func(config, interactions)
    deltaE = new_energy - energy
    delta_accept_count = 0  # always accept cluster flip in this formulation

    return config, deltaE, delta_accept_count

def spin_stiffness(beta, config, interactions, rows, cols):
    """
    rows, cols are DataFrames from rows_and_columns.
    """
    N = len(config)
    L = int(np.sqrt(N))

    pprint("config is {} with shape {}".format(config, config.shape))
    # pprint("interactions is {}".format(interactions))

    avg_ps_over_rows = 0.0
    for row in range(1, L + 1):
        # sites_in_row: ignore NaNs
        sites_in_row = rows[f"row{row}"].dropna().astype(int).tolist()
        cos_contributions_x = 0.0
        sin_contributions_x = 0.0
        for site in sites_in_row:
            neighbors_x = interactions.loc[site - 1, f"{site},x"]
            neighbors_x_idx = np.array(neighbors_x) - 1
            pprint("in site {} the neighbors_x_idx is {}".format(site, neighbors_x_idx))
            angle_diff_x = config[site - 1] - config[neighbors_x_idx]
            pprint("angle diff x is {}".format(angle_diff_x))
            cos_contributions_x += np.sum(0.5 * np.cos(angle_diff_x))
            sin_contributions_x += np.sum(0.5 * np.sin(angle_diff_x))
        print("sin_contributions_x is {}".format(sin_contributions_x))
        avg_ps_over_rows += (1.0 / N) * cos_contributions_x - (
            (beta / N) * (sin_contributions_x ** 2)
        )

    avg_ps_over_cols = 0.0
    for col in range(1, L + 1):
        sites_in_cols = cols[f"col{col}"].dropna().astype(int).tolist()
        cos_contributions_y = 0.0
        sin_contributions_y = 0.0
        for site in sites_in_cols:
            neighbors_y = interactions.loc[site - 1, f"{site},y"]
            neighbors_y_idx = np.array(neighbors_y) - 1
            angle_diff_y = config[site - 1] - config[neighbors_y_idx]
            cos_contributions_y += np.sum(0.5 * np.cos(angle_diff_y))
            sin_contributions_y += np.sum(0.5 * np.sin(angle_diff_y))
        avg_ps_over_cols += (1.0 / N) * cos_contributions_y - (
            (beta / N) * (sin_contributions_y ** 2)
        )

    spin_stiffness_val = (avg_ps_over_rows + avg_ps_over_cols) / 2.0

    return spin_stiffness_val, avg_ps_over_rows, avg_ps_over_cols




def MonteCarlo_XY(
    beta,
    Lx,
    Ly,
    warmup_steps,
    steps,
    build_lattice_function,
    rows_and_cols_function,
    update_function,
    energy_function,
    spin_stiffness_function,
):

    # system information
    N = Lx * Ly
    interactions = build_lattice_function(Lx, Ly)
    rows, cols = rows_and_cols_function(Lx, Ly)
    initial_config = np.random.rand(N) * 2 * np.pi

    # initialize dataframes
    maindf = pd.DataFrame(
        columns=["avenergy", "avenergy2", "pss", "avpss", "accept_rate"]
    )
    configs = pd.DataFrame()

    # warmup
    config = initial_config.copy()
    configs["initial"] = config
    energy = energy_function(config, interactions)
    for step in range(1, warmup_steps + 1):
        config, deltaE, _ = update_function(
            config, energy, interactions, beta, energy_function
        )
        energy = energy + deltaE
    configs["after_warmup"] = config

    # Monte Carlo sampling
    sumE = 0.0
    sumE2 = 0.0
    sumps = 0.0
    accept_count = 0

    for step in range(1, steps + 1):
        new_config, deltaE, delta_ac = update_function(
            config, energy, interactions, beta, energy_function
        )
        config = new_config
        energy = energy + deltaE
        accept_count += delta_ac

        sumE += energy
        sumE2 += energy ** 2
        ps, _, _ = spin_stiffness_function(beta, config, interactions, rows, cols)
        sumps += ps

        maindf.loc[len(maindf)] = [
            sumE / (step * N),
            sumE2 / step,
            ps,
            sumps / step,
            accept_count / step,
        ]

    configs["final"] = config

    return maindf, configs


def MonteCarlo_XY_with_samples(
    beta,
    Lx,
    Ly,
    warmup_steps,
    steps,
    build_lattice_function,
    rows_and_cols_function,
    update_function,
    energy_function,
    spin_stiffness_function,
    sample_every = 10,
):
    """
    Same as MonteCarlo_XY but also returns:
       samples : np.array of saved configurations
       energies: list of energies for each saved sample
    """

    N = Lx * Ly
    interactions = build_lattice_function(Lx, Ly)
    rows, cols = rows_and_cols_function(Lx, Ly)

    config = np.random.rand(N) * 2 * np.pi
    energy = energy_function(config, interactions)

    samples = []
    warmup_samples = []
    energies = []
    helicity_modulus = []

    project_name = "mcmc"
    with wandb.init(project = project_name) as run:
        # Warmup
        warmup_time = 0
        for step in tqdm.tqdm(range(1, warmup_steps + 1)):
            warmup_time_begin = time.time()
            config, deltaE, _ = update_function(
                config, energy, interactions, beta, energy_function
            )
            warmup_time_end = time.time()
            warmup_time = warmup_time_end - warmup_time_begin + warmup_time
            warmup_samples.append(config.copy())
            
            
            if step % sample_every == 0:
                metric_samples = np.array(warmup_samples).reshape((-1,Lx,Ly))
                energies, E_mean, Cv, mean_ps, rho_v, C_r, magnetization, susceptibility, _, _ = get_metrics(
                    theta_cpu = metric_samples, 
                    L = Lx, 
                    beta = beta
                )
                run.log({
                    "MCMC Energy": E_mean, 
                    "MCMC Energy per spin": E_mean / (Lx * Ly),
                    "heat capacity": Cv, 
                    "helicity modulus": mean_ps,
                    "magnetization": magnetization,
                    "susceptibility": susceptibility,
                    "vortex density": rho_v,
                    "wolff_steps": step, 
                    "L": L, 
                    "beta": beta
                })
            energy += deltaE

        # Main MC loop
        print("warm up time is {}".format(warmup_time))
        main_time = warmup_time
        for step in tqdm.tqdm(range(1, steps + 1)):
            main_time_begin = time.time()
            config, deltaE, _ = update_function(
                config, energy, interactions, beta, energy_function
            )
            main_time_end = time.time()
            main_time = main_time_end - main_time_begin + main_time
            if step < 10 or step % 10000000 == 0:
                if step % sample_every == 0:
                    samples.append(config.copy())
                    energies.append(energy)
                magnetization, susceptibility = xy_magnetization_susceptibility(theta = config.reshape((1, Lx, Ly)), beta = beta)
                rho_v, w, rho_v_batch = vortex_density_torch(theta = torch.from_numpy(config.reshape((1, Lx, Ly))))
                mean_ps, ps_x, ps_y = ps(samples = np.array(samples).reshape((-1,Lx,Ly)), beta = beta)
            run.log({
                "MCMC Energy": energy, 
                "MCMC Energy per spin": energy / (Lx * Ly),
                "helicity modulus": mean_ps,
                "magnetization": magnetization,
                "Susceptibility": susceptibility,
                "vortex density": rho_v,
                "wolff_steps": step, 
                "L": L, 
                "beta": beta,
                "time": main_time
            })
            energy += deltaE

    samples = np.array(samples)
    return samples



if __name__ == "__main__":
    # config
    beta = 1
    L = 20
    steps = 1000
    warmup_steps = 100

    print("Generating samples for beta {} L {}|".format(beta,L))



    samples = MonteCarlo_XY_with_samples(
        beta=beta,
        Lx=L,
        Ly=L,
        warmup_steps=warmup_steps,
        steps=steps,
        build_lattice_function=build_lattice_df,
        rows_and_cols_function=rows_and_columns,
        update_function=XY_cluster_update,
        energy_function=Calculate_XY_Energy,
        spin_stiffness_function=spin_stiffness,
        sample_every=10,
    )

    np.save("assets/data/wolff_samples_beta_{}_L_{}.npy".format(beta, L), np.array(samples))


