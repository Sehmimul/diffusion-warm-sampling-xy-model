import numpy as np
from config import ModelConfig, SamplingConfig
import torch
from rich import print as pprint
import tqdm
import matplotlib.pyplot as plt 
import wandb
import time

from lib.sampling import sample
from lib.sde_lib import VESDE
from lib.sde_lib import VESDE
from lib.model import Model, ScoreNetAngles, ScoreNetAngles1, CorrectionNetwork
from lib.sampling import sample

from config import ModelConfig, XYDataConfig
from train import train_step

from sandbox.mala import get_energy, mala_xy
from sandbox.helicity_modulus import ps

from sandbox.wolfe_correction import get_metrics, wolff_cluster_correct_xy

def sample_single_beta(model, sde, model_config, sampling_config, beta: float = 1.0, L: int = 40):
    """
    output: diffusion samples as a numpy array
    """
    gs = sampling_config.guidance_scale
    gs = 1 + (beta ** 2) * 0.12 # annealied guidance
    diffusion_output = sample(
        sde, 
        model, 
        L = sampling_config.L,
        guidance_scale = gs, 
        beta = beta, 
        eps = model_config.eps, 
        num_samples = sampling_config.num_samples
    )
    diffusion_output = diffusion_output.squeeze().detach().cpu().numpy()
    # if beta in sampling_config.wolff_correction_save_betas:
    #     if sampling_config.cfg == False:
    #         np.save("sandbox/assets/diffusion_samples_beta_{}_L_{}.npy".format(beta, L), diffusion_output)
    #     else:
    #         np.save("sandbox/assets/diffusion_samples_beta_{}_L_{}_cfg_{}.npy".format(beta, L, sampling_config.guidance_scale), diffusion_output)
    
    return diffusion_output

def same_every_beta(
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ):

    # first we get the config
    model_config = ModelConfig()
    data_config = XYDataConfig()
    sampling_config = SamplingConfig()

    # initialize the SDE
    sde = VESDE(sigma_min = model_config.sigma_min, sigma_max = model_config.sigma_max, N = model_config.N)

    # initialize the model
    model = ScoreNetAngles1().to(device)

    # get beta_list and L
    L = sampling_config.L
    beta_list = sampling_config.beta_list

    # load the trained checkpoint
    model_chkpt = sampling_config.checkpoint
    model.load_state_dict(torch.load("sandbox/assets/{}.pth".format(model_chkpt)))
    mala_beta_present = True
    wandb_config = {
        'sampling_config': sampling_config.dict(),
        'model_config': model_config.dict()
    }
    with wandb.init(project = sampling_config.project_name, config = wandb_config) as run:
        for beta in beta_list:
            # get diffusion samples and then reshape
            pprint("generating results for beta : {}".format(beta))
            diffusion_start = time.time()
            diffusion_out = sample_single_beta(model = model, sde = sde, model_config = model_config, sampling_config = sampling_config, beta = beta, L = L)
            diffusion_end = time.time()
            diffusion_out = diffusion_out.reshape((diffusion_out.shape[0], L,L))
            
            # then get the results (metrics) for diffusion
            diffusion_results = get_metrics(theta_cpu = diffusion_out, L = L, beta = beta)
            # try:
            #     with open('sandbox/assets/wolfe_samples_beta_{}_L_{}.npy'.format(beta, L), "r") as f:
            #         pass
            #     mala_beta_present = True

            #     # # get mala samples and then reshape
            #     try:
            #         mala_out = np.load('sandbox/assets/wolfe_samples_beta_{}_L_{}.npy'.format(beta, L))
            #         mala_out = mala_out.reshape((mala_out.shape[0], L,L))

            #         # then get results (metrics) for mala
            #         mala_results = get_metrics(theta_cpu = mala_out, L = L, beta = beta)

            #         # # get the elements from the respective arrays
            #         energies_mala, E_mean_mala, Cv_mala, mean_ps_mala, rho_v_mala, C_r_mala, magnetization_mala, susceptibility_mala = mala_results
            #     except ValueError:
            #         mala_beta_present = False
            # except FileNotFoundError:
            #     mala_beta_present = False
            energies_diff, E_mean_diff, Cv_diff, mean_ps_diff, rho_v_diff, C_r_diff, magnetization_diff, susceptibility_diff = diffusion_results
            pprint("beta {} is {}".format(beta, mala_beta_present))
            if sampling_config.wolff_correction == True:
                start = time.time()
                theta_corr, avg_sz, sizes, mse_list = wolff_cluster_correct_xy(
                    diffusion_out, 
                    beta = beta, 
                    n_steps=sampling_config.wolff_correction_steps, 
                    project = None, 
                    diffusion_results = None, 
                    mala_results = None
                )
                end = time.time()

                # avg_sz = 0
                # correction_model = CorrectionNetwork().to(device)
                # diffusion_out = torch.from_numpy(diffusion_out).to(device)
                # # print(diffusion_out.device)
                # # 2. Load weights
                # state_dict = torch.load("sandbox/assets/correction_model.pth")
                # beta_tensor = torch.ones(diffusion_out.size(0)).to(device) * beta
                # correction_model.load_state_dict(state_dict)
                # theta_corr = correction_model(diffusion_out.unsqueeze(1), beta_tensor).squeeze(1)
                # # print("theta_coree has dim : {}".format(theta_corr.size()))

                theta_corr_cpu = theta_corr.detach().cpu().numpy()

                # if beta in sampling_config.wolff_correction_save_betas:
                #     if sampling_config.cfg == False:
                #         np.save("sandbox/assets/diffusion_samples_corrected_beta_{}_L_{}.npy".format(beta, L), theta_corr_cpu)
                #     else:
                #         np.save("sandbox/assets/diffusion_samples_corrected_beta_{}_L_{}_cfg_{}.npy".format(beta, L, sampling_config.guidance_scale), theta_corr_cpu)

                energies, E_mean, Cv, mean_ps, rho_v, C_r, magnetization, susceptibility = get_metrics(
                    theta_cpu = theta_corr_cpu, 
                    L = L, 
                    beta = beta
                )
            if mala_beta_present == True:
                print("logging")
                run.log({
                    "Expected Energy/corrected": E_mean,
                    "Expected Energy/diffusion": E_mean_diff,
                    # "Expected Energy/mala": E_mean_mala,
                    "Heat Capacity/corrected": Cv, 
                    "Heat Capacity/diffusion": Cv_diff,
                    # "Heat Capacity/mala": Cv_mala,
                    "Helicity Modulus/corrected": mean_ps,
                    "Helicity Modulus/diffusion": mean_ps_diff,
                    # "Helicity Modulus/mala": mean_ps_mala,                
                    "Vortex Density/corrected": rho_v,
                    "Vortex Density/diffusion": rho_v_diff,
                    # "Vortex Density/mala": rho_v_mala,
                    "magnetization/corrected": magnetization,
                    "magnetization/diffusion": magnetization_diff,
                    # "magnetization/mala": magnetization_mala,
                    "susceptibility/corrected": susceptibility,
                    "susceptibility/diffusion": susceptibility_diff,
                    # "susceptibility/mala": susceptibility_mala,
                    "average_cluster_size": avg_sz,
                    "beta": beta,
                    "T": 1/beta,
                    "L": L,
                    "wolff time": end - start,
                    "diffusion time": diffusion_end - diffusion_start,
                    "annealed_guidance": 1 + (beta ** 2) * 0.12 
                })
                # C(r) plot
                R = len(C_r_diff)
                # assert len(C_r_mala) == R
                assert len(C_r) == R

                r = np.arange(R)

                fig, ax = plt.subplots(figsize=(6, 4))
                colors = {
                    "mala": "#0072B2",       # blue
                    "corrected": "#D55E00",  # vermillion
                    "diffusion": "#009E73",  # green
                }

                # ax.plot(r, C_r_mala, label="MALA", linewidth=1, alpha = 0.5, marker = 's', linestyle = '--', color=colors["mala"])
                ax.plot(r, C_r, label="Corrected", linewidth=1, alpha = 0.5, marker="x", linestyle = '-', color=colors["corrected"])
                ax.plot(r, C_r_diff, label="Diffusion", linewidth=1, alpha = 0.5, marker = "o", linestyle = '--', color=colors["diffusion"])

                ax.set_xlabel("r")
                ax.set_ylabel("C(r)")
                ax.set_title(f"C(r) vs r (beta = {beta}, L = {L})")
                ax.legend()
                ax.grid(True)

                wandb.log({
                    "C_r_plot": wandb.Image(fig),
                })

                plt.close(fig)

                # C(r) log plot
                R = len(C_r_diff)
                # assert len(C_r_mala) == R
                assert len(C_r) == R

                r = np.arange(R)

                fig, ax = plt.subplots(figsize=(6, 4))
                colors = {
                    # "mala": "#0072B2",       # blue
                    "corrected": "#D55E00",  # vermillion
                    "diffusion": "#009E73",  # green
                }

                # ax.plot(r, C_r_mala, label="MALA", linewidth=1, alpha = 0.5, marker = 's', linestyle = '--', color=colors["mala"])
                ax.plot(r, C_r, label="Corrected", linewidth=1, alpha = 0.5, marker="x", linestyle = '-', color=colors["corrected"])
                ax.plot(r, C_r_diff, label="Diffusion", linewidth=1, alpha = 0.5, marker = "o", linestyle = '--', color=colors["diffusion"])
                ax.set_yscale('log')
                ax.set_xlabel("r")
                ax.set_ylabel("C(r)")
                ax.set_title(f"C(r) vs r [log plot] (beta = {beta}, L = {L})")
                ax.legend()
                ax.grid(True)

                wandb.log({
                    "C_r_log_plot": wandb.Image(fig),
                })

                plt.close(fig)


                # plot energy diagram
                # plot histograms
                fig, ax = plt.subplots(figsize=(6, 4))


                # ax.hist(
                #     energies_mala,
                #     bins=50,
                #     density=True,
                #     histtype="step",
                #     linewidth=2,
                #     label="MALA",
                #     alpha = 0.5
                # )
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
                ax.set_title(f"Energy distribution (beta = {beta}, L = {L})")
                ax.grid(alpha=0.3)
                ax.legend()

                wandb.log({
                    "Energy plot": wandb.Image(fig),
                })
                plt.close(fig) 

if __name__ == "__main__":
    same_every_beta()