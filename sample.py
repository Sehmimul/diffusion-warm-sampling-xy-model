import numpy as np
from config import ModelConfig, SamplingConfig
import torch
from rich import print as pprint
import tqdm
import matplotlib.pyplot as plt 
import wandb

from lib.sampling import sample
from lib.sde_lib import VESDE
from lib.sde_lib import VESDE
from lib.model import Model, ScoreNetAngles, ScoreNetAngles1
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
    diffusion_output = sample(
        sde, 
        model, 
        guidance_scale = sampling_config.guidance_scale, 
        beta = beta, 
        eps = model_config.eps, 
        num_samples = sampling_config.num_samples
    )
    diffusion_output = diffusion_output.squeeze().detach().cpu().numpy()
    # if sampling_config.cfg == False:
    #     np.save("sandbox/assets/diffusion_samples_beta_{}_L_{}.npy".format(beta, L), diffusion_output)
    # else:
    #     np.save("sandbox/assets/diffusion_samples_beta_{}_L_{}_cfg_{}.npy".format(beta, L, sampling_config.guidance_scale), diffusion_output)
    
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

    with wandb.init(project = sampling_config.project_name) as run:
        for beta in beta_list:
            # get diffusion samples and then reshape
            pprint("generating results for beta : {}".format(beta))
            diffusion_out = sample_single_beta(model = model, sde = sde, model_config = model_config, sampling_config = sampling_config, beta = beta, L = L)
            diffusion_out = diffusion_out.reshape((diffusion_out.shape[0], L,L))

            # then get the results (metrics) for diffusion
            diffusion_results = get_metrics(theta_cpu = diffusion_out, L = L, beta = beta)

            # get mala samples and then reshape
            mala_out = np.load('sandbox/assets/wolfe_samples_beta_{}_L_{}.npy'.format(beta, L))
            mala_out = mala_out.reshape((mala_out.shape[0], L,L))

            # then get results (metrics) for mala
            mala_results = get_metrics(theta_cpu = mala_out, L = L, beta = beta)

            # get the elements from the respective arrays
            energies_mala, E_mean_mala, Cv_mala, mean_ps_mala, rho_v_mala, C_r_mala, magnetization_mala, susceptibility_mala = mala_results
            energies_diff, E_mean_diff, Cv_diff, mean_ps_diff, rho_v_diff, C_r_diff, magnetization_diff, susceptibility_diff = diffusion_results

            if sampling_config.wolff_correction == True:
                
                theta_corr, avg_sz, sizes, mse_list = wolff_cluster_correct_xy(
                    diffusion_out, 
                    beta = beta, 
                    n_steps=sampling_config.wolff_correction_steps, 
                    project = None, 
                    diffusion_results = None, 
                    mala_results = None
                )

                theta_corr_cpu = theta_corr.detach().cpu().numpy()

                energies, E_mean, Cv, mean_ps, rho_v, C_r, magnetization, susceptibility = get_metrics(
                    theta_cpu = theta_corr_cpu, 
                    L = L, 
                    beta = beta
                )

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
                    "average_cluster_size": avg_sz,
                    "beta": beta,
                    "T": 1/beta,
                    "L": L
                })


if __name__ == "__main__":
    same_every_beta()