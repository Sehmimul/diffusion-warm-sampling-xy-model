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
from lib.model import Model, ScoreNetAngles, ScoreNetAngles1, CorrectionNetwork
from lib.sampling import sample

from config import ModelConfig, XYDataConfig
from train import train_step

from lib.wolff_correction import get_metrics, wolff_cluster_correct_xy

def sample_single_beta(model, sde, model_config, sampling_config, beta: float = 1.0, L: int = 40):
    """
    output: diffusion samples as a numpy array
    """
    gs = sampling_config.guidance_scale
    gs = 1 + (beta ** 2) * 0.10 # annealied guidance
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
    model.load_state_dict(torch.load("assets/{}.pth".format(model_chkpt)))
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

            diffusion_out_list = []
            for i in range(1):
                diffusion_out_list.append(sample_single_beta(model = model, sde = sde, model_config = model_config, sampling_config = sampling_config, beta = beta, L = L))
            diffusion_end = time.time()
            diffusion_out = np.concatenate(diffusion_out_list, axis=0)
            diffusion_out = diffusion_out.reshape((diffusion_out.shape[0], L,L))

            diffusion_time = diffusion_end - diffusion_start
            print("diffusion time is {}".format(diffusion_time))

            diffusion_results = get_metrics(theta_cpu = diffusion_out, L = L, beta = beta)
            energies_diff, E_mean_diff, Cv_diff, mean_ps_diff, rho_v_diff, C_r_diff, magnetization_diff, susceptibility_diff, _, _ = diffusion_results
            
            if sampling_config.wolff_correction == True:
                start = time.time()
                theta_corr, avg_sz, sizes, mse_list = wolff_cluster_correct_xy(
                    diffusion_out, 
                    beta = beta, 
                    n_steps=sampling_config.wolff_correction_steps, 
                    project = None, 
                    diffusion_results = None,
                    run = run 
                )
                end = time.time()
            else:
                print("No Wolff steps performed")
            print("Wolff time is {}".format(end - start))


if __name__ == "__main__":
    same_every_beta()