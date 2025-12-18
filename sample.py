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


def sample_single_beta(model, model_config, sampling_config, beta: float = 1.0, L: int = 40):
    """
    output: diffusion samples as a numpy array
    """
    diffusion_output = sample(sde, model, eps = model_config.eps, sampling_config.num_samples = 3000)
    diffusion_output = x_mean.squeeze().detach().cpu().numpy()
    if sampling_config.cfg == False:
        np.save("sandbox/assets/diffusion_samples_beta_{}_L_{}.npy".format(beta, L), diffusion_output)
    else:
        np.save("sandbox/assets/diffusion_samples_beta_{}_L_{}_cfg_{}.npy".format(beta, L, sampling_config.guidance_scale), diffusion_output)
    
    return diffusion_out

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
    model.load_state_dict(torch.load("sandbox/assets/{}.pth").format(model_chkpt))

    for beta in beta_list:
        sample_single_beta(model = model, model_config = model_config, sampling_config = sampling_config, beta = beta, L = L)

