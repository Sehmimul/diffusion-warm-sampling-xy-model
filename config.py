from pydantic import BaseModel
import torch

class ModelConfig(BaseModel):

    # training (common)
    learning_rate: float =  0.0001
    batch_size: int = 500
    num_epochs: int = 60000
    display_epoch: int = 5
    optimizer: str = "adam"
    project_name: str = "diffusion-xy-1"
    sigma_min: float = 0.01
    sigma_max: float = 50
    N: int = 500
    eps: float = 1e-5
    train_split: float = 0.7
    val_split: float = 0.27
    variational = False
    pre_trained = False

    # variational
    num_variational_samples: int = 500
    min_ess_counter: int = 5
    min_size: int = 100
    ess_fraction: float = 1/50

class SamplingConfig(BaseModel):
    # configuration sampling from
    trained_size: int = 20
    L: int = 60
    # beta_list: list = [0.5, 0.6, 0.7, 0.8, 0.82, 0.84, 0.86, 0.88, 0.9, 0.905, 0.91, 0.915, 0.92, 0.94, 0.96, 0.98, 1.0, 1.1, 1.2, 1.25, 1.3, 1.4, 1.5, 1.6]
    beta_list: list = [0.9, 0.905, 0.91, 0.915, 0.92, 0.94, 0.96, 0.98, 1.0, 1.1, 1.2, 1.25, 1.3, 1.4, 1.5, 1.6]

    num_samples: int = 10#2000

    # sampling model checkpoint
    checkpoint: str = 'model_L_20_cfg_all_T'

    # wandb project
    project_name: str = "sampling"

    # classifier free guidance
    cfg: bool = True
    guidance_scale: float = 1.05 # only for CFG

    # Wolff correction
    wolff_correction: bool = True
    wolff_correction_steps: int = 30
    wolff_correction_save_betas: list = [0.5, 1, 1.5]
    train_split: float = 0.7
    val_split: float = 0.27
    batch_size: int = 200
    num_epoch: int = 500
    learning_rate: float =  0.001


class GaussianDataConfig(BaseModel):
    mu = [3]
    sigma = [5]
    num_samples = 10000 # data
    beta: int = 1

class GaussianMixtureDataConfig(BaseModel):
    mu = [-10,4,10]
    sigma = [1,1,1]
    weights = [1/3, 1/3, 1/3]
    num_samples = 100000 # data
    beta: int = 1

class XYDataConfig(BaseModel):
    # beta: float = 0.85
    betas: list = [0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.2, 1.3, 1.4, 1.5, 1.6] #[0.5, 0.6, 0.7, 0.8, 0.9, 1]
    L: list = [20]