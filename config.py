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
    N: int = 1000*2
    eps: float = 1e-5
    train_split: float = 0.7
    val_split: float = 0.15
    variational = False

    # variational
    num_variational_samples: int = 500
    min_ess_counter: int = 5
    min_size: int = 100
    ess_fraction: float = 1/50

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
    beta: int = 1.25
    L: list = [4,6,8]