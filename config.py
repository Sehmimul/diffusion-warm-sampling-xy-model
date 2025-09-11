from pydantic import BaseModel
import torch

class ModelConfig(BaseModel):

    learning_rate: float =  0.001
    batch_size: int = 250
    num_epochs: int = 1000
    optimizer: str = "adam"

    sigma_min: float = 0.01
    sigma_max: float = 50
    N: int = 1000
    eps: float = 1e-5

    train_split: float = 0.7
    val_split: float = 0.15
