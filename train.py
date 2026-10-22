import torch
import numpy as np
import tqdm
import matplotlib.pyplot as plt 
import wandb
import argparse
import plotly.graph_objects as go
import random
from rich import print as pprint

from lib.sde_lib import VESDE
from lib.model import Model, ScoreNetAngles, RotInvScoreNet
from lib.sampling import sample
from config import ModelConfig, GaussianDataConfig, GaussianMixtureDataConfig, XYDataConfig
from training_step import train_step
from data.gaussian import gaussian_mixture_data
from data.utils import convert_to_dataloader
from data.xy import xy_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_type", type=str, required=True)

    args = parser.parse_args()

    data_type = args.data_type

    config = ModelConfig()
    project = config.project_name

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sde = VESDE(sigma_min = config.sigma_min, sigma_max = config.sigma_max, N = config.N)

    model = RotInvScoreNet().to(device)
    optim = torch.optim.Adam(model.parameters(), lr=config.learning_rate)


    if data_type == 'gaussian':
        data_config = GaussianDataConfig()
        mu = data_config.mu
        sigma = data_config.sigma
        beta = data_config.beta

        train_loader, val_loader, test_loader = gaussian_mixture_data(mu = mu, 
                                                                    sigma = sigma, 
                                                                    num_samples = data_config.num_samples,
                                                                    config = config,
                                                                    device = device)

    if data_type == 'gaussian_mixture':
        data_config = GaussianMixtureDataConfig()
        mu = data_config.mu
        sigma = data_config.sigma
        beta = data_config.beta

        train_loader, val_loader, test_loader = gaussian_mixture_data(mu = mu, 
                                                                    sigma = sigma, 
                                                                    num_samples = data_config.num_samples,
                                                                    config = config,
                                                                    device = device)

    elif data_type == 'xy':
        data_config = XYDataConfig()
        betas = data_config.betas
        L = data_config.L
        loaders = []
        for beta in betas:
            for l in L:
                loaders.append(xy_data(config = config, device = device, beta = beta, L = l))
        train_loaders = [pair[0] for pair in loaders]
        val_loaders = [pair[1] for pair in loaders]

    pre_trained = config.pre_trained
    if pre_trained == True:
        chkpt = config.checkpoint_name
        pprint("Loading saved checkpoint...")
        model.load_state_dict(torch.load("assets/{}.pth".format(chkpt)))
        pprint("Loaded saved checkpoint")


    wandb_config = {
        'model_config': config.dict(),
        'data_config': data_config.dict()
    }
    with wandb.init(project=project, config=wandb_config) as run:

        for epoch in range(config.num_epochs):
            loss_array = []
            val_loss_array = []

            for batches in zip(*train_loaders):
                for loader_idx, batch in enumerate(batches):
                    beta = betas[loader_idx]
                    loss = train_step(config, batch, model, sde, optim, beta = beta)
                    loss_array.append(loss.item())

            for batches in zip(*val_loaders):
                for loader_idx, batch in enumerate(batches):
                    beta = betas[loader_idx]
                    val_loss = train_step(config, batch, model, sde, optim, beta = beta, train = False)
                    val_loss_array.append(val_loss.item())

            if epoch % config.display_epoch == 0:
                pprint("\nepoch: {} and training loss is {}\tvalidation loss is {}".format(epoch, np.mean(loss_array), np.mean(val_loss_array)))
                torch.save(model.state_dict(), "assets/model_checkpoint.pth")

            run.log({"loss/train_loss": np.mean(loss_array), "loss/validation_loss": np.mean(val_loss_array)})

    for batch in test_loader:
        test_loss = train_step(config, batch, model, sde, optim, train = False)
        pprint("Test loss is {}".format(test_loss.item()))
