import torch
import numpy as np
from rich import print as pprint
from lib.sde_lib import VESDE
from lib.model import Model, ScoreNetAngles, ScoreNetAngles1
from lib.sampling import sample
import tqdm
import matplotlib.pyplot as plt 
from config import ModelConfig, GaussianDataConfig, GaussianMixtureDataConfig, XYDataConfig
from train import train_step
from data.gaussian import gaussian_mixture_data
from data.utils import convert_to_dataloader
from data.xy import xy_data
from sandbox.mala import get_energy
import wandb
from lib.potentials import get_weights
from scipy.stats import gaussian_kde
import argparse
import plotly.graph_objects as go
import random

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_type", type=str, required=True)

    args = parser.parse_args()

    data_type = args.data_type

    config = ModelConfig()
    project = config.project_name

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sde = VESDE(sigma_min = config.sigma_min, sigma_max = config.sigma_max, N = config.N)

    model = ScoreNetAngles1().to(device)
    optim = torch.optim.Adam(model.parameters(), lr=config.learning_rate)


    if data_type == 'gaussian':
        data_config = GaussianDataConfig()
        mu = data_config.mu
        sigma = data_config.sigma
        beta = data_config.beta

        # fix so that it only produces one gaussian and not a mixture
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
        # beta = data_config.beta
        betas = data_config.betas
        L = data_config.L
        # train_loader, val_loader, test_loader = xy_data(config = config, device = device, beta = beta, L = L)
        loaders = []
        for beta in betas:
            for l in L:
                loaders.append(xy_data(config = config, device = device, beta = beta, L = l))
        train_loaders = [pair[0] for pair in loaders]
        val_loaders = [pair[1] for pair in loaders]

    variational = config.variational
    ess_counter = 0


    pre_trained = config.pre_trained
    if pre_trained == True:
        pprint("Loading saved checkpoint...")
        model.load_state_dict(torch.load("sandbox/assets/model.pth"))
        pprint("Loaded saved checkpoint")


    wandb_config = {
        'model_config': config.dict(),
        'data_config': data_config.dict()
    }
    with wandb.init(project=project, config=wandb_config) as run:

        for epoch in range(config.num_epochs):
            loss_array = []
            val_loss_array = []

            # if data_type == 'xy':
            #     loader = random.choice(loaders)
            #     train_loader, val_loader, test_loader = loader

            if variational == True:
                if epoch == 0 or ess_counter > max(int(ess * config.ess_fraction), config.min_ess_counter):
                    ess_counter = 0
                    data = sample(sde, model, eps = config.eps, num_samples = config.num_variational_samples)

                    data_np = data.detach().cpu().numpy()

                    # kde = gaussian_kde(data_np)
                    # density = kde(data_np)
                    # print(density.shape)
                    # Suppose data is shape (500, 4, 4)
                    data_np = data_np.reshape((data_np.size(0), -1)).T  # transpose to shape (D, N) for KDE

                    kde = gaussian_kde(data_np, bw_method='scott')  # or 'silverman' or float value

                    # Evaluate density at data points
                    density = kde(data_np) 


                    fig = go.Figure(data=[
                        go.Scatter(x=data_np.reshape((-1)), y=density, mode="markers", line=dict(color="green"))
                    ])
                    fig.update_layout(
                        title="Probability Distribution Function",
                        xaxis_title="data",
                        yaxis_title="density",
                    )
                    wandb.log({"pdf": wandb.Plotly(fig)})
                    weights = get_weights(data, data_config = data_config)[1]
                    weights = weights.detach().cpu().numpy()
                    weights = weights/sum(weights)


                    weights = weights/density
                    weights = weights/sum(weights)

                    ess = (1.0 / sum(weights ** 2))
                    size = max(int(ess), config.min_size)
                    idx = np.random.choice(len(data), p = weights, size = size, replace = False) # previously was 100000
                    data = data[idx]
                    ess_loss = 1-ess

                    run.log({"ess": ess})
                    run.log({"loss/ess_loss": ess_loss})
                    train_loader, val_loader, test_loader = convert_to_dataloader(data = data, 
                                                            config = config)
                ess_counter = ess_counter + 1

                pprint("ess_counter is {} and ess is {}".format(ess_counter, ess))


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


            # for batch in train_loader:
            #     loss = train_step(config, batch, model, sde, optim)
            #     loss_array.append(loss.item())

            # for batch in val_loader:
            #     val_loss = train_step(config, batch, model, sde, optim, train = False)

            if epoch % config.display_epoch == 0:
                # pprint("\nepoch: {} and training loss is {}\tvalidation loss is {}".format(epoch, np.mean(loss_array), val_loss.item()))
                pprint("\nepoch: {} and training loss is {}\tvalidation loss is {}".format(epoch, np.mean(loss_array), np.mean(val_loss_array)))
                torch.save(model.state_dict(), "sandbox/assets/model_L_20_cfg_low_T.pth")

            # run.log({"loss/train_loss": np.mean(loss_array), "loss/validation_loss": val_loss.item()})
            run.log({"loss/train_loss": np.mean(loss_array), "loss/validation_loss": np.mean(val_loss_array)})

    for batch in test_loader:
        test_loss = train_step(config, batch, model, sde, optim, train = False)
        pprint("Test loss is {}".format(test_loss.item()))
