import torch
import numpy as np
from rich import print as pprint
from lib.sde_lib import VESDE
from lib.model import Model
from lib.sampling import sample
import tqdm
import matplotlib.pyplot as plt 
from config import ModelConfig
from train import train_step
from data.gaussian import gaussian_mixture_data

if __name__ == "__main__":

    config = ModelConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sde = VESDE(sigma_min = config.sigma_min, sigma_max = config.sigma_max, N = config.N)

    model = Model().to(device)
    optim = torch.optim.Adam(model.parameters(), lr=config.learning_rate)


    mu = [-3.0, 3.0, 9.0, 15]
    sigma = [1.0, 1.0, 1.0, 1.0]
    train_loader, val_loader, test_loader = gaussian_mixture_data(mu = mu  
                                                                sigma = sigma, 
                                                                num_samples = 10000,
                                                                config = config,
                                                                device = device)

    loss_array = []


    for epoch in range(config.num_epochs):

        for batch in train_loader:
            loss = train_step(config, batch, model, sde, optim)
            loss_array.append(loss.item())

        if epoch % 50 == 0:
            for batch in val_loader:
                val_loss = train_step(config, batch, model, sde, optim, train = False)
            pprint("\nepoch: {} and training loss is {}\tvalidation loss is {}".format(epoch, loss.item(), val_loss.item()))

    for batch in test_loader:
        test_loss = train_step(config, batch, model, sde, optim, train = False)
        pprint("Test loss is {}".format(test_loss.item()))


    # code to generate samples
    x_mean = sample(sde, model, eps = config.eps)
    arr_np = x_mean.squeeze().detach().cpu().numpy()   # shape [200]
    # Set bin width
    bin_width = 0.5
    bins = np.arange(min(arr_np), max(arr_np) + bin_width, bin_width)

    mu = mu
    sigma = sigma

    choices = torch.randint(0, 4, (1000,))  # 0 or 1

    z = torch.randn(1000)

    z = mu[choices] + sigma[choices] * z
    z = z.cpu().numpy()

    ###
    # Plot
    plt.hist(arr_np, bins=bins, alpha = 0.5, label = "diffusion")
    plt.hist(z, bins=bins, histtype="step", edgecolor="black", linewidth=0.5, label = "exact")

    plt.xlabel("x")
    plt.ylabel("Frequency")
    plt.legend()
    plt.savefig("assets/hist1d.png")
    plt.close()

    mean = np.mean(arr_np)
    var = np.var(arr_np)

    pprint("--------------------------")
    pprint("mean is {}".format(mean))
    pprint("variance is {}".format(var))
    pprint("--------------------------")
