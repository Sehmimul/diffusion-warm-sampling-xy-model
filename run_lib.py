import torch
import numpy as np
from rich import print as pprint
from lib.sde_lib import VESDE
from lib.model import Model
from lib.utils import get_score_fn
from lib.sampling import sample
import tqdm
import matplotlib.pyplot as plt 

if __name__ == "__main__":
    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # config
    sigma_min=0.01
    sigma_max=50
    N=1000
    eps=1e-5

    # data
    x = torch.randn(10000)*2 + 5
    x = x.to(device)

    t = 1
    t = torch.tensor(t).to(device)

    

    sde = VESDE(sigma_min = sigma_min, sigma_max = sigma_max, N = N)

    model = Model().to(device)

    t = torch.ones(x.shape).to(device)*t
    x = x.unsqueeze(1)

    out = model(x,t)

    num_epoch = 10000
    optim = torch.optim.Adam(model.parameters(), lr=0.001)
    mu = 6
    sigma = 2

    loss_array = []
    batch = x


    for epoch in range(num_epoch):
        # x = (torch.randn(3000)*sigma + mu - 10) + (torch.randn(3000)*sigma + mu + 10)
        ###
        # Means and stds of two Gaussians
        mu = torch.tensor([-3.0, 3.0, 9.0, 15])
        sigma = torch.tensor([1.0, 1.0, 1.0, 1.0])

        # Randomly choose which Gaussian each sample comes from
        choices = torch.randint(0, 4, (3000,))  # 0 or 1

        # Sample from standard normal
        z = torch.randn(3000)

        # Transform using chosen mean and sigma
        x = mu[choices] + sigma[choices] * z

        ###
        x = x.unsqueeze(1)
        x = x.to(device)
        batch = x

        model.train()
        
        score_fn = get_score_fn(sde, model)
        t = torch.rand(batch.shape[0], device=batch.device) * (sde.T - eps) + eps
        z = torch.randn_like(batch)
        mean, std = sde.marginal_prob(batch, t)
        perturbed_data = mean + std[:, None] * z
        score = score_fn(perturbed_data, t)

        losses = torch.square(score * std[:, None] + z)
        losses = torch.mean(losses.reshape(losses.shape[0], -1), dim=-1)
        loss = torch.mean(losses)

        optim.zero_grad()
        loss.backward()
        optim.step()

        loss_array.append(loss.item())
            # progress.display(current_iteration)
        if epoch % 50 == 0:
            pprint("\nepoch: {} and loss is {}".format(epoch, loss.item()))

    x_mean = sample(sde, model)
    arr_np = x_mean.squeeze().detach().cpu().numpy()   # shape [200]
    # Set bin width
    bin_width = 0.5
    bins = np.arange(min(arr_np), max(arr_np) + bin_width, bin_width)

    # z = torch.rand(1000).squeeze().cpu().numpy()*sigma + mu

    ###
    # Means and stds of two Gaussians
    mu = torch.tensor([-3.0, 3.0, 9.0, 15])
    sigma = torch.tensor([1.0, 1.0, 1.0, 1.0])

    # Randomly choose which Gaussian each sample comes from
    choices = torch.randint(0, 4, (1000,))  # 0 or 1

    # Sample from standard normal
    z = torch.randn(1000)

    # Transform using chosen mean and sigma
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
