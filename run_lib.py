import torch
import numpy as np
from rich import print as pprint
from lib.sde_lib import VESDE
from lib.model import Model
from lib.utils import get_score_fn
from lib.sampling import EulerMaruyamaPredictor
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
    mu = 5
    sigma = 2

    loss_array = []
    batch = x


    for epoch in range(num_epoch):
        x = torch.randn(3000)*sigma + mu
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

score_fn = get_score_fn(sde, model)

sampler = EulerMaruyamaPredictor(sde = sde, score_fn = score_fn)
num_samples = 1000
shape = [num_samples, 1]
x = sde.prior_sampling(shape).to(device)
print(x.size())
timesteps = torch.linspace(sde.T, eps, sde.N, device=device)

for i in tqdm.tqdm(range(sde.N)):
    t = torch.ones(shape[0], device=device) * timesteps[i]
    x, x_mean = sampler.update_fn(x, t)

# pprint(x_mean)
# pprint(x_mean.shape)

# Convert to 1D numpy array for plotting
arr_np = x_mean.squeeze().detach().cpu().numpy()   # shape [200]
# Set bin width
bin_width = 0.5
bins = np.arange(min(arr_np), max(arr_np) + bin_width, bin_width)

z = torch.randn(1000).squeeze().cpu().numpy()*sigma + mu
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
