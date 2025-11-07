
import numpy as np
from lib.sampling import sample
from config import ModelConfig
import torch
from lib.sde_lib import VESDE
import torch
import numpy as np
from rich import print as pprint
from lib.sde_lib import VESDE
from lib.model import Model, ScoreNetAngles, ScoreNetAngles1
from lib.sampling import sample
import tqdm
import matplotlib.pyplot as plt 
from config import ModelConfig
from train import train_step
from data.gaussian import gaussian_mixture_data
from data.xy import xy_data
from sandbox.mala import get_energy, mala_xy
import wandb

# this is the updated sample file which will have helicity modulus as well
project = "inference"
wandb.init(project=project)

beta = 1.2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = ModelConfig()
sde = VESDE(sigma_min = config.sigma_min, sigma_max = config.sigma_max, N = config.N)


model = ScoreNetAngles1().to(device)            # initialize architecture
model.load_state_dict(torch.load("sandbox/assets/model.pth"))
x_mean = sample(sde, model, eps = config.eps, num_samples = 10000)
arr_np = x_mean.squeeze().detach().cpu().numpy()   # shape [200]

np.save("sandbox/assets/diffusion_samples_3.npy", arr_np)
pprint(arr_np.shape)
L = int(np.sqrt(arr_np.shape[1]))

samples, energies = get_energy(samples = arr_np, L = L)
samples = samples.reshape((arr_np.shape[0], L,L))
samples = (samples + np.pi) % (2*np.pi) - np.pi

cmap_choice = 'twilight'

fig, axes = plt.subplots(2, 5, figsize=(14, 6))

for i, ax in enumerate(axes.flat[:10]):
    # Plot heatmap
    i = i
    im = ax.imshow(samples[i], cmap=cmap_choice, vmin=-np.pi, vmax=np.pi)
    ax.set_title(f'Set {i}', fontsize=12)
    ax.axis('off')
    
    # Overlay arrows for XY spins
    Lx, Ly = samples[i].shape
    X, Y = np.meshgrid(np.arange(Ly), np.arange(Lx))
    U = np.cos(samples[i])
    V = np.sin(samples[i])
    
    ax.quiver(X, Y, U, V, scale=12, color='black', width=0.008)

# Adjust layout for colorbar
fig.subplots_adjust(right=0.85, wspace=0.2, hspace=0.3)

# Add a colorbar
cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7])
cbar = fig.colorbar(im, cax=cbar_ax)
cbar.set_label('Angle (radians)', fontsize=14)
cbar.ax.tick_params(labelsize=12)  # change font size for colorbar ticks
plt.title("L: {} and beta: {}".format(L,beta))

wandb.log({"Genrated Angles": wandb.Image(plt)})

plt.savefig('sandbox/assets/angles.png')
plt.close()

arr_np = energies


bin_width = 0.15

# Plot
z = np.load('sandbox/assets/energies_beta_{}_L_{}.npy'.format(beta, L))
bins = np.arange(min(min(arr_np), min(z)), max(max(arr_np), max(z)) + bin_width, bin_width)

plt.hist(arr_np, bins=bins, histtype = "step", alpha = 0.5, label = "diffusion", density = True, linewidth = 0.5)
plt.hist(z, bins=bins, histtype="step", edgecolor="black", linewidth=0.5, label = "exact", density = True)


plt.xlabel("Energy")
plt.ylabel("Frequency")
plt.legend()
plt.title("L: {} and beta: {}".format(L,beta))
wandb.log({"Frequency versus Energy": wandb.Image(plt)})

plt.savefig("assets/hist_xy_beta_{}.png".format(beta))
plt.close()

mean = np.mean(arr_np)
var = np.var(arr_np)

mean_mala = np.mean(z)

pprint("--------------------------")
pprint("mean is {}".format(mean))
pprint("variance is {}".format(var))

pprint("\nmean MALA is {}".format(mean_mala))
pprint("--------------------------")

with wandb.init(project=project, config=config.dict()) as run:
    run.log({"beta": beta, "L": L, "mala_expected_energy": mean_mala, "diffusion_expected_energy": mean})
    print("beta is {}".format(beta))
    print("L is {}".format(L))
    print("diffusion mean is {}".format(mean))
    print("diffusion variance is {}".format(var))
    print("mean MALA is {}".format(mean_mala))
