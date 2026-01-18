import numpy as np
from sandbox.helicity_modulus import ps
from sandbox.vortex import vortex_density_torch
from rich import print as pprint
from lib.model import ScoreNetAngles1
import torch
######################
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

pprint("Trainable parameters:", count_parameters(model = ScoreNetAngles1()))
######################

L = 40
beta = 1.25


mala_energies = np.load('sandbox/assets/wolfe_energies_beta_{}_L_{}.npy'.format(beta, L))
# diffusion_samples = np.load('sandbox/assets/diffusion_samples_beta_{}_L_{}.npy'.format(beta, L))
# diffusion_samples_1 = np.load('sandbox/assets/diffusion_samples_{}_{}_{}.npy'.format(L, 2000, 1))
# diffusion_samples_2 = np.load('sandbox/assets/diffusion_samples_{}_{}_{}.npy'.format(L, 2000, 2))
# diffusion_samples_3 = np.load('sandbox/assets/diffusion_samples_{}_{}_{}.npy'.format(L, 2000, 3))

# diffusion_samples = np.concatenate([diffusion_samples_1, diffusion_samples_2, diffusion_samples_3], axis = 0)


pprint("shape of mala energies {}".format(mala_energies.shape))
pprint("mean mala is {}".format(np.mean(mala_energies)))
# print(diffusion_samples.shape)

mala_samples = np.load('sandbox/assets/wolfe_samples_beta_{}_L_{}.npy'.format(beta, L))
# mala_samples = np.random.uniform(size = mala_samples.shape)*2*np.pi
pprint("shape of mala samples {}".format(mala_samples.shape))

# ps
def get_ps(samples):
    mean_ps, ps_x, ps_y = ps(samples = samples.reshape((-1,L,L)), beta = beta)
    print("mean_ps is {}".format(mean_ps))
    print("ps_x is {}".format(ps_x))
    print("ps_y is {}".format(ps_y))

pprint("MALA ps")
get_ps(samples = mala_samples)


# vorticity
# mala
rho_v_mala, w_mala, rho_v_batch_mala = vortex_density_torch(theta = torch.from_numpy(mala_samples.reshape((-1,L,L))))
pprint("rho_v_mala is {}".format(rho_v_mala))
# pprint("w_mala is {}".format(w_mala))
# pprint("rho_v_batch_mala is {}".format(rho_v_batch_mala))

# diffusion
# rho_v_diffusion, w_diffusion, rho_v_batch_diffusion = vortex_density_torch(theta = torch.from_numpy(diffusion_samples.reshape((diffusion_samples.shape[0],L,L))))
# pprint("rho_v_diffusion is {}".format(rho_v_diffusion))
# pprint("rho_v_batch_diffusion is {}".format(rho_v_batch_diffusion))

# pprint("Diffusion ps")
# get_ps(samples = diffusion_samples)


############
energies_np = np.array(mala_energies)
E_mean = np.mean(energies_np)
E2_mean = np.mean(energies_np**2)

Cv = beta**2 * (E2_mean - E_mean**2)
Cv = Cv / (L*L)
pprint("mala heat capacity is {}".format(Cv))
###########
