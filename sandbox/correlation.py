import numpy as np
from rich import print as pprint

def correlation_function_Cr(theta_samples):
    """
    Compute real-space correlation function C(r)
    for XY model samples with periodic boundary conditions.

    Parameters
    ----------
    theta_samples : array, shape (n_samples, L, L)
        XY spin angles in radians.

    Returns
    -------
    C : array of shape (L//2 + 1,)
        Real-space correlation function C(r) averaged over samples
        and over directions (x and y).
        C[0] = 1 by definition.
    """
    n_samples, L, _ = theta_samples.shape
    max_r = L // 2

    C = np.zeros(max_r + 1)

    for s in range(n_samples):
        theta = theta_samples[s]

        for r in range(max_r + 1):
            # Shift in x-direction
            shifted_x = np.roll(theta, shift=-r, axis=0)
            Cx = np.cos(theta - shifted_x).mean()

            # Shift in y-direction
            shifted_y = np.roll(theta, shift=-r, axis=1)
            Cy = np.cos(theta - shifted_y).mean()

            # Average both directions
            C[r] += 0.5 * (Cx + Cy)
            # pprint("Cr is {} with r {}".format(C[r], r))

    C /= n_samples
    return C


if __name__ == "__main__":
    from rich import print as pprint
    import torch
    L = 64
    beta = 0.75
    theta_samples = np.load('sandbox/assets/diffusion_samples_beta_{}_L_{}.npy'.format(beta, L))
    # theta_samples_1 = np.load('sandbox/assets/wolfe_samples_beta_{}_L_{}.npy'.format(beta, L))
    # theta_samples_2 = np.load('sandbox/assets/wolfe_samples_beta_{}_L_{}_v2.npy'.format(beta, L))
    # theta_samples = np.concatenate([theta_samples, theta_samples_2], axis = 0)

    # theta_samples_1 = np.load('sandbox/assets/diffusion_samples_40_6000_1.npy')
    # theta_samples_2 = np.load('sandbox/assets/diffusion_samples_40_6000_2.npy')
    # theta_samples_3 = np.load('sandbox/assets/diffusion_samples_40_6000_3.npy')


    # theta_samples = np.concatenate([theta_samples_1, theta_samples_2], axis = 0)

    pprint("shape of theta is {}".format(theta_samples.shape))
    theta_samples = theta_samples.reshape((theta_samples.shape[0], L,L))

    # Suppose theta_samples has shape (N_samples, L, L)
    C_r = correlation_function_Cr(theta_samples)

    r_list = []
    C_r_list = []
    for r, val in enumerate(C_r):
        # pprint(f"r = {r},  C(r) = {val}")
        r_list.append(r)
        C_r_list.append(val)
    pprint("r is {}".format(r_list))
    pprint("C_r is {}".format(C_r_list))