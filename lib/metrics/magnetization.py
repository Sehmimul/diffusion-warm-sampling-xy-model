import numpy as np

def xy_magnetization_susceptibility(theta, beta):
    """
    theta: ndarray of shape (B, L, L), angles in radians
    returns:
      m      : scalar RMS magnetization per site
      m_b    : (B,) per-sample magnetization per site
    """
    B, L, _ = theta.shape
    N = L * L

    Mx = np.cos(theta).sum(axis=(-2, -1))   # (B,)
    My = np.sin(theta).sum(axis=(-2, -1))   # (B,)

    M = np.sqrt(Mx**2 + My**2)        # (B,)

    # m_b = M / N                             # per-sample
    # m = m_b.mean()                          # ⟨·⟩ over batch

    m = M / N

    # susceptibility
    term1 = (Mx**2 + My**2).mean()
    term2 = M.mean() **2

    susceptibility = (beta * (term1 - term2)) / N


    return m, susceptibility

if __name__ == "__main__":
    from rich import print as pprint
    beta = 1.5
    L = 20

    mala_out = np.load('sandbox/assets/wolfe_samples_beta_{}_L_{}.npy'.format(beta, L))
    
    mala_out = mala_out.reshape((mala_out.shape[0], L,L))

    magnetization, susceptibility = xy_magnetization_susceptibility(theta = mala_out, beta = beta)

    pprint("magnetization for L : {} beta: {} is {}".format(L, beta, magnetization))
    pprint("susceptibility for L : {} beta: {} is {}".format(L, beta, susceptibility))