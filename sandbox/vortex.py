import torch

def angle_wrap_torch(dtheta):
    return (dtheta + torch.pi) % (2 * torch.pi) - torch.pi

def vortex_density_torch(theta):
    """
    theta: (B, L, L) or (L, L) tensor of angles (radians)
    returns:
      rho_v: scalar tensor (mean over batch)
      w: (B, L, L) vorticity map (or (L, L) if input was 2D)
    """
    if theta.dim() == 2:
        theta = theta.unsqueeze(0)  # add batch dim

    B, L, _ = theta.shape

    theta00 = theta
    theta10 = torch.roll(theta, shifts=-1, dims=1)          # i+1, j
    theta11 = torch.roll(theta10, shifts=-1, dims=2)        # i+1, j+1
    theta01 = torch.roll(theta, shifts=-1, dims=2)          # i, j+1

    d1 = angle_wrap_torch(theta10 - theta00)
    d2 = angle_wrap_torch(theta11 - theta10)
    d3 = angle_wrap_torch(theta01 - theta11)
    d4 = angle_wrap_torch(theta00 - theta01)

    w = (d1 + d2 + d3 + d4) / (2 * torch.pi)                # ~ -1, 0, +1

    rho_v_batch = torch.mean(torch.abs(w), dim=(1, 2))      # per config
    rho_v = rho_v_batch.mean()                              # mean over batch

    if w.shape[0] == 1:
        w = w.squeeze(0)
    return rho_v, w, rho_v_batch
