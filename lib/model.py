import torch.nn as nn
import torch.nn.functional as F
import torch
import math
import numpy as np

# Time and Beta embeddings

class TimeEmbedding(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.linear = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.SiLU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )

    def forward(self, t):
        half_dim = self.embed_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return self.linear(emb)

class BetaEmbedding(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.linear = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.SiLU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )

    def forward(self, beta):
        """
        beta: shape (B,) or (B,1)
        """
        if beta.dim() == 2:
            beta = beta.squeeze(-1)

        half_dim = self.embed_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=beta.device) * -emb)
        emb = beta[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return self.linear(emb)


# Residual CNN block with FiLM time conditioning
class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 5, padding=2) # 3, padding=1
        self.conv2 = nn.Conv2d(out_ch, out_ch, 5, padding=2)# 3, padding=1
        self.time_mlp = nn.Linear(time_emb_dim, out_ch * 2)
        self.act = nn.SiLU()
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        h = self.conv1(self.act(x))
        # time embedding -> scale & shift (FiLM)
        scale, shift = self.time_mlp(t_emb).chunk(2, dim=1)
        h = h * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        h = self.conv2(self.act(h))
        return h + self.skip(x)

class PeriodicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=0, **kwargs)
        self.pad = padding

    def forward(self, x):
        # Apply periodic (circular) padding in both spatial dimensions
        x = F.pad(x, (self.pad, self.pad, self.pad, self.pad), mode='circular')
        return self.conv(x)

# Main CNN score network
class RotInvScoreNet(nn.Module):
    def __init__(self, lattice_size=4, base_channels=32, time_emb_dim=64, out_mode="theta"):
        
        super().__init__()
        self.out_mode = out_mode
        self.time_embed = TimeEmbedding(time_emb_dim)
        self.conv_in = PeriodicConv2d(2, base_channels, 5, padding=2) #3, padding=1
        self.res1 = ResidualBlock(base_channels, base_channels, time_emb_dim) ## removed
        self.res2 = ResidualBlock(base_channels, base_channels*2, time_emb_dim)
        self.res3 = ResidualBlock(base_channels*2, base_channels*2, time_emb_dim) ## removed
        self.conv_out = PeriodicConv2d(base_channels*2, 2 if out_mode == "cartesian" else 1, 5, padding=2) #3, padding=1
        self.beta_embed = BetaEmbedding(time_emb_dim)
        self.cond_proj = nn.Linear(2 * time_emb_dim, time_emb_dim)
    
    def forward(self, theta, t, beta = None):
        # Convert to periodic embedding
        L = int(theta.size(1)**(1/2))
        theta = theta.reshape((-1,1, L, L))
        x = torch.cat([torch.cos(theta), torch.sin(theta)], dim=1)  # (B, 2, L, L)
        t_emb = self.time_embed(t)

        if beta is None:
            cond = t_emb
        else:
            beta_emb = self.beta_embed(beta)
            cond = self.cond_proj(torch.cat([t_emb, beta_emb], dim=-1))

        h = self.conv_in(x)
        h = self.res1(h, cond)
        h = self.res2(h, cond)
        h = self.res3(h, cond)
        h = self.conv_out(F.silu(h))

        h = h.view(h.size(0), -1)
        return h



# Extra helper classes for future exploratory work

class SinusoidalEmbedding(nn.Module):
    """Sin/cos embedding for scalar beta. Input (B,) or (B,1) -> (B, dim)."""
    def __init__(self, dim: int, max_freq: float = 1000.0):
        super().__init__()
        assert dim % 2 == 0
        half = dim // 2
        freqs = torch.exp(torch.linspace(0, math.log(max_freq), half))
        self.register_buffer("freqs", freqs, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2 and x.size(1) == 1:
            x = x[:, 0]
        angles = x.float()[:, None] * self.freqs[None, :]
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)

def get_timestep_embedding(timesteps: torch.Tensor, embedding_dim: int, downscale_freq_shift: 'float' = 0, max_period: int = 10000):
    assert len(timesteps.shape) == 1, 'Timesteps should be a 1d-array'
    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * torch.arange(start=0, end=half_dim, dtype=torch.float32, device=timesteps.device)
    exponent = exponent / (half_dim - downscale_freq_shift)
    emb = torch.exp(exponent)
    emb = timesteps[:, None].float() * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
    if embedding_dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb

class Model(nn.Module):
    def __init__(self, p=0.5, temb_dim = 100):
        '''
        Score Network.

        n_steps   : perturbation schedule steps (Langevin Dynamic step)
        sigma_min : sigma min of perturbation schedule
        sigma_min : sigma max of perturbation schedule

        '''
        super().__init__()
        self.temb_dim = temb_dim

        self.fc1 = nn.Linear(25, 256)
        self.fc2 = nn.Linear(256 + self.temb_dim, 512)
        self.drop1 = nn.Dropout(p)

        self.fc3 = nn.Linear(512, 512)
        self.drop2 = nn.Dropout(p)

        self.fc4 = nn.Linear(512, 25)

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        temb = get_timestep_embedding(t, self.temb_dim)

        x = torch.sigmoid(self.fc1(x))
        x = torch.cat([x, temb], dim=-1)
        x = self.fc2(x)
        x = self.drop1(x)
        x = torch.tanh(self.fc3(x))
        # x = self.drop2(x)
        x = self.fc4(x)

        return x

class TimeEmbedding_prev(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.proj = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, t):
        half_dim = self.dim // 2
        emb_scale = math.log(10000) / (half_dim - 1)
        freqs = torch.exp(-emb_scale * torch.arange(half_dim, device=t.device))
        args = t[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return self.proj(emb)


# -----------------------------
# Score network for 5x5 XY lattice in angle space
# -----------------------------
class ScoreNetAngles1_prev(nn.Module):
    def __init__(self, n_sites=16, hidden_dim=256, time_dim=64):
        super().__init__()
        self.n_sites = n_sites
        self.time_emb = TimeEmbedding_prev(time_dim)

        input_dim = n_sites
        self.net = nn.Sequential(
            nn.Linear(input_dim + time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, theta, t):
        """
        theta: (batch, 25) — angles in radians
        t: (batch,) — diffusion time ∈ [0,1]
        returns: (batch, 25) — score
        """
        t_emb = self.time_emb(t)
        h = torch.cat([theta, t_emb], dim=-1)
        s = self.net(h)
        return s

class ScoreNetAngles(nn.Module):
    def __init__(self, n_sites=16, hidden_dim=512, time_embed_dim=64):
        super().__init__()
        self.n_sites = n_sites

        # Time embedding using sinusoidal functions
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        # Input: (cos θ_i, sin θ_i) → 2 * n_sites
        input_dim = 2 * n_sites

        self.net = nn.Sequential(
            nn.Linear(input_dim + time_embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, n_sites)
        )

    def forward(self, theta, t):
        """
        theta: tensor of shape (batch_size, 16) in radians
        t: tensor of shape (batch_size, 1), diffusion timestep
        """
        # Encode angles as (cos, sin)
        x = torch.cat([torch.cos(theta), torch.sin(theta)], dim=-1)  # (B, 32)

        # Embed time
        t = t.unsqueeze(1)
        t_embed = self.time_mlp(t)  # (B, time_embed_dim)

        # Concatenate
        h = torch.cat([x, t_embed], dim=-1)

        # Predict score (∇_theta log p_t)
        score = self.net(h)  # (B, 16)

        # Optionally normalize periodic gradients
        return score