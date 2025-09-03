import torch.nn as nn
import torch
import math

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

        self.fc1 = nn.Linear(1, 256)
        self.fc2 = nn.Linear(256 + self.temb_dim, 512)
        self.drop1 = nn.Dropout(p)

        self.fc3 = nn.Linear(512, 512)
        self.drop2 = nn.Dropout(p)

        self.fc4 = nn.Linear(512, 1)

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        temb = get_timestep_embedding(t, self.temb_dim)

        x = torch.sigmoid(self.fc1(x))
        x = torch.cat([x, temb], dim=-1)
        x = self.fc2(x)
        x = self.drop1(x)
        x = torch.tanh(self.fc3(x))
        x = self.drop2(x)
        x = self.fc4(x)

        return x


        
