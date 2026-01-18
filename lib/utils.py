import torch
def get_score_fn(sde, model):
    def score_fn(x, t, beta = None):
        labels = t * 999 # [0, 1] -> [0, 999]
        score = model(x, labels, beta = beta)
        std = sde.marginal_prob(torch.zeros_like(x), t)[1]
        score = -score / std[:, None]
        return score
    return score_fn