import torch
from torch.utils.data import DataLoader, random_split
from data.common import DataSet


def gaussian_mixture_data(config, device, mu = [-3.0, 3.0, 9.0, 15], sigma = [1.0, 1.0, 1.0, 1.0], num_samples = 10000):
    mu = torch.tensor(mu)
    sigma = torch.tensor(sigma)
    choices = torch.randint(0, len(mu), (num_samples,))
    z = torch.randn(num_samples)
    x = mu[choices] + sigma[choices] * z
    x = x.unsqueeze(1)
    x = x.to(device)
    batch = x

    dataset = DataSet(data = x)

    # Split sizes
    train_size = int(config.train_split * len(dataset))
    val_size = int(config.val_split * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    # Dataloaders
    train_loader = DataLoader(train_dataset, batch_size = config.batch_size, shuffle = True)
    val_loader = DataLoader(val_dataset, batch_size = len(val_dataset), shuffle = True)
    test_loader = DataLoader(test_dataset, batch_size = len(test_dataset),  shuffle = True)

    return train_loader, val_loader, test_loader