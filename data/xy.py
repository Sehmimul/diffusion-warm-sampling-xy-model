import numpy as np
import torch
from torch.utils.data import DataLoader, random_split, TensorDataset, ConcatDataset
from data.common import DataSet


def xy_data(config, device, beta = 2, L = 3):
    data = torch.from_numpy(np.load('sandbox/assets/wolfe_samples_beta_{}_L_{}.npy'.format(beta,L))).to(device)
    mean_per_sample = data.mean(dim=1, keepdim=True)   # shape: (B, 1, 1)
    # Subtract
    data = data - mean_per_sample
    dataset = DataSet(data = data)
    # dataset_list = []
    # for l in L:
    #     dataset_list.append(TensorDataset(torch.from_numpy(np.load('sandbox/assets/samples_beta_{}_L_{}.npy'.format(beta,l))).to(device)))
    # dataset = ConcatDataset(dataset_list)

    # Split sizes
    train_size = int(config.train_split * len(dataset))
    val_size = int(config.val_split * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    # Dataloaders
    train_loader = DataLoader(train_dataset, batch_size = config.batch_size, shuffle = True)
    val_loader = DataLoader(val_dataset, batch_size = config.batch_size, shuffle = True)
    test_loader = DataLoader(test_dataset, batch_size = len(test_dataset),  shuffle = True)

    return train_loader, val_loader, test_loader
