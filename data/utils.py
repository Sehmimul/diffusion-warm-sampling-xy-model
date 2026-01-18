from torch.utils.data import DataLoader, random_split
from data.common import DataSet    

def convert_to_dataloader(data, config):
    """
    Converts a tensor to a data loader
    """
    dataset = DataSet(data = data)

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