from torchvision import datasets, transforms
from data_loader.datasets import AbdomenDataset, KidneyDataset
from base import BaseDataLoader


class AbdomenDataLoader(BaseDataLoader):
    """
    Abdomen data loader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize((0.5,), (0.5,))
        ])
        self.data_dir = data_dir
        self.dataset = AbdomenDataset(self.data_dir, train=training, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


class KidneyDataLoader(BaseDataLoader):
    """
    Kidney data loader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize((0.5,), (0.5,))
        ])
        self.data_dir = data_dir
        self.dataset = KidneyDataset(self.data_dir, train=training, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
