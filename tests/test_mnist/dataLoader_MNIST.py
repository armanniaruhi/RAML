from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import torch
import random

# Set random seeds for reproducibility
random.seed(42)
torch.manual_seed(42)


# Same as before: Siamese dataset class
class SiameseMNIST(Dataset):
    def __init__(self, mnist_dataset):
        self.mnist = mnist_dataset
        self.targets = self.mnist.targets

    def __getitem__(self, index):
        img1, label1 = self.mnist[index]
        should_get_same_class = random.randint(0, 1)

        if should_get_same_class:
            indices = torch.where(self.targets == label1)[0]
        else:
            indices = torch.where(self.targets != label1)[0]

        # Safer index handling
        indices = indices.tolist()
        if not indices:
            indices = torch.where(self.targets == label1)[0].tolist()

        img2 = self.mnist[random.choice(indices)][0]
        label = torch.tensor(float(should_get_same_class), dtype=torch.float32)

        # Return pair as a tuple
        return (img1, img2), label

    def __len__(self):
        return len(self.mnist)


# New class: handles transform, dataset, and dataloader
class SiameseMNISTLoader:
    def __init__(self, root='./data', train=True, batch_size=32, normalize=False):
        self.root = root
        self.train = train
        self.batch_size = batch_size
        self.normalize = normalize
        self.transform = self._get_transform()
        self.dataset = self._get_dataset()
        self.loader = self._get_dataloader()

    def _get_transform(self):
        t = [transforms.ToTensor()]
        # normalize to [0, 1] range
        if self.normalize:
            t.append(transforms.Normalize((0.5,), (0.5,)))
        return transforms.Compose(t)

    def _get_dataset(self):
        mnist = datasets.MNIST(
            root=self.root,
            train=self.train,
            download=True,
            transform=self.transform
        )
        return SiameseMNIST(mnist)

    def _get_dataloader(self):
        return DataLoader(
            self.dataset,
            shuffle=True,
            batch_size=self.batch_size
        )

    def get_loader(self) -> DataLoader:
        return self.loader
