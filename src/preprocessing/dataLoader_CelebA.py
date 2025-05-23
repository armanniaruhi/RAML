from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets

# https://www.kaggle.com/datasets/kushsheth/face-vae/data

IMG_SIZE = 64
BATCH_SIZE = 128

def load_transformed_dataset():
    data_transforms = [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), # Scales data into [0,1]
        transforms.Lambda(lambda t: (t * 2) - 1) # Scale between [-1, 1]
    ]
    data_transform = transforms.Compose(data_transforms)
    # train=datasets.ImageFolder(root="/kaggle/input/imagenet-train-subset-100k/imagenet_subtrain", transform=data_transform)
    train=datasets.ImageFolder(root="../data/img_align_celeba", transform=data_transform)
    # train = datasets.OxfordIIITPet(root='./data', split="test", download=True, transform=data_transform)

    # test = datasets.OxfordIIITPet(root='./data', split="test", download=True, transform=data_transform)
    # return torch.utils.data.ConcatDataset([train, test])
    return train
def show_tensor_image(image):
    reverse_transforms = transforms.Compose([
        transforms.Lambda(lambda t: (t + 1) / 2),
        transforms.Lambda(lambda t: t.permute(1, 2, 0)), # CHW to HWC
        transforms.Lambda(lambda t: t * 255.),
        transforms.Lambda(lambda t: t.numpy().astype(np.uint8)),
        transforms.ToPILImage(),
    ])

    # Take first image of batch
    if len(image.shape) == 4:
        image = image[0, :, :, :]
    plt.imshow(reverse_transforms(image))

data = load_transformed_dataset()
dataloader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True,num_workers=4)