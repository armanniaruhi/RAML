import matplotlib.pyplot as plt
import numpy as np
import random
from PIL import Image
from torch.utils.data import Dataset
import torch

class SiameseNetworkDataset(Dataset):
    def __init__(self,imageFolderDataset,transform=None, mode="RGB"):
        self.imageFolderDataset = imageFolderDataset    
        self.transform = transform
        self.mode = mode
        
    def __getitem__(self, index):
        while True:
            img0_tuple = random.choice(self.imageFolderDataset.imgs)

            # Approximately 50% same-class pairs
            should_get_same_class = random.randint(0, 1)

            if should_get_same_class:
                img1_tuple = random.choice([
                    x for x in self.imageFolderDataset.imgs if x[1] == img0_tuple[1]
                ])
            else:
                img1_tuple = random.choice([
                    x for x in self.imageFolderDataset.imgs if x[1] != img0_tuple[1]
                ])

            try:
                img0 = Image.open(img0_tuple[0]).convert(self.mode)
                img1 = Image.open(img1_tuple[0]).convert(self.mode)
                break  # success: break the retry loop
            except Exception as e:
                print(f"[Siamese] Error opening images: {img0_tuple[0]}, {img1_tuple[0]} - {e}")
                # Try again with different samples

        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)

        label = torch.tensor([int(img1_tuple[1] != img0_tuple[1])], dtype=torch.float32)

        return img0, img1, label, img0_tuple[1], img1_tuple[1]

    def __len__(self):
        return len(self.imageFolderDataset.imgs)
    
    
# Showing images
def imshow(img, text=None):
    npimg = img.numpy()
    plt.axis("off")
    if text:
        plt.text(75, 8, text, style='italic',fontweight='bold',
            bbox={'facecolor':'white', 'alpha':0.8, 'pad':10})
        
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()    

# Plotting data
def show_plot(iteration,loss):
    plt.plot(iteration,loss)
    plt.show()
    
