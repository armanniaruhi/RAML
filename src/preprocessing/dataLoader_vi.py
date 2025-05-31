import matplotlib.pyplot as plt
import numpy as np
import random
from PIL import Image
from torch.utils.data import Dataset
import torch

class SiameseNetworkDataset(Dataset):
    def __init__(self, imageFolderDataset, transform=None):
        self.imageFolderDataset = imageFolderDataset    
        self.transform = transform
        
    def __getitem__(self, index):
        img0_tuple = random.choice(self.imageFolderDataset.imgs)

        # We need approximately 50% of images to be in the same class
        should_get_same_class = random.randint(0, 1) 
        if should_get_same_class:
            while True:
                # Look until the same class image is found
                img1_tuple = random.choice(self.imageFolderDataset.imgs) 
                if img0_tuple[1] == img1_tuple[1]:
                    break
        else:
            while True:
                # Look until a different class image is found
                img1_tuple = random.choice(self.imageFolderDataset.imgs) 
                if img0_tuple[1] != img1_tuple[1]:
                    break

        img0 = Image.open(img0_tuple[0])
        img1 = Image.open(img1_tuple[0])

        # Convert to RGB if image is in another format (e.g., RGBA)
        if img0.mode != 'RGB':
            img0 = img0.convert('RGB')
        if img1.mode != 'RGB':
            img1 = img1.convert('RGB')

        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)
        
        # Return the images, similarity label (0 or 1), and the original class labels
        return img0, img1, torch.from_numpy(np.array([int(img1_tuple[1] != img0_tuple[1])], dtype=np.float32)), img0_tuple[1], img1_tuple[1]
    
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
    
    
# # Load the training dataset
# folder_dataset = datasets.ImageFolder(root="data/v2/data/faces/training")

# # Resize the images and transform to tensors
# transformation = transforms.Compose([transforms.Resize((100,100)),
#                                      transforms.ToTensor()
#                                     ])

# # Initialize the network
# siamese_dataset = SiameseNetworkDataset(imageFolderDataset=folder_dataset,
#                                         transform=transformation)

# # Create a simple dataloader just for simple visualization
# vis_dataloader = DataLoader(siamese_dataset,
#                         shuffle=True,
#                         num_workers=2,
#                         batch_size=8)

# # Extract one batch
# example_batch = next(iter(vis_dataloader))

# # Example batch is a list containing 2x8 images, indexes 0 and 1, an also the label
# # If the label is 1, it means that it is not the same person, label is 0, same person in both images
# concatenated = torch.cat((example_batch[0], example_batch[1]),0)

# imshow(torchvision.utils.make_grid(concatenated))
# print(example_batch[2].numpy().reshape(-1))