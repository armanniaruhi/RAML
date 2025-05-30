import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.utils
import torch
from torch import optim
import torch.nn.functional as F
from src.ml.resNet18 import SiameseNetwork
from src.preprocessing.dataLoader_vi import SiameseNetworkDataset, imshow, show_plot
from src.ml.loss_utils import ContrastiveLoss
from torch.utils.data import DataLoader
import argparse

def train_model():
    # Load the training dataset
    folder_dataset = datasets.ImageFolder(root="data/v2/data/faces/training")

    # Resize the images and transform to tensors
    transformation = transforms.Compose([
        transforms.Resize((100,100)),
        transforms.ToTensor()
    ])

    # Initialize the network
    siamese_dataset = SiameseNetworkDataset(
        imageFolderDataset=folder_dataset,
        transform=transformation
    )

    # Create a simple dataloader just for simple visualization
    vis_dataloader = DataLoader(
        siamese_dataset,
        shuffle=True,
        num_workers=2,
        batch_size=8
    )

    # Extract one batch
    example_batch = next(iter(vis_dataloader))

    # Example batch is a list containing 2x8 images, indexes 0 and 1, an also the label
    # If the label is 1, it means that it is not the same person, label is 0, same person in both images
    concatenated = torch.cat((example_batch[0], example_batch[1]), 0)
    imshow(torchvision.utils.make_grid(concatenated))

    # Load the training dataset
    train_dataloader = DataLoader(
        siamese_dataset,
        shuffle=True,
        num_workers=8,
        batch_size=64
    )

    net = SiameseNetwork()
    criterion = ContrastiveLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.0005)

    counter = []
    loss_history = [] 
    iteration_number = 0

    # Iterate through the epochs
    for epoch in range(100):
        # Iterate over batches
        for i, (img0, img1, label) in enumerate(train_dataloader, 0):
            # Zero the gradients
            optimizer.zero_grad()

            # Pass in the two images into the network and obtain two outputs
            output1, output2 = net(img0, img1)

            # Pass the outputs of the networks and label into the loss function
            loss_contrastive = criterion(output1, output2, label)

            # Calculate the backpropagation
            loss_contrastive.backward()

            # Optimize
            optimizer.step()

            # Every 10 batches print out the loss
            if i % 10 == 0:
                print(f"Epoch number {epoch}\n Current loss {loss_contrastive.item()}\n")
                iteration_number += 10
                counter.append(iteration_number)
                loss_history.append(loss_contrastive.item())

    show_plot(counter, loss_history)
    return net

def test_model(net):
    # Load the test dataset
    transformation = transforms.Compose([
        transforms.Resize((100,100)),
        transforms.ToTensor()
    ])
    
    folder_dataset_test = datasets.ImageFolder(root="./data/faces/testing/")
    siamese_dataset = SiameseNetworkDataset(
        imageFolderDataset=folder_dataset_test,
        transform=transformation
    )
    test_dataloader = DataLoader(
        siamese_dataset,
        num_workers=2,
        batch_size=1,
        shuffle=True
    )

    # Grab one image that we are going to test
    dataiter = iter(test_dataloader)
    x0, _, _ = next(dataiter)

    for i in range(5):
        # Iterate over 5 images and test them with the first image (x0)
        _, x1, label2 = next(dataiter)

        # Concatenate the two images together
        concatenated = torch.cat((x0, x1), 0)
        
        output1, output2 = net(x0, x1)
        euclidean_distance = F.pairwise_distance(output1, output2)
        imshow(
            torchvision.utils.make_grid(concatenated),
            f'Dissimilarity: {euclidean_distance.item():.2f}'
        )

def main():
    parser = argparse.ArgumentParser(description='Siamese Network Training and Testing')
    parser.add_argument('--test', action='store_true', help='Run in test mode')
    parser.add_argument('--train', action='store_true', help='Run in train mode')
    
    args = parser.parse_args()
    
    if args.train:
        net = train_model()
        # Save the trained model if you want
        torch.save(net.state_dict(), 'siamese_net.pth')
    
    if args.test:
        # Load the trained model
        net = SiameseNetwork()
        net.load_state_dict(torch.load('siamese_net.pth'))
        net.eval()
        test_model(net)

if __name__ == '__main__':
    main()