from src.preprocessing.preprocess import SiameseMNISTLoader
from src.ml.train import train
from src.visualization.plot import plot
from src.ml.losses import ContrastiveLoss

def main():
    # Load the data
    data_loader_wrapper = SiameseMNISTLoader(root='./data', train=True, batch_size=32)
    dataloader = data_loader_wrapper.get_loader()

    # Loss function: Contrastive, TripletLoss or BCE Loss
    contrastive_loss_fn = ContrastiveLoss()
    # triplet_loss_fn = nn.TripletMarginLoss(margin=1.0)
    # bce_loss_fn = nn.BCELoss()

    # Training the model
    train(dataloader=dataloader, loss_func=contrastive_loss_fn, epoch_num= 10, lr = 0.001)

    # Plot the results
    plot()

if __name__ == "__main__":
    main()