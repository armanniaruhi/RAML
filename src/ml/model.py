import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt

# Sample encoder network
class EmbeddingNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(28 * 28, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.fc(x)

class SiameseNet(nn.Module):
    def __init__(self, embedding_net):
        super().__init__()
        self.embedding_net = embedding_net

    def forward(self, x1, x2):
        return self.embedding_net(x1), self.embedding_net(x2)

    def fit(self, dataloader, loss_func, epoch_num=10, lr=0.001, experiment_name="SiameseNet_Experiment"):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        losses = []

        mlflow.set_experiment(experiment_name)

        with mlflow.start_run():
            mlflow.log_param("learning_rate", lr)
            mlflow.log_param("epochs", epoch_num)
            mlflow.log_param("batch_size", dataloader.batch_size if hasattr(dataloader, 'batch_size') else 'Unknown')

            for epoch in range(epoch_num):
                running_loss = 0.0
                progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epoch_num}")

                for img1, img2, label in progress_bar:
                    out1, out2 = self(img1, img2)
                    loss = loss_func(out1, out2, label)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()

                avg_loss = running_loss / len(dataloader)
                losses.append(avg_loss)
                mlflow.log_metric("train_loss", avg_loss, step=epoch)
                progress_bar.set_postfix(loss=avg_loss)

            # Save and log loss curve
            plt.figure()
            plt.plot(range(1, epoch_num + 1), losses, marker='o')
            plt.title("Training Loss per Epoch")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.grid(True)
            loss_plot_path = "loss_curve.png"
            plt.savefig(loss_plot_path)
            mlflow.log_artifact(loss_plot_path)

            # Save and log model
            model_path = "siamese_model.pt"
            torch.save(self.state_dict(), model_path)
            mlflow.log_artifact(model_path)





