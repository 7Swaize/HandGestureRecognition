import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool
from torch.optim import Adam
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split

from config import CSV_PATH, MODEL_PATH, gesture_labels
from utils import load_graph_data_from_csv

# https://medium.com/@andrea.rosales08/introduction-to-graph-neural-networks-78cbb6f64011
# https://github.com/sw-gong/GNN-Tutorial/blob/master/GNN-tutorial-solution.ipynb
# https://github.com/senadkurtisi/pytorch-GCN/tree/main/src

class GCN(nn.Module):
    def __init__(self, out_channels, in_channels=3, dropout_rate=0.2):
        super(GCN, self).__init__()

        self.conv1 = GCNConv(in_channels, 64)
        self.bn1 = nn.BatchNorm1d(64)

        self.conv2 = GCNConv(64, 128)
        self.bn2 = nn.BatchNorm1d(128)

        self.conv3 = GCNConv(128, 256)
        self.bn3 = nn.BatchNorm1d(256)

        self.conv4 = GCNConv(256, 128)
        self.bn4 = nn.BatchNorm1d(128)

        self.conv5 = GCNConv(128, 64)
        self.bn5 = nn.BatchNorm1d(64)

        self.dropout = nn.Dropout(dropout_rate)

        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, out_channels)

    def encode(self, x, edge_index):
        x1 = F.relu(self.bn1(self.conv1(x, edge_index)))
        x1 = self.dropout(x1)

        x2 = F.relu(self.bn2(self.conv2(x1, edge_index)))
        x2 = self.dropout(x2)

        x3 = F.relu(self.bn3(self.conv3(x2, edge_index)))
        x3 = self.dropout(x3)

        x4 = F.relu(self.bn4(self.conv4(x3, edge_index)))
        x4 = self.dropout(x4)

        x5 = F.relu(self.bn5(self.conv5(x4, edge_index)))
        x5 = self.dropout(x5)

        return x1, x2, x3, x4, x5

    def decode(self, features, batch):
        _, z2, _, _, x5 = features

        z2_mean = global_mean_pool(z2, batch)
        z5_mean = global_mean_pool(x5, batch)
        z5_max = global_max_pool(x5, batch)

        z = torch.concat((z2_mean, z5_mean, z5_max), dim=1)

        z = self.dropout(F.relu(self.fc1(z)))
        z = self.dropout(F.relu(self.fc2(z)))
        z = self.fc3(z)

        return z

    def forward(self, data):
        z = self.encode(data.x, data.edge_index)
        return self.decode(z, data.batch)  # data.batch tells which node belongs to which graph

    def train_model(self, train_loader, val_loader=None, epochs=150, lr=0.001, weight_decay=1e-4):
        optimizer = Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

        train_losses = []
        learning_rates = []
        validation_loss = []

        plt.ion()
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))

        start_time = time.time()

        for epoch in range(epochs):
            self.train()
            total_loss = 0

            for data in train_loader:
                optimizer.zero_grad()
                output = self.forward(data)
                loss = criterion(output, data.y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_train_loss = total_loss / len(train_loader)
            train_losses.append(avg_train_loss)

            current_lr = optimizer.param_groups[0]['lr']
            learning_rates.append(current_lr)

            self.eval()
            val_loss = 0

            if val_loader:
                with torch.no_grad():
                    for data in val_loader:
                        output = self.forward(data)
                        loss = criterion(output, data.y)
                        val_loss += loss.item()

                avg_val_loss = val_loss / len(val_loader)
                validation_loss.append(avg_val_loss)

            scheduler.step(avg_val_loss)

            print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, "
                  f"LR: {current_lr:.6f}, Time: {time.time() - start_time:.2f}s")

            self._update_training_plots(ax, train_losses, learning_rates, validation_loss=validation_loss)

        plt.ioff()

    # Graph rendering written by AI
    def _update_training_plots(self, ax, train_losses, learning_rates, validation_loss=None):
        ax[0].clear()
        ax[0].plot(train_losses, 'b-', label='Training Loss')

        if validation_loss and len(validation_loss) > 0:
            ax[0].plot(validation_loss, 'g-', label='Validation Loss')
            ax[0].legend()

        ax[0].set_title('Loss Curves')
        ax[0].set_xlabel('Epochs')
        ax[0].set_ylabel('Loss')

        ax[1].clear()
        ax[1].plot(learning_rates, 'r-')
        ax[1].set_title('Learning Rate')
        ax[1].set_xlabel('Epochs')
        ax[1].set_ylabel('Learning Rate')

        plt.tight_layout()
        plt.draw()
        plt.pause(0.1)


    def predict(self, graph):
        self.eval()
        with torch.no_grad():
            output = self.forward(graph)
            probabilities = F.softmax(output, dim=1)
            predicted_class = torch.argmax(probabilities).item()

            for gesture, idx in gesture_labels.items():
                if idx == predicted_class:
                    class_name = gesture
                    break

            return class_name, probabilities.squeeze().tolist()


def save_model():
    dataset = load_graph_data_from_csv(CSV_PATH)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    model = GCN(out_channels=len(gesture_labels))
    model.train_model(train_loader=train_loader, val_loader=val_loader, epochs=50, lr=0.001)
    torch.save(model.state_dict(), MODEL_PATH)



if __name__ == '__main__':
    save_model()
