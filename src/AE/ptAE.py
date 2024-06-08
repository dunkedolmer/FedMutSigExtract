import os
import numpy as np
import pandas as pd
import torch
from torch import nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def prepare_data(df: pd.DataFrame):
    test_set_percent = 0.1
    noise_factor = 0.0

    df = df.div(df.sum(axis=1), axis=0)

    x_test = df.sample(frac=test_set_percent)
    x_train = df.drop(x_test.index)
    x_train_noisy = x_train + noise_factor * np.random.normal(
        loc=0.0, scale=1.0, size=x_train.shape
    )
    x_test_noisy = x_test + noise_factor * np.random.normal(
        loc=0.0, scale=1.0, size=x_test.shape
    )
    x_train_noisy = np.clip(x_train_noisy, 0.0, 1.0)
    x_test_noisy = np.clip(x_test_noisy, 0.0, 1.0)

    return x_train, x_train_noisy, x_test, x_test_noisy


class DeepMS(nn.Module):
    def __init__(self, original_dim, encoding_dim):
        super(DeepMS, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(original_dim, encoding_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, original_dim),
            nn.Softmax(dim=-1),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def encode(self, x):
        x = self.encoder(x)
        return x

    def decode(self, x):
        x = self.decoder(x)
        return x


# Create dataset from pandas dataframe
class Dataset(torch.utils.data.Dataset):
    def __init__(self, df1, df2):
        self.x1 = torch.tensor(df1.values, dtype=torch.float32)
        self.x2 = torch.tensor(df2.values, dtype=torch.float32)

    def __len__(self):
        return len(self.x1)

    def __getitem__(self, idx):
        return self.x1[idx], self.x2[idx]


def train_model(
    epochs: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
):
    total_train_loss = []
    total_test_loss = []
    for e in range(epochs):
        train_losses = []
        for x_n, x_o in train_loader:
            x_n = x_n.to(device)
            x_o = x_o.to(device)
            optimizer.zero_grad()
            output = model(x_n)
            loss = criterion(output, x_o)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        total_train_loss.append(np.mean(train_losses))

        if __name__ == "__main__":
            with torch.no_grad():
                test_losses = []
                for x_n, x_o in test_loader:
                    x_n = x_n.to(device)
                    x_o = x_o.to(device)
                    test_output = model(x_n)
                    test_loss = criterion(test_output, x_o)
                    test_losses.append(test_loss.item())
            total_test_loss.append(np.mean(test_losses))
            # print(
            #     f"Epoch: {e + 1}/{epochs}...",
            #     f"Train Loss: {np.mean(train_losses)}... ",
            #     f"Test Loss: {np.mean(test_losses)}...",
            # )

    if __name__ == "__main__":
        import matplotlib.pyplot as plt

        plt.plot(total_train_loss, label="Training loss")
        plt.plot(total_test_loss, label="Testing loss")
        plt.legend(frameon=False)
        plt.show()

    return total_train_loss[-1]


def _AE(df: pd.DataFrame, components: int = 200, criterion=nn.MSELoss()):
    batch_size = 8
    epochs = 500
    learning_rate = 1e-3
    original_dim = df.shape[1]

    x_train, x_train_noisy, x_test, x_test_noisy = prepare_data(df)

    train_dataset = Dataset(x_train_noisy, x_train)
    test_dataset = Dataset(x_test_noisy, x_test)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )

    model = DeepMS(original_dim, components).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    loss = train_model(epochs, model, optimizer, criterion, train_loader, test_loader)

    # get all latents
    latents = (
        model.encode(torch.tensor(df.values, dtype=torch.float32).to(device))
        .cpu()
        .detach()
        .numpy()
    )

    # get all weights
    weights = (
        [x.weight.data for i, x in enumerate(model.encoder.modules()) if i == 1][0]
        .cpu()
        .detach()
        .numpy()
    )

    return latents, weights, loss


def mseAE(df: pd.DataFrame, components: int = 200):
    return _AE(df, components, nn.MSELoss())


def klAE(df: pd.DataFrame, components: int = 200):
    return _AE(df, components, nn.KLDivLoss())


if __name__ == "__main__":
    df = pd.read_csv("eval/datasets/.wgs/WGS_PCAWG.96.txt", sep="\t", index_col=0)
    sig, weights = mseAE(df, 200)
    print(sig.shape)
    print(weights.shape)
