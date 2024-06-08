import torch.nn as nn

class DeepMSAutoencoder(nn.Module):
    def __init__(self, original_dim, encoding_dim) -> None:
        super(DeepMSAutoencoder, self).__init__()
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