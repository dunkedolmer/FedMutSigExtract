from collections import OrderedDict

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from models import Net
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

import flwr as fl

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#Load data

#Dataloader for test and training set
#We might be able to use the prepare_data from our initial AE
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

#Next up we need our training and test functions
def _AE(df: pd.DataFrame, components: int = 42, loss=keras.losses.MeanSquaredError()):
    batch_size = 32
    epochs = 800
    learning_rate = 1e-3
    original_dim = df.shape[1]

    x_train, x_train_noisy, x_test, x_test_noisy = prepare_data(df)

    # input
    input_dim = keras.layers.Input(shape=(original_dim,))
    # encoder
    encoder = keras.layers.Dense(
        components, activation="relu", activity_regularizer=keras.regularizers.l1(1e-12)
    )(input_dim)
    # decoder
    decoded = keras.layers.Dense(original_dim, activation="softmax")(encoder)
    # autoencoder model
    autoencoder = keras.Model(inputs=input_dim, outputs=decoded)
    # encoder model
    encoder_model = keras.Model(inputs=input_dim, outputs=encoder)

    # compile autoencoder
    autoencoder.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=loss,
    )
    # training
    hist = autoencoder.fit(
        x_train_noisy,
        x_train,
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        validation_data=(x_test_noisy, x_test),
        verbose=0,
    )

    if __name__ == "__main__":
        plot_loss(hist)

    latents = encoder_model.predict(df, verbose=0)

    weights = [layer.get_weights() for layer in encoder_model.layers]

    weights = np.transpose(weights[1][0])

    return latents, weights


#Then we need some kind of test function that validates the network on the entire test set
#Here is an example of doing it with images taken from the flower github
#We need to redefine this function so it allow testing of our framewokr instead of the image based one
def test(net, testloader):
    """Validate the network on the entire test set."""
    total, loss = 0, 0.0
    with torch.no_grad():
        for data in testloader:
            images = data[0].to(DEVICE)
            recon_images, mu, logvar = net(images)
            recon_loss = F.mse_loss(recon_images, images)
            kld_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            loss += recon_loss + kld_loss
            total += len(images)
    return loss / total



#Then some main function that trains the model and test it against the other models
#This is so that we can run multiple clients at the same time with the server to mimic federated learning 