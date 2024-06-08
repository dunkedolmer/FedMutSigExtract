import os

os.environ["KERAS_BACKEND"] = "torch"
import keras
import pandas as pd
import numpy as np


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


def plot_loss(hist):
    import matplotlib.pyplot as plt

    plt.plot(hist.history["loss"], label="Training loss")
    plt.plot(hist.history["val_loss"], label="Validation loss")
    plt.legend()
    plt.show()


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


def mseAE(df: pd.DataFrame, components: int = 42):
    return _AE(df, components, keras.losses.MeanSquaredError())


def klAE(df: pd.DataFrame, components: int = 42):
    return _AE(df, components, keras.losses.KLDivergence())


if __name__ == "__main__":
    df = pd.read_csv("eval/datasets/.simple/dataset.txt", sep="\t", index_col=0)
    sig, weights = mseAE(df, 200)
    print(sig.shape)
    print(weights.shape)
