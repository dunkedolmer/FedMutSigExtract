import argparse
import flwr as fl
import tensorflow as tf
from sklearn.decomposition import NMF
import numpy as np

class AutoencoderClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return model.get_weights()
    
    # def fit(self, parameters, config):
        

class CifarClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return model.get_weights()
    
    def fit(self, parameters, config):
        model.set_weights(parameters)
        model.fit(x_train, y_train, epochs=1, batch_size=32)
        return model.get_weights(), len(x_train), {}
            
    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(x_test, y_test)
        return loss, len(x_test), {"accuracy": accuracy}
            
class NMFClient(fl.client.NumPyClient):
    def __init__(self, x_train, x_test):
        self.x_train = x_train
        self.x_test = x_test
        
    def get_parameters(self, config):
        return [] # Because NMF does not have trainable parameters
        
    def fit(self, parameters, config):
        model = NMF(n_components=2, init="random", random_state=0)
        W = model.fit_transform(self.x_train)
        H = model.components_
        print("NMFClient.fit() called")
        result = (W, H), len(self.x_train), {}
        print("Type of result: ", type(result))
        return result            
    
    def evaluate(self, parameters, config):
        if not parameters:
            return float('inf'), len(self.x_test), {}
        else:
            W, H, _ = parameters
            reconstructed = np.dot(W, H)
            loss = np.mean((reconstructed - self.x_test) ** 2)
            return loss, len(self.x_test), {}
            
def load_data():
    # Generate dummy data for testing purposes
    num_samples = 1000
    num_features = 50
    train_ratio = 0.8
    
    # Generate random training examples and test examples
    x_train = np.random.rand(int(num_samples * train_ratio), num_features)
    x_test = np.random.rand(int(num_samples * (1 - train_ratio)), num_features)
    
    return x_train, x_test
            
# def nmf(df, components):
#     model = NMF(n_components=components, init="random", random_state=0)
#     W = model.fit_transform(df)
#     H = model.components_
#     return W, H, model.reconstruction_err_

def main():
    
    ## 1. Parse config
    parser = argparse.ArgumentParser(description='Client for federated learning with Flower')
    parser.add_argument('mode', choices=['nmf', 'autoencoder'], help='Operation mode (NMF or autoencoder)')
    args = parser.parse_args()
    
    ## 2. Define clients
        # TODO: Implement function for generating a client dynamically
    
    ## 4. Define strategy
        # Use FedAvg
    
    ## 5. Distribute work (e.g., NMF or AE) to clients

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Client for federated learning with Flower')
    parser.add_argument('mode', choices=['nmf', 'autoencoder', 'test'], help='Operation mode (NMF or autoencoder)')
    args = parser.parse_args()
    
    x_train, x_test = load_data()
    
    if args.mode == 'nmf':
        print("Running NMF ...")
        fl.client.start_client(
            server_address="localhost:8080",
            client=NMFClient(x_train, x_test).to_client())
                
    elif args.mode == 'autoencoder':
        print("Running autoencoder ...")
        # Run autoencoder pipeline for a client
        
    elif args.mode == 'test':
        # Note: this pipeline is only for testing purposes
        print('INFO: Running test pipeline')
        
        # Define model
        print("Defining model ...")
        model = tf.keras.applications.MobileNetV2(
            (32,32,3),
            classes=10,
            weights=None
        )
        
        model.compile("adam", 
              "sparse_categorical_crossentropy",
              metrics=['accuracy']
              )
        
        print("Loading data ...")
        (x_train, y_train),  (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        
        print("Initializing client ...")    
        fl.client.start_client(
            server_address="127.0.0.1:8080",
            client=CifarClient().to_client()
        )
        
    else:
        print("Invalid argument. Did you mean NMF or autoencoder?")