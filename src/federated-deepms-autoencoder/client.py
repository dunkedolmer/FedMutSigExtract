import os
import sys
import torch
import time
import argparse
import shutil
import numpy as np
import pandas as pd
import flwr as fl
from collections import OrderedDict
from cluster import ak_cluster
from models import DeepMSAutoencoder
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from method_evaluation import MethodEvaluator
sys.path.append(os.path.dirname(__file__) + "/../tools")
from pathmanager import PathManager

pathmanager: PathManager = PathManager() 
evaluator = MethodEvaluator()

current_file_path = os.path.dirname(__file__) + "/.." + "/.."
parent_folder = os.path.dirname(__file__) + "/.."

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def prepare_data(df: pd.DataFrame):
    test_set_percent = 0.1
    noise_factor = 0.0
    df = df.div(df.sum(axis=1), axis=0)
    
    split_index =int(len(df) * test_set_percent)
    
    # x_test = df.sample(frac=test_set_percent)
    x_test = df.iloc[:split_index]
    # x_train = df.drop(x_test.index)
    x_train = df.iloc[split_index:]
    x_train_noisy = x_train + noise_factor * np.random.normal(
        loc=0.0, scale=1.0, size=x_train.shape
    )
    x_test_noisy = x_test + noise_factor * np.random.normal(
        loc=0.0, scale=1.0, size=x_test.shape
    )
    x_train_noisy = np.clip(x_train_noisy, 0.0, 1.0)
    x_test_noisy = np.clip(x_test_noisy, 0.0, 1.0)

    return x_train, x_train_noisy, x_test, x_test_noisy

def load_data(x_train, x_train_noisy, x_test, x_test_noisy, batch_size=32):
    # TODO: Min/max scaling
    
    # Convert Pandas DataFrames to NumPy arrays
    x_train = x_train.values
    x_train_noisy = x_train_noisy.values
    x_test = x_test.values
    x_test_noisy = x_test_noisy.values
    
    # Convert NumPy arrays to PyTorch tensors
    x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
    x_train_noisy_tensor = torch.tensor(x_train_noisy, dtype=torch.float32)
    x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
    x_test_noisy_tensor = torch.tensor(x_test_noisy, dtype=torch.float32)

    # PyTorch Dataset objects
    train_dataset = TensorDataset(x_train_tensor, x_train_noisy_tensor)
    test_dataset = TensorDataset(x_test_tensor, x_test_noisy_tensor)

    # PyTorch DataLoader objects
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

# Create dataset from pandas dataframe
class Dataset(torch.utils.data.Dataset):
    def __init__(self, df1, df2):
        self.x1 = torch.tensor(df1.values, dtype=torch.float32)
        self.x2 = torch.tensor(df2.values, dtype=torch.float32)

    def __len__(self):
        return len(self.x1)

    def __getitem__(self, idx):
        return self.x1[idx], self.x2[idx]

def train(model, trainloader, testloader, epochs, criterion):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    total_train_loss = []
    total_test_loss = []    
    for _ in range(epochs):
        train_losses = []
        for input_data, target_data in trainloader:
            input_data = input_data.to(DEVICE)
            target_data = target_data.to(DEVICE)
            optimizer.zero_grad()
            reconstructed_output = model(input_data)
            loss = criterion(reconstructed_output, target_data)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        total_train_loss.append(np.mean(train_losses))
        
        if __name__ == "__main__":
            with torch.no_grad():
                test_losses = []
                for x_n, x_o in testloader:
                    x_n = x_n.to(DEVICE)
                    x_o = x_o.to(DEVICE)
                    test_output = model(x_n)
                    test_loss = criterion(test_output, x_o)
                    test_losses.append(test_loss.item())
            total_test_loss.append(np.mean(test_losses))
    if __name__ == "__main__":
        import matplotlib.pyplot as plt

        plt.plot(total_train_loss, label="Training loss")
        plt.plot(total_test_loss, label="Testing loss")
        plt.legend(frameon=False)
        plt.show()

    plt.plot(total_train_loss, label="Training loss")
    plt.plot(total_test_loss, label="Testing loss")
    plt.legend(frameon=False)
    plt.show()

    return total_train_loss[-1]


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
    for _ in range(epochs):
        train_losses = []
        for x_n, x_o in train_loader:
            x_n = x_n.to(DEVICE)
            x_o = x_o.to(DEVICE)
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
                    x_n = x_n.to(DEVICE)
                    x_o = x_o.to(DEVICE)
                    test_output = model(x_n)
                    test_loss = criterion(test_output, x_o)
                    test_losses.append(test_loss.item())
            total_test_loss.append(np.mean(test_losses))
    if __name__ == "__main__":
        import matplotlib.pyplot as plt

        plt.plot(total_train_loss, label="Training loss")
        plt.plot(total_test_loss, label="Testing loss")
        plt.legend(frameon=False)
        plt.show()

    plt.plot(total_train_loss, label="Training loss")
    plt.plot(total_test_loss, label="Testing loss")
    plt.legend(frameon=False)
    plt.show()

    return total_train_loss[-1]

def _AE(df: pd.DataFrame, components: int = 200, criterion=nn.KLDivLoss(), model=None):
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

    # Check if we are passed an existing model
    if model is None:
        model = DeepMSAutoencoder(original_dim, components).to(DEVICE)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    loss = train_model(epochs, model, optimizer, criterion, train_loader, test_loader)
    
    # get all latents
    latents = (
        model.encode(torch.tensor(df.values, dtype=torch.float32).to(DEVICE))
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

    return latents, weights, loss, model

def mseAE(df: pd.DataFrame, components: int = 200, modFel=None):
    return _AE(df, components, nn.MSELoss(), model)


def klAE(df: pd.DataFrame, components: int = 200, model=None):
    return _AE(df, components, nn.KLDivLoss(), model)

def save_output(method) -> None:
    '''Saves the output of a method based on a dataset in the results directory'''
    # Save output in results folder
    result_folder = f"{current_file_path}/results/federated/{method}"
    # if not os.path.exists(pathmanager.results_centralized() + f"/{method}"):
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
        print(f"Made the directory: {result_folder}")

    shutil.copy(
        parent_folder + "/datasets/nmf_output/aux_loss.png",
        f"{result_folder}/aux_loss.png"
        # dataset["folder"] + f"/results/{method}/aux_loss.png",
    )
    shutil.copy(
        parent_folder + "/datasets//nmf_output/signatures.tsv",
        f"{result_folder}/signatures.tsv"
        # dataset["folder"] + f"/results/{method}/signatures.tsv",
    )
    shutil.copy(
        parent_folder
        + "/datasets/nmf_output/output_weights/Assignment_Solution/Activities/Assignment_Solution_Activities.txt",
        f"{result_folder}/weights.txt"
        # dataset["folder"] + f"/results/{method}/weights.txt",
    )
    
def evaluate_synthetic(partition):
    print(f"Received args.partition: {partition}")
    result = evaluator.evaluate(
        parent_folder + "/datasets/nmf_output/signatures.tsv",
        parent_folder + "/datasets/nmf_output/output_weights/Assignment_Solution/Activities/Assignment_Solution_Activities.txt",
        current_file_path + "/data" + "/external" + "/sample8" + "/sigMatrix.csv",
        current_file_path + "/data" + "/external" + "/sample8" + "/weights" + f"/weights_{partition}.csv")
        
    return result

def evaluate_wgs(partition):
    print(f"Received args.partition: {partition}")
    return evaluator.COSMICevaluate(
            parent_folder + "/datasets/nmf_output/signatures.tsv",
            "GRCh37",
        )

def save_results(results, methods):
    columns = [
        "found",
        ">0.8",
        ">0.95",
        "best>0.95",
        "best>0.99",
        "match",
        "mse",
        "mae",
        "rmse",
        ]
    
    print(f"results: {results}")
    print(f"columns (len): {len(columns)}")
    print(f"index: {methods}")

    return pd.DataFrame(
        results,
        columns=columns,
        index=methods,
    )


def main():
    # Parser setup for choosing partition to run AE on
    parser = argparse.ArgumentParser(description='FedNMFClient')
    parser.add_argument('partition', choices=['1', '2', '3'], help='Choose a partition')
    args = parser.parse_args()
    file = f"file_{args.partition}.txt"
    
    # Prepare file path for dataset
    code_dir = os.path.abspath(os.path.dirname(__file__))
    # data_dir = os.path.abspath(os.path.join(code_dir, '..', '..', 'data', 'external', 'sample8', 'uneven'))
    # file_path = os.path.join(data_dir, file)
    wgs_path = os.path.abspath(os.path.join(code_dir, '..', '..', 'data', 'external', 'wgs', 'data'))
    wgs_file = f"file_{args.partition}.txt"
    wgs_dataset_path = os.path.join(wgs_path, wgs_file)
    print(f"Succesfully read file: {wgs_dataset_path}")
    # weights_file = current_file_path + "/data" + "/external" + "/sample8" + "/weights" + f"/weights_{args.partition}.csv"

    
    # Read dataset from file
    # print(f"file_path: {file_path}")
    # df = pd.read_csv(file_path, sep='\t', index_col=0, header=None, skiprows=[0]) # Shape: (96,167)
    # print(f"df.shape: {df.shape}")
    df_wgs = pd.read_csv(wgs_dataset_path, sep='\t', index_col=0, header=None, skiprows=[0]) # Shape: (96,2781)
    print(f"df_wgs.shape: {df_wgs.shape}")
    print(df_wgs.iloc[:,0])
    # model = DeepMSAutoencoder(df.shape[0], df.shape[1])
    # model = model.to(DEVICE)
    # task = ak_cluster(file_path, mseAE, latents=200)

    # Autoencoder setup
    # Hyperparameters
    batch_size = 8
    epochs = 100 # used to be 500
    learning_rate = 1e-3
    components = 200

    # x_train, x_train_noisy, x_test, x_test_noisy = prepare_data(df_wgs)
    # trainloader, testloader = load_data(x_train, x_train_noisy, x_test, x_test_noisy)
    # train_dataset = Dataset(x_train_noisy, x_train)
    # test_dataset = Dataset(x_test_noisy, x_test)
    # train_loader = torch.utils.data.DataLoader(
    #     train_dataset, batch_size=batch_size, shuffle=True
    # )
    # test_loader = torch.utils.data.DataLoader(
    #     test_dataset, batch_size=batch_size, shuffle=False
    # )
    model = DeepMSAutoencoder(df_wgs.shape[1], components).to(DEVICE)

    class FedDeepMSClient(fl.client.NumPyClient):
        def __init__(self, dataset: pd.DataFrame):
            self.round = 1 # Initial round 
            # self.model = DeepMSAutoencoder(dataset.shape[1], 200).to(DEVICE)
            self.latents = []
            self.weights = []
            self.loss = float(0.0) # Initial loss    
            self.t_start = None
            self.t_end = None
                    
        def get_parameters(self, config):
            print(f"[Client x.x] get_parameters")
            parameters = [val.cpu().numpy() for _, val in model.state_dict().items()]  
            return parameters

        def set_parameters(self, parameters):
            if isinstance(parameters, list):  # Receiving global model from server
                print("Updating model parameters based on global model")
                params_dict = zip(model.state_dict().keys(), parameters)
                state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
                self._load_state_dict_with_mismatch(model, state_dict)
            elif isinstance(parameters, OrderedDict):  # Receiving locally updated model
                self._load_state_dict_with_mismatch(model, parameters)
                print("Updating model parameters based on locally trained model")
            else:
                raise ValueError(f"Unsupported type for parameters: {type(parameters)}")

        def _load_state_dict_with_mismatch(self, model, state_dict):
            model_dict = model.state_dict()
            # Filter out unnecessary keys and mismatched shapes
            state_dict = {k: v for k, v in state_dict.items() if k in model_dict and model_dict[k].shape == v.shape}
            # Update the existing state dict
            model_dict.update(state_dict)
            model.load_state_dict(model_dict)
            
        def fit(self, parameters, config):
            print(f"[Client x.x.] fit")
            self.set_parameters(parameters) # Update local parameters received from server
            self.extract_mutational_signatures()
            self.round += 1 # Increment the round
            return self.get_parameters(config={}), 741, {}
            
        def extract_mutational_signatures(self):
            print("Extracting mutational signatures ...")
            execution_times = [] # Store the execution time of each method (for experiments)
            
            if (self.t_start == None):
                self.t_start = time.time()
                print(f"Started execution time!")
            task = ak_cluster(model, wgs_dataset_path, klAE, 10, 200)
            task.run()
            
            # Get the parameters
            trained_parameters = task.get_model_parameters()
            self.set_parameters(trained_parameters)
            self.loss = task.loss # the loss from the final training round of the autoencoder
            
            # Find the execution time
            self.t_end = time.time() # Save end time
            if (self.round == 3):
                t_total = self.t_end - self.t_start
                print(f"Method AE_kl took {t_total:.2f} seconds")
            
            save_output("AE_ak_mse")
            # result = evaluate_synthetic(args.partition)
            result = evaluate_wgs(args.partition)
            print(f"Results: {result}")
                    
        def evaluate(self, parameters, config):
            print(f"[Client x.x] evaluate")
            self.set_parameters(parameters)
            return self.loss, 185, {}
            # return task.loss, len(testloader), {}            

    # Initiate client
    client = FedDeepMSClient(df_wgs) 
    
    # # Start Flower client
    fl.client.start_client(server_address="127.0.0.1:8080",  client=client.to_client())

if __name__ == "__main__":
    main()
