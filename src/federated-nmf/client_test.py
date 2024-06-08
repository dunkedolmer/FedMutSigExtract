import os
import sys
import shutil
import pandas as pd
import numpy as np
import flwr as fl
import argparse
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.decomposition import NMF
from cluster import mk_cluster
sys.path.append(os.path.dirname(__file__) + "/../tools")
from pathmanager import PathManager
from collections import OrderedDict

sys.path.append(os.path.dirname(__file__) + "/../tools")
from pathmanager import PathManager

pathmanager: PathManager = PathManager() 

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
   
def nmf(df, components):
    model = NMF(n_components=components, init="random", random_state=0)
    W = model.fit_transform(df)
    H = model.components_
    return W, H, model.reconstruction_err_


def klnmf(df, components):
    model = NMF(
        n_components=components,
        init="random",
        random_state=0,
        beta_loss="kullback-leibler",
        solver="mu",
    )
    W = model.fit_transform(df)
    H = model.components_
    return W, H, model.reconstruction_err_
   
def main():
    parser = argparse.ArgumentParser(description='FedNMFClient')
    parser.add_argument('partition', choices=['1', '2', '3'], help='Choose a partition')
    args = parser.parse_args()
    file = f"file_{args.partition}.txt"
    
    # Get the current directory of the code
    code_dir = os.path.abspath(os.path.dirname(__file__))

    # Navigate up two levels and then descend into the data directory
    data_dir = os.path.abspath(os.path.join(code_dir, '..', '..', 'data', 'external', 'sample8', 'output_folder'))

    # Now you can access files in the data directory
    file_path = os.path.join(data_dir, file)

    # Prepare data
    df = pd.read_csv(file_path, sep='\t', index_col=0, header=None, skiprows=[0]) # Shape: (96,167)
    
    # task = ak_cluster(dataset_path, mseAE, latents=200)
    task = mk_cluster(file_path, nmf)
    
    class FedNMFClient(fl.client.NumPyClient):
        def __init__(self, A, components):
            self.A = A 
            self.components = components
            self.model = NMF(n_components=components, init='random', random_state=0)
            self.W = np.empty((0,0)) # Initial W matrix
            self.H = np.empty((self.components,self.A.shape[1])) # Initial H matrix
            print("FedNMFClient initiated...")
            
        def get_parameters(self, config):
            print(f"[Client x.x] get_parameters")
            return [self.W] # -> List[List[W matrices]]
                    
        def fit(self, parameters_prev, config):
            print(f"[Client x.x] fit")

            parameters_next = self.get_parameters(config={}), len(self.A), {}
            # Return type -> Tuple[NDArrays, int, Dict[str, Scalar]]

            return parameters_next
        
        def extract_mutational_signatures(self, dataset):
            print("Extracting mutational signatures ...")
            task.run()

        def evaluate(self, parameters, config):
            return self.model.reconstruction_err_, len(self.A), {}

    # Initiate client
    client = FedNMFClient(df, components=5)
    
    # Start client
    fl.client.start_client(
        server_address="127.0.0.1:8080", client=client.to_client())

if __name__ == "__main__":
    main()