import os
import sys
import shutil
import pandas as pd
import numpy as np
import flwr as fl
import argparse
import torch
import time
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.decomposition import NMF
sys.path.append(os.path.dirname(__file__) + "/../tools")
from pathmanager import PathManager
from collections import OrderedDict

sys.path.append(os.path.dirname(__file__) + "/../tools")
from pathmanager import PathManager

pathmanager: PathManager = PathManager() 

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Generate original matrix A with random float values
def generate_synthetic_matrix(m, n):
    return np.random.rand(m,n)
   
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
    
    # Define the directory where you want to save the files
    output_dir = "matrix_files"
    os.makedirs(output_dir, exist_ok=True)  # Create the directory if it doesn't exist
    
    class FedNMFClient(fl.client.NumPyClient):
        def __init__(self, A, components):
            self.A = A 
            self.components = components
            self.model = NMF(n_components=components, init='random', random_state=0)
            self.W = np.empty((0,0)) # Initial W matrix
            self.W_test = []
            self.round = 1
            self.loss = 0
            self.t_start = None
            #self.H = np.empty((self.components,self.A.shape[1])) # Initial H matrix
            print("FedNMFClient initiated...")

        def klnmf(self, df, components):
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

        def klnmf_fed(self, df, components, W, H):
            model = NMF(
                n_components=components,
                init="random",
                random_state=0,
                beta_loss="kullback-leibler",
                solver="mu",
            )
            model = NMF(n_components=components, init='custom', random_state=0)
            W = model.fit_transform(df, W=W, H=self.H) # Update W matrix based on the one received from parameters
            H = model.components_  # Update H matrix after fitting
            return W, H, model.reconstruction_err_




        def nmf(self, df, components):
            model = NMF(n_components=components, init="random", random_state=0)
            W = model.fit_transform(df)
            H = model.components_
            return W, H, model.reconstruction_err_
        
        def nmf_fed(self, df, components, W, H):
            model = NMF(n_components=components, init='custom', random_state=0)
            W = model.fit_transform(df, W=W, H=self.H) # Update W matrix based on the one received from parameters
            H = model.components_  # Update H matrix after fitting
            return W, H, model.reconstruction_err_

        def get_parameters(self, config):
            print(f"[Client x.x] get_parameters")
            return [self.W] # -> List[List[W matrices]]
                    
        def fit(self, parameters_prev, config):
            print(f"[Client x.x] fit")

            # Case of having the initial W matrix
            if self.W.size == 0: 
                self.t_start = time.time()
                self.W, self.H, self.loss = self.nmf(df, 100) 
            else:
                self.W_test = parameters_prev[0]
                np.savetxt(os.path.join(output_dir, f"self.W_test{args.partition}.txt"), self.W_test)
                t_end = time.time()
                elapsed_time = t_end - self.t_start
                print(f"Time end: {elapsed_time} round{self.round}")
                self.round = self.round + 1            
                self.W, self.H, self.loss = self.nmf_fed(df, 100, parameters_prev[0], self.H)

            parameters_next = self.get_parameters(config={}), len(self.A), {}
            
            # Return type -> Tuple[NDArrays, int, Dict[str, Scalar]]
            return parameters_next
        
        def evaluate(self, parameters, config):
            return self.loss, len(self.A), {}

    # Initiate client
    client = FedNMFClient(df, components=10)
    
    # Start client
    fl.client.start_client(
        server_address="127.0.0.1:8080", client=client.to_client())

if __name__ == "__main__":
    main()