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

class NMFClient: 
    def nonnegative_matrix_factorization(self, file_path, components):
        df = pd.read_csv(file_path, sep='\t', header=None, skiprows=[0])
        df = df.drop(df.columns[0], axis=1)
                       
        # Perform NMF on the dataset
        model = NMF(n_components=components, init="random", random_state=0)
        W = model.fit_transform(df)
        H = model.components_.T  # Transpose H to have components along columns
        reconstruction_err = model.reconstruction_err_

        return reconstruction_err, W, H, df
    
    # Only used for testing purposes
    def process_files_in_folder(self, folder_path, components):
        results = []
        file_list = os.listdir(folder_path)
        for file_name in file_list:
            if file_name.endswith('.txt'):
                file_path = os.path.join(folder_path, file_name)
                reconstruction_err, W, H, df = self.nonnegative_matrix_factorization(file_path, components)
                results.append((file_name, reconstruction_err, W, H, df))
        return results
    
    def old_calculate_average_W(self, results):
        # Extract H matrices from results
        W_matrices = [result[2] for result in results]  # Assuming H matrices are at index 3 in each result

        # Add all H matrices element-wise to get the total sum
        total_sum_H = np.sum(W_matrices, axis=0)  # Sum along axis 0 (rows)

        # Calculate the average H matrix
        num_files = len(results)
        average_W = total_sum_H / num_files

        return average_W 
   
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

            # Case of having the initial W matrix
            if self.W.size == 0: 
                print("self.W.size == 0")
                print(f"W matrix (shape) before: {self.W.shape}") 
                self.W = self.model.fit_transform(self.A) # Calculate a W matrix based on input matrix A if empty
                print(f"W matrix (shape) after: {self.W.shape}")
            else:
                print(f"self.H: {self.H}")
                print("Updating W matrix ...")
                previous_W = self.W
                print(f"previous_W (before): {previous_W}")
                print(f"self_W (before): {self.W}")
                self.model = NMF(n_components=self.components, init='custom', random_state=0)
                self.W = self.model.fit_transform(self.A, W=self.W, H=self.H) # Update W matrix based on the one received from parameters
                print(f"previous_W (after): {previous_W}")
                print(f"self_W (after): {self.W}")
                print("W matrix (equal): ", np.array_equal(previous_W, self.W))

            parameters_next = self.get_parameters(config={}), len(self.A), {}
            
            # Return type -> Tuple[NDArrays, int, Dict[str, Scalar]]
            print(f"Reconstruction loss: {self.model.reconstruction_err_}")
            print(f"parameters_next (type): {type(parameters_next)}")
            print(f"parameters_next[0]: {parameters_next[0]}")
            return parameters_next
        

        def evaluate(self, parameters, config):
            # return float(1.0), len(self.A), {}    
            return self.model.reconstruction_err_, len(self.A), {}

    # Initiate client
    client = FedNMFClient(df, components=5)
    
    # Start client
    fl.client.start_client(
        server_address="127.0.0.1:8080", client=client.to_client())

# if __name__ == "__main__":
#     client = FedNMFClient()  # Create an instance of FederatedNMFClient
#     #folder_path = os.path.abspath(pathmanager.data_external() + "/sample8/output_folder/")
#     file_name = os.path.abspath(pathmanager.data_external() + "/sample8/output_folder/file_1.txt")
#     components = 5 # Used for testing purposes
#     #results = client.process_files_in_folder(folder_path, components) # Used for testing purposes
#     #average_W_matrix = client.old_calculate_average_W(results) # Used for testing purposes
#     #output_file = "average_W_matrix.csv" # Used for testing purposes
#     #np.savetxt(output_file, average_W_matrix, fmt="%0.6f")
#     #print(f"Average H matrix saved to '{output_file}'.")
#     client.nonnegative_matrix_factorization(file_name, components)

if __name__ == "__main__":
    main()