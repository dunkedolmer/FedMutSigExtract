import os
import sys
import shutil
import pandas as pd
import numpy as np
from sklearn.decomposition import NMF
sys.path.append(os.path.dirname(__file__) + "/../tools")
from pathmanager import PathManager

pathmanager: PathManager = PathManager() 

class FederatedNMFClient:
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
   



if __name__ == "__main__":
    client = FederatedNMFClient()  # Create an instance of FederatedNMFClient
    #folder_path = os.path.abspath(pathmanager.data_external() + "/sample8/output_folder/")
    file_name = os.path.abspath(pathmanager.data_external() + "/sample8/output_folder/file_1.txt")
    components = 5 # Used for testing purposes
    #results = client.process_files_in_folder(folder_path, components) # Used for testing purposes
    #average_W_matrix = client.old_calculate_average_W(results) # Used for testing purposes
    #output_file = "average_W_matrix.csv" # Used for testing purposes
    #np.savetxt(output_file, average_W_matrix, fmt="%0.6f")
    #print(f"Average H matrix saved to '{output_file}'.")
    client.nonnegative_matrix_factorization(file_name, components)