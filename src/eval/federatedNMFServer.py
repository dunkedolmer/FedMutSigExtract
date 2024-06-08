import os
import sys
import shutil
import pandas as pd
import numpy as np
from sklearn.decomposition import NMF
from federatedNMFClient import FederatedNMFClient

sys.path.append(os.path.dirname(__file__) + "/../tools")
from pathmanager import PathManager

pathmanager: PathManager = PathManager() 
nmf = FederatedNMFClient()

class FederatedNMFClient:
    # Old version used for local testing
    def calculate_average_W(self, results):
        # Extract H matrices from results
        W_matrices = [result[2] for result in results]  # Assuming H matrices are at index 3 in each result

        # Add all H matrices element-wise to get the total sum
        total_sum_H = np.sum(W_matrices, axis=0)  # Sum along axis 0 (rows)

        # Calculate the average H matrix
        num_files = len(results)
        average_W = total_sum_H / num_files

        return average_W
    
    def calculate_average_W(self, w_matrices):
        # Add all H matrices element-wise to get the total sum
        total_sum_w = np.sum(w_matrices, axis=0)  # Sum along axis 0 (rows)

        # Calculate the average H matrix
        num_files = len(w_matrices)
        average_w = total_sum_w / num_files

        return average_w
    
if __name__ == "__main__":
    results = []
    folder_path = os.path.abspath(pathmanager.data_external() + "/sample8/output_folder/")
    file_name1 = os.path.abspath(pathmanager.data_external() + "/sample8/output_folder/file_1.txt")
    file_name2 = os.path.abspath(pathmanager.data_external() + "/sample8/output_folder/file_2.txt")
    components = 5
    # Append w_matrix to an array and find the average
    # Have to run all clients before finding average
    results.append(nmf.nonnegative_matrix_factorization(file_name1, components))
    results.append(nmf.nonnegative_matrix_factorization(file_name2, components))
    print(len(results))
