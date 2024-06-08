import os
import pandas as pd

current_file_path = os.path.dirname(__file__) + "/.." + "/.."

def fix_wgs_dataset():
    code_dir = os.path.abspath(os.path.dirname(__file__))
    wgs_path = os.path.abspath(os.path.join(code_dir, '..', '..', 'data', 'external', 'wgs'))
    wgs_file = "WGS_PCAWG.96.csv"
    wgs_dataset_path = os.path.join(wgs_path, wgs_file)

    # Read dataset from file
    df_wgs = pd.read_csv(wgs_dataset_path, sep=',') # Shape: (96,2781)
    if not df_wgs.empty:
        print("Succesfully loaded the WGS dataset!")
        print(f"df_wgs.shape: {df_wgs.shape}")
        
        # Drop the first column after index column
        df_wgs = df_wgs.drop(columns="Trinucleotide", axis=1)
        print("Dropping first column ...")
        
        df_wgs.to_csv(wgs_file, index=False)
        print("Saving the updated file!")
    
    else:
        print("Failed to load the WGS dataset...")    
    

def print_current_file_path():
    """
    Prints the current file path.
    """
    current_file_path = os.path.abspath(__file__)
    print(f"The current file path is: {current_file_path}")

def save_output(method):
    if not os.path.exists(f"{current_file_path}/results/federated/{method}"):
        os.makedirs(f"{current_file_path}/results/federated/{method}")
        print(f"Made the directory: {current_file_path}/results/federated/{method}")

if __name__ == "__main__":
    fix_wgs_dataset()
    
    