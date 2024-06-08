import os
import pandas as pd
import random

def split_by_columns(source_file, num_files):
    # Read the source file
    # df = pd.read_csv(source_file, sep='\t', index_col=0)
    # myPath = "C:\Users\Frederik\Documents\Repositories\p10-federated-learning\data\external\sample8\dataset.txt"
    df = pd.read_csv(source_file, sep=',', index_col=0)
    df = df.iloc[:, 2:]

    # Check if the number of columns is divisible by num_files
    num_columns = len(df.columns)
    if num_columns % num_files == 0:
        columns_per_chunk = num_columns // num_files
    else:
        print("Number of columns is not divisible by num_files. Distributing extra columns to the first one or two datasets.")
        extra_columns = num_columns % num_files
        columns_per_chunk = (num_columns + extra_columns) // num_files
    
    # Create a directory to store the split files if it doesn't exist
    output_dir = 'data/external/sample8/output_folder'
    os.makedirs(output_dir, exist_ok=True)
    
    # Write each chunk to a separate file
    for i in range(num_files):
        start_idx = i * columns_per_chunk
        end_idx = min((i + 1) * columns_per_chunk, num_columns)  # Ensure end index does not exceed num_columns
        output_file = os.path.join(output_dir, f'file_{i+1}.txt')
        df.iloc[:, start_idx:end_idx].to_csv(output_file, sep='\t')
        print(f'Chunk {i+1} saved to {output_file}')
        
    print('Splitting complete.')

def split_by_columns_old(source_file, num_files):
    # Read the source file
    df = pd.read_csv(source_file, sep='\t', index_col=0)
    
    # Check if the number of columns is divisible by num_files
    num_columns = len(df.columns)
    if num_columns % num_files == 0:
        columns_per_chunk = num_columns // num_files
    else:
        print("Number of columns is not divisible by num_files. Distributing extra columns to the first one or two datasets.")
        extra_columns = num_columns % num_files
        columns_per_chunk = (num_columns + extra_columns) // num_files
    
    # Create a directory to store the split files if it doesn't exist
    output_dir = 'data/external/sample8/output_folder'
    os.makedirs(output_dir, exist_ok=True)
    
    # Write each chunk to a separate file
    for i in range(num_files):
        start_idx = i * columns_per_chunk
        end_idx = min((i + 1) * columns_per_chunk, num_columns)  # Ensure end index does not exceed num_columns
        output_file = os.path.join(output_dir, f'file_{i+1}.txt')
        df.iloc[:, start_idx:end_idx].to_csv(output_file, sep='\t')
        print(f'Chunk {i+1} saved to {output_file}')

def split_by_columns_percentage(source_file, num_files, random_numbers):
    # Read the source file
    df = pd.read_csv(source_file, sep='\t', index_col=0)

    # Calculate the number of columns for each chunk based on the random numbers
    columns_per_chunk = random_numbers

    # Create a directory to store the split files if it doesn't exist
    output_dir = 'data/external/sample8/output_folder'
    os.makedirs(output_dir, exist_ok=True)

    # Write each chunk to a separate file
    for i, num_cols in enumerate(columns_per_chunk):
        start_idx = sum(columns_per_chunk[:i])  # Calculate start index for each chunk
        end_idx = start_idx + int(num_cols)  # Calculate end index
        output_file = os.path.join(output_dir, f'file_{i+1}.txt')
        df.iloc[:, start_idx:end_idx].to_csv(output_file, sep='\t')
        print(f'Chunk {i+1} saved to {output_file}')

    print('Splitting complete.')

def generate_random_numbers(total, num_count, lower_bound, upper_bound):
    if num_count <= 0:
        raise ValueError("Number count should be greater than zero.")
    if lower_bound > upper_bound:
        raise ValueError("Lower bound cannot be greater than upper bound.")

    # Create a list to store the random numbers
    random_numbers = []

    # Generate random numbers within the specified range and add them to the list
    for _ in range(num_count - 1):
        random_num = random.randint(max(lower_bound, 0), min(upper_bound, total))
        random_numbers.append(random_num)
        total -= random_num

    # Generate the last random number within the remaining total and specified range
    last_random_num = random.randint(max(lower_bound, 0), min(upper_bound, total))
    random_numbers.append(last_random_num)

    return random_numbers


def add_first_column(source_file, dest_dir):
    # Ensure that the output directory exists
    os.makedirs(dest_dir, exist_ok=True)
    
    # Read the source file to get the first column
    first_column = pd.read_csv(source_file, sep='\t', usecols=[0])
    
    # Iterate over files in the destination directory
    for filename in os.listdir(dest_dir):
        file_path = os.path.join(dest_dir, filename)
        
        # Read the file
        df = pd.read_csv(file_path, sep='\t', index_col=0)
        
        # Save the modified DataFrame back to the file without writing index and without extra tabs
        df.to_csv(file_path, sep='\t', index=True)


if __name__ == "__main__":
    source_file = 'WGS_PCAWG.96.csv'
    num_files = 1
    # total_columns = len(pd.read_csv(source_file, sep='\t', index_col=0).columns)
    # random_numbers = generate_random_numbers(total_columns, num_files, 1, 100)
    
    split_by_columns(source_file, num_files)