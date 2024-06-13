import os
import shutil

def create_identical_files(base_folder):
    for component in range(2, 101):
        loss_folder = os.path.join(base_folder, f'{component}_components', 'Loss')
        w_matrix_folder = os.path.join(base_folder, f'{component}_components', 'W_matrix')

        print(f"Checking folders for {component} components...")

        # Ensure the component directories exist
        if not os.path.exists(loss_folder):
            print(f"Missing folder: {loss_folder}")
            continue
        if not os.path.exists(w_matrix_folder):
            print(f"Missing folder: {w_matrix_folder}")
            continue

        # Source files from each component's folder
        source_loss_file = os.path.join(loss_folder, 'Loss2.txt')
        source_w_matrix_file = os.path.join(w_matrix_folder, 'W_matrix2.txt')

        # Check if source files exist
        if not os.path.exists(source_loss_file):
            print(f"Missing source file: {source_loss_file}")
            continue
        if not os.path.exists(source_w_matrix_file):
            print(f"Missing source file: {source_w_matrix_file}")
            continue

        # Copy and rename files for Loss and W_matrix from 2 to 10
        for i in range(2, 11):
            dest_loss_file = os.path.join(loss_folder, f'Loss{i}.txt')
            dest_w_matrix_file = os.path.join(w_matrix_folder, f'W_matrix{i}.txt')

            try:
                shutil.copyfile(source_loss_file, dest_loss_file)
                shutil.copyfile(source_w_matrix_file, dest_w_matrix_file)
                print(f"Copied Loss{i}.txt and W_matrix{i}.txt to {component} components.")
            except Exception as e:
                print(f"Error copying files to {component} components: {e}")

# Specify the base folder containing the matrix_files directory

# Specify the base folder containing the 
base_folder = 'src/federated-nmf/matrix_files'
create_identical_files(base_folder)