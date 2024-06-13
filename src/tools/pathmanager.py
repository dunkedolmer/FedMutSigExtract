# /root/src/tools/pathmanager.py
import os
import sys

class PathManager:
    """
    A helper module for returning paths.

    This module provides methods for managing paths within the project.
    """
    def __init__(self):
        script_dir = os.path.dirname(os.path.realpath(__file__))
        self.root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')) # Returns "\p10-federated-learning"

    # Project path
    def root(self):
        return self.root_path

    # Data directory 
    def data(self):
        return os.path.join(self.root_path, 'data')
    
    def data_intermediate(self):
        return os.path.join(self.data(), 'intermediate')
    
    def data_external(self):
        return os.path.join(self.data(), 'external')
    
    def data_synthetic(self):
        return os.path.join(self.data(), 'synthetic')
    
    # Outout directory
    def output(self):
        return os.path.join(self.root_path, 'output')
    
    def output_results(self):
        return os.path.join(self.output(), 'results')
    
    def output_nmf_output(self):
        return os.path.join(self.output(), 'nmf_output')
    
    # results directory
    def results(self):
        return os.path.join(self.root_path, 'results')
    
    def results_centralized(self):
        return os.path.join(self.results(), 'centralized')
    
    def results_federated(self):
        return os.path.join(self.results(), 'federated')

    # src directory
    def src(self):
        return (os.path.join(self.root_path, 'src'))
    
    def src_AE(self):
        return os.path.join(self.src(), 'AE')

    def src_eval(self):
        return os.path.join(self.src(), 'eval')
    
    # sigGen directory
    def src_sigGen(self):
        return os.path.join(self.src(), 'sigGen')
    
    # tools directory
    def src_tools(self):
        return os.path.join(self.src(), 'tools')

    # To see the contents of a directory (e.g., for debugging)
    def print_directory_contents(self, dir):
        for item in os.listdir(dir):
            print(item)

if __name__ == "__main__":
    print("The PathManager module was called")
    pathmanager = PathManager()

    # Printing paths
    print("Root Path:", pathmanager.root())
    print("Data Directory:", pathmanager.data())
    print("Intermediate Data Directory:", pathmanager.data_intermediate())
    print("External Data Directory:", pathmanager.data_external())
    print("Synthetic Data Directory:", pathmanager.data_synthetic())
    print("Output Directory:", pathmanager.output())
    print("Output Results Directory:", pathmanager.output_results())
    print("Output NMF_Output:", pathmanager.output_nmf_output())
    print("Evaluation AE Directory:", pathmanager.src_AE())
    print("Evaluation Source Directory:", pathmanager.src_eval())
    print("Signal Generator Source Directory:", pathmanager.src_sigGen())
    print("Tools Source Directory:", pathmanager.src_tools())