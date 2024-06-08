# experiments.py
import os
import sys

# Add the path to the centralized-setting/eval directory
centralized_setting_eval_path = os.path.join(os.path.dirname(__file__), '..', 'centralized-setting', 'eval')
sys.path.append(os.path.abspath(centralized_setting_eval_path))

from main import centralized_pipeline

def run_experiments():
    print("Running experiments ...")
    # Run the centralized pipeline 
    centralized_pipeline()
    
    # Produce results 
        # Extract signatures (calculate accuracy)
        # Compute efficiency (execution time)
        # Save all to file (in root/results)
    
        
    # Run the federated pipeline (even split)
        # NMF (MSE), NMF (KL), AE (MSE), and AE (KL)
    # Produce results
        # Extract signatures (calculate accuracy)
        # Compute efficiency (execution time)
        # Compute the communication overhead
        # Save all to file (in root/results)
    
    # Run the federated pipeline (uneven split)
        # NMF (MSE), NMF (KL), AE (MSE), and AE (KL)
    # Produce results
        # Extract signatures (calculate accuracy)
        # Compute efficiency (execution time)
        # Compute the communication overhead
        # Save all to file (in root/results)
    
    # Run the federated pipeline (uneven and weighted split)
        # NMF (MSE), NMF (KL), AE (MSE), and AE (KL)
    # Produce results
        # Extract signatures (calculate accuracy)
        # Compute efficiency (execution time)
        # Compute the communication overhead
        # Save all to file (in root/results)
    
    # Compare the results 
            
if __name__ == '__main__':
    run_experiments()