# server.py
import flwr as fl
import numpy as np
import os
import io
from typing import Callable, Dict, List, Optional, Tuple, Union
from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)

class NMFStrategy(fl.server.strategy.FedAvg):
    def __init__(self):
        super().__init__(),
        self.W_global = [] # Global model (W matrix) 
        self.H_global = [] # Global model (H matrix) 
        self.W_global_cut = [] # Global W matrix with first 5 elements (for debugging in console) 
    
    def aggregate_fit(self, rnd: int, results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]], failures: List[BaseException]) -> Tuple[fl.common.Parameters, dict]:
        """Aggregate fit results using average"""
        print(f"Round {rnd}")
        if not results:
            return None, {} # If we do not have anything to aggregate, return nothing

        W_matrices = [] # Store W matrices received from clients

        print(f"Length (results): {len(results)}")
        
        # Get and translate matrices
        for _, fit_res in results:
            parameters = fit_res.parameters
            if parameters:
                W_bytes = parameters.tensors[0]  # Byte string for W matrix
                W_buffer = io.BytesIO(W_bytes)
                W_matrix = np.load(W_buffer, allow_pickle=False)
                W_matrices.append(W_matrix) # Get each local W matrix in a list
             
        if not W_matrices:
            return None, {}
        
        # Aggregation
        W_avg = np.mean(W_matrices, axis=0) # Aggregate the results from local W matrices
        self.W_global = W_avg # Store the aggregated result
        self.W_global_cut = W_avg[:5]

        # Convert to Flower output    
        W_avg_buffer = io.BytesIO()
        np.save(W_avg_buffer, W_avg)
        W_avg_bytes = W_avg_buffer.getvalue()

        parameters_aggregated = fl.common.Parameters(tensors=[W_avg_bytes], tensor_type="numpy.ndarray")
        # print("parameters_aggregated ...")
        # print(parameters_aggregated)
        return parameters_aggregated, {}

    
strategy = NMFStrategy(
#    fraction_fit=1.0,
#    min_fit_clients=2,
#    min_available_clients=2,
)

myStrategy = fl.server.strategy.FedAvg() 

fl.server.start_server(
    server_address="127.0.0.1:8080",
    config=fl.server.ServerConfig(num_rounds=2),
    strategy=myStrategy,
    
)
