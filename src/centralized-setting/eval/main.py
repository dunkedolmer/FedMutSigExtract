import os
import sys
import shutil
import pandas as pd
import numpy as np
import time
from cluster import mk_cluster, ak_cluster, rk_cluster
from sklearn.decomposition import NMF
from method_evaluation import MethodEvaluator

sys.path.append(os.path.dirname(__file__) + "/../AE")
from ptAE import mseAE, klAE

sys.path.append(os.path.dirname(__file__) + "/../tools")
from tools import plot_signatures, plot_weights

current_file_path = os.path.dirname(__file__) + "/.."
evaluator = MethodEvaluator()

from pathmanager import PathManager
pathmanager: PathManager = PathManager() 

def find_datasets(path: str = os.path.dirname(__file__) + "/../datasets"):
    datasets = []
    # for all folders in path, check if they contain a dataset
    for folder in os.listdir(path):
        if folder[0] == ".":
            continue

        if os.path.isdir(path + "/" + folder):
            # if folder contains one file else if contains 3 files (my contain one folder but shuld be ignored)
            folder_list = [
                x
                for x in os.listdir(path + "/" + folder)
                if not os.path.isdir(path + "/" + folder + "/" + x)
            ]
            if len(folder_list) == 1:
                # find dataset
                datasets.append(
                    {
                        "type": 0,
                        "folder": path + "/" + folder,
                        "dataset": folder_list[0],
                    }
                )

            elif len(folder_list) == 3:
                # find dataset, signature and weights
                dataset = None
                signature = None
                weights = None
                for file in folder_list:
                    if "sig" in file:
                        signature = file
                    elif "wei" in file:
                        weights = file
                    else:
                        dataset = file

                datasets.append(
                    {
                        "type": 1,
                        "folder": path + "/" + folder,
                        "dataset": dataset,
                        "signature": signature,
                        "weights": weights,
                    }
                )
    return datasets


def nmf(df, components):
    model = NMF(n_components=components, init="random", random_state=0)
    W = model.fit_transform(df)
    H = model.components_
    return W, H, model.reconstruction_err_


def klnmf(df, components):
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


def evaluate(dataset: dict):
    # Case of real dataset
    if dataset["type"] == 0:
        return evaluator.COSMICevaluate(
            current_file_path + "/datasets/nmf_output/signatures.tsv",
            "GRCh37",
        )
    # Case of synthetic dataset
    else:
        result = evaluator.evaluate(
            current_file_path + "/datasets/nmf_output/signatures.tsv",
            current_file_path
            + "/datasets/nmf_output/output_weights/Assignment_Solution/Activities/Assignment_Solution_Activities.txt",
            dataset["folder"] + "/" + dataset["signature"],
            dataset["folder"] + "/" + dataset["weights"])
        
        return result


def save_results(results, methods, dataset):
    if dataset["type"] == 0:
        columns = ["found", ">0.8", ">0.95", "best>0.95", "best>0.99", "match"]
    elif dataset["type"] == 1:
        columns = [
            "found",
            ">0.8",
            ">0.95",
            "best>0.95",
            "best>0.99",
            "match",
            "mse",
            "mae",
            "rmse",
        ]
    else:
        columns = None
        
    print(f"results: {results}")
    print(f"columns (len): {len(columns)}")
    print(f"index: {methods}")
    
    return pd.DataFrame(
        results,
        columns=columns,
        index=methods,
    )


def save_output(method, dataset) -> None:
    '''Saves the output of a method based on a dataset in the results directory'''
    # Save output in results folder
    if not os.path.exists(pathmanager.results_centralized() + f"/{method}"):
        os.makedirs(f"{pathmanager.results_centralized()}/{method}")

    shutil.copy(
        current_file_path + "/datasets/nmf_output/aux_loss.png",
        pathmanager.results_centralized() + f"/{method}/aux_loss.png"
        # dataset["folder"] + f"/results/{method}/aux_loss.png",
    )
    shutil.copy(
        current_file_path + "/datasets//nmf_output/signatures.tsv",
        pathmanager.results_centralized() + f"/{method}/signatures.tsv"
        # dataset["folder"] + f"/results/{method}/signatures.tsv",
    )
    shutil.copy(
        current_file_path
        + "/datasets/nmf_output/output_weights/Assignment_Solution/Activities/Assignment_Solution_Activities.txt",
        pathmanager.results_centralized() + f"/{method}/weights.txt"
        # dataset["folder"] + f"/results/{method}/weights.txt",
    )


def plot_output(method, dataset):
    plot_signatures(
        pathmanager.results_centralized() + f"/{method}/signatures.tsv"
        ,
        method,
        pathmanager.results_centralized() + f"/{method}"
    )

    plot_weights(
        pathmanager.results_centralized() + f"/{method}/weights.txt",
        pathmanager.results_centralized() + f"/{method}"
    )


def test_dataset(dataset, methods):
    results = []
    execution_times = [] # Store the execution time of each method (for experiments)
    for method in methods:
        print(f"Testing {dataset['folder'].split('/')[-1]} with {method}")
        t_start = time.time()
        if method == "nmf_mk_mse":
            mk_cluster(dataset["folder"] + "/" + dataset["dataset"], nmf).run()
        elif method == "nmf_mk_kl":
            mk_cluster(dataset["folder"] + "/" + dataset["dataset"], klnmf).run()
        elif method == "AE_ak_mse":
            ak_cluster(
                dataset["folder"] + "/" + dataset["dataset"], mseAE, latents=200
            ).run()
        elif method == "AE_ak_kl":
            ak_cluster(
                dataset["folder"] + "/" + dataset["dataset"], klAE, latents=200
            ).run()

        # Find the execution time
        t_end = time.time() # Save end time
        t_total = t_end - t_start
        execution_times.append((method, t_total)) # Append the total execution time for the current method
        
        save_output(method, dataset)
        results.append(evaluate(dataset))

    # Log timing information in results
    for method, t_total in execution_times:
        print(f"Method {method} took {t_total:.2f} seconds")

    # save results
    save_results(results, methods, dataset).to_csv(
        pathmanager.results_centralized() + "/results.csv"
    )

def centralized_pipeline():
    # Locate and prepare data
    path = pathmanager.data_external()
    datasets = find_datasets(path)
    print(f"datasets: {len(datasets)}")
    print(f"dataset (type): {datasets[0]['type']}")
    # Run methods on the data
    # methods = ["nmf_mk_mse", "nmf_mk_kl", "AE_ak_mse", "AE_ak_kl"]
    methods = ["AE_ak_mse", "AE_ak_kl"]
    # methods = ["nmf_mk_mse", "nmf_mk_kl"]
    # methods = ["AE_ak_kl"]
    for dataset in datasets:
        test_dataset(dataset, methods)

    # for dataset in datasets:
    #     for method in methods:
    #         plot_output(method, dataset)

if __name__ == "__main__":
    centralized_pipeline()