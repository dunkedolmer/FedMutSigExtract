import os
import re
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sigProfilerPlotting as sigPlt

current_path = os.path.dirname(__file__)


# cluster sifnaturea
def cluster_signatures(signatures: str | pd.DataFrame, outputPath: str = current_path):
    """
    Clusters similar mutational signatures by merging those with a cosine similarity of 0.8 or higher.

    Args:
        signatures (str | pd.DataFrame): The file path to a CSV file containing signatures or a DataFrame of signatures.
        outputPath (str): The directory where the clustered signatures will be saved. Defaults to the current script path.

    Returns:
        None
    """
    if isinstance(signatures, str):
        signatures = pd.read_csv(signatures, index_col=0)
    data = signatures.to_numpy().T

    def similarity(a: np.ndarray, b: np.ndarray):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    more = True
    while more and data.shape[0] > 1:
        loop = [
            (x, y) for x in range(data.shape[0]) for y in range(data.shape[0]) if x != y
        ]
        # print(data.shape)
        for i, j in loop:
            if similarity(data[i, :], data[j, :]) >= 0.8:
                # avg i and j, and remove j
                data[i, :] = (data[i, :] + data[j, :]) / 2
                data = np.delete(data, j, 0)
                break
            if (i, j) == loop[-1]:
                more = False

    results = pd.DataFrame(data.T)
    results.index = signatures.index
    results.columns = results.columns + 1
    results.to_csv(outputPath + "/SBS_signatures.tsv", sep="\t")


# plot bar chart for signatures
def plot_signatures(
    sigMatrix: str, project: str = "test", outputPath: str = current_path
):
    """
    Plots mutational signatures as a bar chart and saves it as a PDF.

    Args:
        sigMatrix (str): The file path to the signature matrix in CSV or TSV format.
        project (str): The project name for the plot title and output directory.
        outputPath (str): The directory where the plot will be saved. Defaults to the current script path.

    Returns:
        None
    """
    tmp = None
    # if ends in csv
    if isinstance(sigMatrix, str) and re.search(r"\.csv$", sigMatrix):
        # get file name without path
        tmp = sigMatrix.split("/")[-1]

        df = pd.read_csv(sigMatrix, index_col=0)
        # save as tsv
        tmp = current_path + "/" + re.sub(r"\.csv$", ".tsv", tmp)
        df.to_csv(
            tmp,
            sep="\t",
        )
        sigMatrix = tmp

    sigPlt.plotSBS(
        sigMatrix,
        output_path=outputPath + "/",
        project=project,
        plot_type="96",
        percentage=True,
        savefig_format="pdf",
    )

    if tmp is not None:
        os.remove(tmp)


# plot % bar chart for signature weights
def plot_weights(
    weights: pd.DataFrame | str, outputPath: str = current_path, numWeights: int = 20
):
    """
    Plots the weights of mutational signatures as a stacked bar chart.

    Args:
        weights (pd.DataFrame | str): The weights of the signatures in a DataFrame or the file path to a CSV/TSV file.
        outputPath (str): The directory where the plot will be saved. Defaults to the current script path.
        numWeights (int): The number of weights to plot. Defaults to 20.

    Returns:
        None
    """
    if isinstance(weights, str):
        if re.search(r"\.tsv$", weights) or re.search(r"\.txt$", weights):
            weights = pd.read_csv(weights, sep="\t", index_col=0)
        else:
            weights = pd.read_csv(weights, index_col=0)

    # weights = weights.T

    # normalize
    weights = weights.div(weights.sum(axis=1), axis=0).iloc[
        : numWeights if weights.shape[0] > numWeights else weights.shape[0]
    ]

    # plot
    weights.plot(
        kind="bar",
        stacked=True,
        title="Signature Weights",
        figsize=(weights.shape[0] / 5, 10),
        legend=False,
    ).figure.legend(loc="outside right upper")
    plt.savefig(outputPath + "/weights.jpg", bbox_inches="tight")


if __name__ == "__main__":
    # plot_signatures(pd.read_csv("sigGen/datasetOut/dataset.csv", index_col=0))
    plot_signatures("sigGen/signatures/COSMIC_v3.3.1_SBS_GRCh37.txt")
    # plot_weights("sigGen/mdatasetOut/weights.csv")
