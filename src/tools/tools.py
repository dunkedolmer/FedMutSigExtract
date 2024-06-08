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

    
def list_files(startpath):
    for root, dirs, files in os.walk(startpath):
        level = root.replace(startpath, '').count(os.sep)
        indent = ' ' * 4 * (level)
        print('{}{}/'.format(indent, os.path.basename(root)))
        subindent = ' ' * 4 * (level + 1)
        for f in files:
            print('{}{}'.format(subindent, f))


if __name__ == "__main__":
    # plot_signatures(pd.read_csv("sigGen/datasetOut/dataset.csv", index_col=0))
    plot_signatures("evaltest.tsv")
    df = pd.read_csv("src/tools/evaltest.tsv", sep="\t")
    print(df.head())  # Print the first few rows to check the data
    # plot_weights("sigGen/mdatasetOut/weights.csv")
