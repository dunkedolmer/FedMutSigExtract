import os
import sys
import re
import numpy as np
import pandas as pd
from typing import Literal
import matplotlib.pyplot as plt
from urllib.request import urlretrieve
from scipy.optimize import linear_sum_assignment
sys.path.append(os.path.dirname(__file__) + "/../tools")
from pathmanager import PathManager
pathmanager: PathManager = PathManager() 
current_file_path = os.path.dirname(__file__) + "/.." + "/.."

URL_COSMIC_SIGNATURES_GRCh37 = "https://cancer.sanger.ac.uk/signatures/documents/2046/COSMIC_v3.3.1_SBS_GRCh37.txt"
URL_COSMIC_SIGNATURES_GRCh38 = "https://cancer.sanger.ac.uk/signatures/documents/2047/COSMIC_v3.3.1_SBS_GRCh38.txt"

class MethodEvaluator:
    def __init__(self) -> None:
        self.data_path = current_file_path + "/data" + "/external" + "/.cosmic"
        self.GRCh37, self.GRCh38 = self._getCOSMICData()        

    def _getCOSMICData(self):
        if not os.path.exists(self.data_path):
            os.mkdir(self.data_path)
        if not os.path.exists(
            self.data_path + "/COSMIC_v3.3.1_SBS_GRCh37.txt"
        ) or not os.path.exists(self.data_path + "/COSMIC_v3.3.1_SBS_GRCh38.txt"):
            print("Downloading COSMIC data...")
            urlretrieve(
                URL_COSMIC_SIGNATURES_GRCh37,
                self.data_path + "/COSMIC_v3.3.1_SBS_GRCh37.txt",
            )
            urlretrieve(
                URL_COSMIC_SIGNATURES_GRCh38,
                self.data_path + "/COSMIC_v3.3.1_SBS_GRCh38.txt",
            )
            print("Done")

        GRCh37 = pd.read_table(
            self.data_path + "/COSMIC_v3.3.1_SBS_GRCh37.txt", index_col=0
        )
        GRCh38 = pd.read_table(
            self.data_path + "/COSMIC_v3.3.1_SBS_GRCh38.txt", index_col=0
        )

        return GRCh37, GRCh38

    def _cosineSimilarity(self, a: np.ndarray, b: np.ndarray):
        result = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        if np.isnan(result):
            return 0
        return result

    def _makeCosineSimilarityMatrix(
        self,
        signatures: pd.DataFrame,
        knownSignatures: pd.DataFrame,
    ):
        signatures = signatures.to_numpy()
        knownSignatures = knownSignatures.to_numpy()
        cosMatrix = np.zeros((signatures.shape[1], knownSignatures.shape[1]))
        for i in range(signatures.shape[1]):
            for j in range(knownSignatures.shape[1]):
                cosMatrix[i, j] = self._cosineSimilarity(
                    signatures[:, i], knownSignatures[:, j]
                )

        return cosMatrix

    def _evaluateAgainstKnownSignatures(
        self,
        signatures: pd.DataFrame,
        knownSignatures: pd.DataFrame,
    ):
        self.cosineSimilarityMatrix = self._makeCosineSimilarityMatrix(
            signatures, knownSignatures
        )

        # count how many rows have a collum over 0.8 and 0.95
        numCos80 = np.sum(np.sum(self.cosineSimilarityMatrix >= 0.8, axis=1) >= 1)
        numCos95 = np.sum(np.sum(self.cosineSimilarityMatrix >= 0.95, axis=1) >= 1)

        return numCos80, numCos95

    def _checkDuplicates(self, signatures: pd.DataFrame, knownSignatures: pd.DataFrame):
        # hungarian algorithm to find the best match
        self.hungarian_row_ind, self.hungarian_col_ind = linear_sum_assignment(
            self.cosineSimilarityMatrix, maximize=True
        )

        # best match cosin values
        hungarianMatch = self.cosineSimilarityMatrix[
            self.hungarian_row_ind, self.hungarian_col_ind
        ]

        self.bestMatchCosin = [
            (i, j, k)
            for i, j, k in zip(
                hungarianMatch,
                signatures.columns[self.hungarian_row_ind],
                knownSignatures.columns[self.hungarian_col_ind],
            )
        ]

    def _evaluateAgainstKnownWeights(
        self,
        weights: pd.DataFrame,
        knownWeights: pd.DataFrame,
    ):
        # normalize weights
        weights = weights.div(weights.sum(axis=1), axis=0).to_numpy()
        knownWeights = knownWeights.div(knownWeights.sum(axis=1), axis=0).to_numpy()
        print(f"weights shape: {weights.shape}, knownWeights shape: {knownWeights.shape}")
        
        # compare (mse, mae, rmse) weights with known weights, given the index
        mse = []
        mae = []
        rmse = []
        for i in range(weights.shape[1]):
            mse.append(
                np.sum(
                    (
                        weights[self.hungarian_row_ind, i]
                        - knownWeights[self.hungarian_col_ind, i]
                    )
                    ** 2
                )
            )
            mae.append(
                np.sum(
                    np.abs(
                        weights[self.hungarian_row_ind, i]
                        - knownWeights[self.hungarian_col_ind, i]
                    )
                )
            )
            rmse.append(np.sqrt(mse[-1]))

        return np.average(mse), np.average(mae), np.average(rmse)

        mse = (
            (
                weights[self.hungarian_row_ind, :]
                - knownWeights[self.hungarian_col_ind, :]
            )
            ** 2
        ).mean()
        mae = np.abs(
            weights[self.hungarian_row_ind, :] - knownWeights[self.hungarian_col_ind, :]
        ).mean()
        rmse = np.sqrt(mse)

        return mse, mae, rmse

    def COSMICevaluate(
        self,
        signatures: pd.DataFrame | str,
        GRCh: Literal["GRCh37", "GRCh38"] = "GRCh37",
    ):
        if isinstance(signatures, str):
            if re.search(r"\.tsv$", signatures):
                signatures = pd.read_csv(signatures, sep="\t", index_col=0)
            else:
                signatures = pd.read_csv(signatures, index_col=0)

        # Signatures found by the method
        numFoundSig = signatures.shape[1]

        if GRCh == "GRCh37":
            knownSignatures = self.GRCh37
        elif GRCh == "GRCh38":
            knownSignatures = self.GRCh38

        # Known signatures found by the method
        numFoundCos80, numFoundCos95 = self._evaluateAgainstKnownSignatures(
            signatures, knownSignatures
        )

        # check for duplicate signatures
        self._checkDuplicates(signatures, knownSignatures)

        numBest95 = len([x[0] for x in self.bestMatchCosin if x[0] >= 0.95])
        numBest99 = len([x[0] for x in self.bestMatchCosin if x[0] >= 0.99])

        if __name__ == "__main__":
            print(f"Signatures found: {numFoundSig}")
            print(f"Signatures with cosin > 0.8: {numFoundCos80}")
            print(f"Signatures with cosin > 0.95: {numFoundCos95}")
            print(f"Best match cosin > 0.95: {numBest95}")
            print(f"Best match cosin > 0.99: {numBest99}")
            print(f"Best match cosin: {self.bestMatchCosin}")

        return (
            numFoundSig,
            numFoundCos80,
            numFoundCos95,
            numBest95,
            numBest99,
            self.bestMatchCosin,
        )

    def evaluate(
        self,
        signatures: pd.DataFrame | str,
        weights: pd.DataFrame | str,
        knownSignatures: pd.DataFrame | str,
        knownWeights: pd.DataFrame | str,
    ):
        if isinstance(signatures, str):
            if re.search(r"\.(tsv)|(txt)$", signatures):
                signatures = pd.read_csv(signatures, sep="\t", index_col=0)
            else:
                signatures = pd.read_csv(signatures, index_col=0)
        if isinstance(weights, str):
            if re.search(r"\.(tsv)|(txt)$", weights):
                weights = pd.read_csv(weights, sep="\t", index_col=0)
            else:
                weights = pd.read_csv(weights, index_col=0)
        if isinstance(knownSignatures, str):
            if re.search(r"\.(tsv)|(txt)$", knownSignatures):
                knownSignatures = pd.read_csv(knownSignatures, sep="\t", index_col=0)
            else:
                knownSignatures = pd.read_csv(knownSignatures, index_col=0)
        if isinstance(knownWeights, str):
            if re.search(r"\.(tsv)|(txt)$", knownWeights):
                knownWeights = pd.read_csv(knownWeights, sep="\t", index_col=0)
            else:
                knownWeights = pd.read_csv(knownWeights, index_col=0)
                print(f"Reading weights (.csv)...")
                print(f"knownWeights shape: ({knownWeights.shape})")

        # Signatures found by the method
        numFoundSig = signatures.shape[1]

        if weights.shape[0] != numFoundSig:
            weights = weights.T

        print(f"weights.shape[1]: {weights.shape[1]}, knownWeights.shape[1]: {knownWeights.shape[1]}")
        assert weights.shape[1] == knownWeights.shape[1]

        # Known signatures found by the method
        numFoundCos80, numFoundCos95 = self._evaluateAgainstKnownSignatures(
            signatures, knownSignatures
        )

        # check for duplicate signatures
        self._checkDuplicates(signatures, knownSignatures)

        numBest95 = len([x[0] for x in self.bestMatchCosin if x[0] >= 0.95])
        numBest99 = len([x[0] for x in self.bestMatchCosin if x[0] >= 0.99])

        # accuracy of weights
        # mse, mae, rmse = self._evaluateAgainstKnownWeights(
        #     weights,
        #     knownWeights,
        # )

        if __name__ == "__main__":
            print(f"Signatures found: {numFoundSig}")
            print(f"Signatures with cosine > 0.8: {numFoundCos80}")
            print(f"Signatures with cosine > 0.95: {numFoundCos95}")
            print(f"Best match cosine > 0.95: {numBest95}")
            print(f"Best match cosine > 0.99: {numBest99}")
            print(f"Best match cosine: {self.bestMatchCosin}")
            # print(f"Weight error (MSE): {mse}")
            # print(f"Weight error (MAE): {mae}")
            # print(f"Weight error (RMSE): {rmse}")

        return (
            numFoundSig,
            numFoundCos80,
            numFoundCos95,
            numBest95,
            numBest99,
            self.bestMatchCosin,
            # mse,
            # mae,
            # rmse,
        )


if __name__ == "__main__":
    evaluator = MethodEvaluator()
    results = evaluator.evaluate(
        "sigGen/datasetOut/sigMatrix.csv",
        "sigGen/datasetOut/weights.csv",
        "sigGen/datasetOut/sigMatrix.csv",
        "sigGen/datasetOut/weights.csv",
    )
    # results = evaluator.COSMICevaluate("eval/nmf_output/signatures.tsv", GRCh="GRCh37")
    print(results)
