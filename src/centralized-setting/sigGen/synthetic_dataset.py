import os
import pandas as pd
import numpy as np


def _generate_random_numbers_with_zeros(
    num_numbers, desired_sum, percentage_zeros, use_integers=False
):
    assert 0 <= percentage_zeros <= 1
    num_zeros = int(percentage_zeros * num_numbers)
    # generate random numbers
    random_numbers = np.random.rand(num_numbers - num_zeros)
    # scale them to the desired sum
    random_numbers /= random_numbers.sum()
    random_numbers *= desired_sum
    # round them to integers
    if use_integers:
        random_numbers = np.round(random_numbers)
    # add zeros
    random_numbers = np.append(random_numbers, np.zeros(num_zeros))
    # shuffle
    np.random.shuffle(random_numbers)
    return random_numbers


def _cosineSimilarity(a: np.ndarray, b: np.ndarray):
    result = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    if np.isnan(result):
        return 0
    return result


def _makeCosineSimilarityMatrix(
    signatures: np.ndarray,
    knownSignatures: np.ndarray,
):
    cosMatrix = np.zeros((signatures.shape[1], knownSignatures.shape[1]))
    for i in range(signatures.shape[1]):
        for j in range(knownSignatures.shape[1]):
            cosMatrix[i, j] = _cosineSimilarity(signatures[:, i], knownSignatures[:, j])

    return cosMatrix


def _selectSignatures(signatures: np.ndarray, sigNum: int, cosineTreshhold: float):
    # select signatures
    sigIndex = np.random.choice(signatures.shape[1], sigNum, replace=False)
    sigMatrix = signatures.to_numpy()[:, sigIndex]
    sigName = signatures.columns[sigIndex]

    # cosine similarity
    cosMatrix = _makeCosineSimilarityMatrix(sigMatrix, sigMatrix)
    np.fill_diagonal(cosMatrix, 0)
    cosOverTreshhold = np.sum(cosMatrix > cosineTreshhold) / 2

    # repeat until cosine similarity is below cosineTreshhold
    if cosOverTreshhold > 0:
        sigIndex, sigMatrix, sigName = _selectSignatures(
            signatures, sigNum, cosineTreshhold
        )

    return sigIndex, sigMatrix, sigName


# greate dataset
def create_dataset(
    sigNum: int,
    numDatapoints: int,
    signaturePath: str,
    output_path: str = os.path.dirname(__file__) + "/datasetOut/",
    pSigExposure: float = 0.5,
    cosineTreshhold: float = 0.7,
    mutationRange: tuple = (500, 1000),
    normalDist: bool = False,
    noiseScale: float = 0.01,
    normalize: bool = False,
):
    # create output folder if not exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # dataset (mutations X signatures)
    signatures = pd.read_table(signaturePath, index_col=0)

    # randomly select sigNum signatures
    sigIndex, sigMatrix, sigName = _selectSignatures(
        signatures, sigNum, cosineTreshhold
    )

    # random mutation counts for numDatapoints
    mutationCounts = np.random.randint(
        mutationRange[0], mutationRange[1], numDatapoints
    )

    # random waight (signature x numDatapoints)
    weights = _generate_random_numbers_with_zeros(
        sigNum, mutationCounts[0], pSigExposure, True
    )
    for i in range(numDatapoints - 1):
        weights = np.vstack(
            (
                weights,
                _generate_random_numbers_with_zeros(
                    sigNum, mutationCounts[i + 1], pSigExposure, True
                ),
            )
        )
    weights = weights.T

    # create dataset
    dataset = np.dot(sigMatrix, weights)

    # add noise
    if normalDist:
        noise = np.random.normal(0, 1, dataset.shape)
    else:
        noise = np.random.poisson(1, dataset.shape)

    nsum = np.sum(noise, axis=0)
    noise = np.divide(noise, np.where(nsum == 0, 1, nsum)) * mutationCounts * noiseScale

    dataset = np.clip(dataset + noise, 0, None)

    # save dataset
    dataset = pd.DataFrame(dataset)
    if normalize:
        dataset = dataset.div(dataset.sum(axis=1), axis=0)
    dataset.index = signatures.index
    # dataset = dataset.rename(index=lambda x: x[0] + x[2] + x[6] + "-" + x[4])
    dataset = dataset.rename(columns=lambda x: f"Genome {x} ({mutationCounts[x]})")
    dataset.to_csv(
        output_path + "dataset.txt",
        sep="\t",
    )

    sigMatrix = pd.DataFrame(sigMatrix)
    sigMatrix.index = signatures.index
    sigMatrix.columns = sigName
    sigMatrix.to_csv(output_path + "sigMatrix.csv")

    weights = pd.DataFrame(weights)
    weights.index = signatures.columns[sigIndex]
    weights.columns = dataset.columns
    weights.to_csv(output_path + "weights.csv")


if __name__ == "__main__":
    create_dataset(5, 500, "sigGen/signatures/simple8.txt")
