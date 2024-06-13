import pandas as pd
import numpy as np
import sys
import re
import os
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from sklearn.exceptions import ConvergenceWarning
from SigProfilerAssignment import Analyzer
from typing import Callable
import warnings

# Ignore Kmeans future warning
warnings.filterwarnings("ignore", category=FutureWarning)
# ignore NMF convergence warning
warnings.filterwarnings("ignore", category=ConvergenceWarning)


class k_cluster:
    def __init__(
        self,
        dataset: str,
        method: Callable[[pd.DataFrame, int], tuple[np.ndarray, np.ndarray, float]],
        runs: int = 10,
        components: tuple = (2, 10),
    ):
        assert dataset != None, "dataset must be a string"
        self.dataset = dataset
        if re.search(r"\.(tsv)|(txt)$", dataset):
            self.df = pd.read_csv(dataset, sep="\t", index_col=0)
            print(f"self.df: {self.df}")
        else:
            self.df = pd.read_csv(dataset, index_col=0)
        assert runs > 0, "runs must be greater at least 1"
        self.runs = runs
        assert components[0] > 1, "components[0] must be greater than 1"
        assert (
            components[1] > components[0]
        ), "components[1] must be greater than components[0]"
        self.components = range(components[0], components[1] + 1)
        assert method != None, "method must be a function"
        assert callable(method), "method must be a function"
        self.method = method
        self.output_path = os.path.dirname(__file__) + "/../datasets/nmf_output"
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

    def run(self):
        print("Running method_cluster")
        self._method_iteration()
        print("Finding signatures")
        self._cluster_signatures()
        print(f"Found {len(self.signatures)} signatures")
        print("Finding exposures")
        self._find_exposures()

    def _method_iteration(self):
        W_iterations = []
        H_iterations = []
        losses = []
        for i in self.components:
            W_concat = []
            H_concat = []
            loss = []
            for _ in range(self.runs):
                W, H, l = self.method(
                    self.df,
                    i,
                )
                W_concat.append(W)
                H_concat.append(H)
                loss.append(l)
            W_iterations.append(np.stack(W_concat))
            H_iterations.append(np.stack(H_concat))
            losses.append(loss)

        self.prelim_signatures = [np.hstack(x).T for x in W_iterations]
        self.H_iterations = [np.hstack(x).T for x in H_iterations]
        self.avg_loss = [np.mean(x) for x in losses]

    def _cluster_signatures(self):
        pass

    def _find_exposures(self):
        # save signatures
        signatures = pd.DataFrame(self.signatures, columns=self.df.index).T
        signatures.columns = [f"Sig{i}" for i in range(1, signatures.shape[1] + 1)]
        signatures.to_csv(self.output_path + "/signatures.tsv", sep="\t")
        
        # find exposures
        Analyzer.cosmic_fit(
            samples=self.dataset,
            output=self.output_path + "/output_weights",
            signature_database=self.output_path + "/signatures.tsv",
            input_type="matrix",
        )

    def _plot_aux_loss(self, cluster_components, aux_loss, xlabel, ylabel):
        plt.plot(cluster_components, aux_loss)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.savefig(self.output_path + "/aux_loss.png")


class mk_cluster(k_cluster):
    def _cluster_signatures(self):
        silhouette_scores = []
        cluster_centroids = []

        for i, num_c in enumerate(self.components):
            km = KMeans(n_clusters=num_c).fit(self.prelim_signatures[i])
            cluster_centroids.append(km.cluster_centers_)
            silhouette_scores.append(
                [num_c, silhouette_score(self.prelim_signatures[i], km.labels_)]
            )

        silhouette_alone = [i[1] for i in silhouette_scores]
        silhouette_hat = (
            np.array(silhouette_alone) - np.mean(silhouette_alone)
        ) / np.std(silhouette_alone)
        loss_hat = (self.avg_loss - np.mean(self.avg_loss)) / np.std(self.avg_loss)
        aux_loss = loss_hat - 1 * silhouette_hat

        self.signatures = cluster_centroids[np.argmin(aux_loss)]

        self._plot_aux_loss(
            self.components,
            aux_loss,
            "Number of components",
            "Auxiliary loss",
        )


class rk_cluster(k_cluster):
    def _cluster_signatures(self):
        cluster_centroids = []
        silhouette_scores = []
        inertia_scores = []

        for i, num_c in enumerate(self.components):
            c_cluster_centroids = []
            c_silhouette_scores = []
            c_inertia_scores = []
            for j in range(2, num_c + 1):
                km = KMeans(n_clusters=j).fit(self.prelim_signatures[i])
                c_cluster_centroids.append(km.cluster_centers_)
                c_silhouette_scores.append(
                    silhouette_score(self.prelim_signatures[i], km.labels_)
                )
                c_inertia_scores.append(km.inertia_)
            cluster_centroids.append(c_cluster_centroids)
            silhouette_scores.append(c_silhouette_scores)
            inertia_scores.append(c_inertia_scores)

        cluster_components = []
        for i, centroids in enumerate(cluster_centroids):
            silhouette_alone = silhouette_scores[i]
            inertia_alone = inertia_scores[i]
            silhouette_hat = (
                np.array(silhouette_alone) - np.mean(silhouette_alone)
            ) / np.std(silhouette_alone)
            inertia_hat = (np.array(inertia_alone) - np.mean(inertia_alone)) / np.std(
                inertia_alone
            )
            aux_loss = inertia_hat - 1 * silhouette_hat
            cluster_components.append(np.argmin(aux_loss) + self.components[0])
            cluster_centroids[i] = centroids[np.argmin(aux_loss)]
            silhouette_scores[i] = silhouette_alone[np.argmin(aux_loss)]
            inertia_scores[i] = inertia_alone[np.argmin(aux_loss)]

        silhouette_hat = (
            np.array(silhouette_scores) - np.mean(silhouette_scores)
        ) / np.std(silhouette_scores)
        loss_hat = (self.avg_loss - np.mean(self.avg_loss)) / np.std(self.avg_loss)
        aux_loss = loss_hat - 1 * silhouette_hat

        self.signatures = cluster_centroids[np.argmin(aux_loss)]

        cluster_components = [
            f"{i + self.components[0]}/{x}" for i, x in enumerate(cluster_components)
        ]
        self._plot_aux_loss(
            cluster_components,
            aux_loss,
            "components/clusters",
            "Auxiliary loss",
        )


class ak_cluster(k_cluster):
    '''
    Autoencoder Cluster Class

    This class is designed for mutational signature extraction using an autoencoder clustering approach.

    Attributes:
        dataset (str): The path to the dataset.
        method (Callable[[pd.DataFrame, int], tuple[np.ndarray, np.ndarray]]): A callable method for performing dimensionality reduction and obtaining latent representations. It takes a pandas DataFrame (dataset) and an integer (number of latent dimensions) as input and returns a tuple containing the weight matrix W, the latent representation matrix H, and the reconstruction loss l.
        runs (int): The number of runs for the autoencoder clustering algorithm.
        latents (int): The number of latent dimensions for the autoencoder.

    Methods:
        _method_iteration: Performs the autoencoder method for the specified number of runs, storing preliminary signatures, H iterations, and average loss.
        _cluster_signatures: Clusters the preliminary signatures using KMeans for each run, calculates silhouette scores and inertia scores, selects the cluster with the lowest auxiliary loss, and calculates final signatures.

    '''

    def __init__(
        self,
        model,
        dataset: str,
        method: Callable[[pd.DataFrame, int], tuple[np.ndarray, np.ndarray]],
        runs: int = 10,
        latents: int = 64,
    ):
        self.latents = latents
        self.loss = None # initial loss
        self.model = model 
        super().__init__(dataset, method, runs, (2, latents - 1))

    def _method_iteration(self):
        self.prelim_signatures = []
        self.H_iterations = []
        self.avg_loss = []
        for _ in range(self.runs):
            W, H, l, self.model = self.method(
                self.df,
                self.latents,
                self.model # Pass in the current model
            )
            self.prelim_signatures.append(W.T)
            self.H_iterations.append(H)
            self.avg_loss.append(l)
            self.loss = l # Store loss

    def _cluster_signatures(self):
        cluster_centroids = []
        silhouette_scores = []
        inertia_scores = []

        for i in range(self.runs):
            c_cluster_centroids = []
            c_silhouette_scores = []
            c_inertia_scores = []
            
            for j in self.components:
                km = KMeans(n_clusters=j).fit(self.prelim_signatures[i])
                c_cluster_centroids.append(km.cluster_centers_)
                c_silhouette_scores.append(
                    silhouette_score(self.prelim_signatures[i], km.labels_)
                )
                c_inertia_scores.append(km.inertia_)
            cluster_centroids.append(c_cluster_centroids)
            silhouette_scores.append(c_silhouette_scores)
            inertia_scores.append(c_inertia_scores)

        cluster_components = []
        for i, centroids in enumerate(cluster_centroids):
            silhouette_alone = silhouette_scores[i]
            inertia_alone = inertia_scores[i]
            silhouette_hat = (
                np.array(silhouette_alone) - np.mean(silhouette_alone)
            ) / np.std(silhouette_alone)
            inertia_hat = (np.array(inertia_alone) - np.mean(inertia_alone)) / np.std(
                inertia_alone
            )
            aux_loss = inertia_hat - 1 * silhouette_hat
            cluster_components.append(np.argmin(aux_loss) + self.components[0])
            cluster_centroids[i] = centroids[np.argmin(aux_loss)]
            silhouette_scores[i] = silhouette_alone[np.argmin(aux_loss)]
            inertia_scores[i] = inertia_alone[np.argmin(aux_loss)]

        silhouette_hat = (
            np.array(silhouette_scores) - np.mean(silhouette_scores)
        ) / np.std(silhouette_scores)
        loss_hat = (self.avg_loss - np.mean(self.avg_loss)) / np.std(self.avg_loss)
        aux_loss = loss_hat - 1 * silhouette_hat

        self.signatures = cluster_centroids[np.argmin(aux_loss)]

        cluster_components = [f"{i}-{x}" for i, x in enumerate(cluster_components)]
        self._plot_aux_loss(
            cluster_components,
            aux_loss,
            "Run-components",
            "Auxiliary loss",
        )
    
    def get_model_parameters(self):
        '''Retrieves the parameters of the trained model'''
        return self.model.state_dict()

if __name__ == "__main__":
    from sklearn.decomposition import NMF

    def nmf(df, components):
        model = NMF(n_components=components, init="random", random_state=0)
        W = model.fit_transform(df)
        H = model.components_
        return W, H, model.reconstruction_err_

    mk_cluster(
        dataset="sigGen/datasetOut/dataset.txt",
        method=nmf,
        runs=10,
        components=(2, 10),
    ).run()