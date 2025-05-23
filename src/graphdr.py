import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import scipy
from sklearn.neighbors import kneighbors_graph
from scipy.sparse.csgraph import laplacian
from sklearn.decomposition import PCA


def graphdr_preprocessing(data, pca=True):
    """performs data preprocessing for the graphdr algorithm

    Parameters
    ----------
    data : numpy.ndarray
        input data to be dimension reduced, shape (d, n)
    pca : bool, optional
        whether to perform PCA as part of preprocessing, by default True

    Returns
    -------
    preprocessed_data : numpy.ndarray
        normalized per cell data, may be PCA'ed
    """
    #We will first normalize each cell by total count per cell.
    percell_sum = data.sum(axis=0)
    pergene_sum = data.sum(axis=1)

    preprocessed_data = data / percell_sum.values[None, :] * np.median(percell_sum)
    preprocessed_data = preprocessed_data.values

    #transform the preprocessed_data array by `x := log (1+x)`
    preprocessed_data = np.log(1 + preprocessed_data)

    #standard scaling
    preprocessed_data_mean = preprocessed_data.mean(axis=1)
    preprocessed_data_std = preprocessed_data.std(axis=1)
    preprocessed_data = (preprocessed_data - preprocessed_data_mean[:, None]) / \
                        preprocessed_data_std[:, None]

    if pca: # simple PCA
        pca_tool = PCA(n_components = 20)
        pca_tool.fit(preprocessed_data.T)

        # shape (n,d)
        preprocessed_data = pca_tool.transform(preprocessed_data.T)

    return preprocessed_data

def graphdr(data, lambda_=20, no_rotations=True):
    """performs graphDR dimension reduction

    Parameters
    ----------
    data : numpy.ndarray
        shape (n,d), may be PCA preprocessed
    lambda_ : int, optional
        lambda value in objective function, scaling factor for the sparse matrix, by default 20
    no_rotations : bool, optional
        skip W computing if no rotation, by default True

    Returns
    -------
    Z : numpy.ndarray
        graphDR output, solution to the objective function
    """
    n,d = data.shape

    # construct cell graph
    cell_graph = kneighbors_graph(data, n_neighbors=10)

    # laplacian transform, matrix need to be symmetrical
    graph_laplace = laplacian(cell_graph, symmetrized=True)

    # create operator matrix
    G = scipy.sparse.identity(n) + lambda_*graph_laplace
    K = np.linalg.inv(G.toarray())

    # solve for Z depending on rotation
    if no_rotations: # skips W
        Z = K @ data

    else:

        # first find the eigenvectors as W
        W = data.T @ K @ data
        eig_vals, W = np.linalg.eig(W)

        # solve for Z with W
        Z = K @ data @ W
    
    return Z

        
def main():
    # #TODO change data and annotation file here
    # data = pd.read_csv('../data/hochgerner_2018.data.gz',sep='\t',index_col=0)
    # anno = pd.read_csv('../data/hochgerner_2018.anno',sep='\t',header=None)
    # anno = anno[1].values

    # private use
    data = pd.read_csv('../data/imc_data.csv')
    data = np.abs(data.T) # match hochgerner single cell format
    anno = pd.read_csv('../data/imc_annot.csv')
    anno = anno['positive_marker'].values

    # preprocess and run graphDR algorithm
    data = graphdr_preprocessing(data)
    graphdr_data = graphdr(data, lambda_=10, no_rotations=True)

    # visualize and save plot
    plt.figure(figsize=(15,10))
    sns.scatterplot(x=graphdr_data[:,0], y=graphdr_data[:,1], linewidth=0, s=20, hue=anno)
    plt.xlabel('GraphDR 1')
    plt.ylabel('GraphDR 2')
    plt.savefig("graphdr_imc.png")


if __name__ == "__main__":
    main()


