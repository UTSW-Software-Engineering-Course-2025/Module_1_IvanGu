import numpy as np
import matplotlib.pyplot as plt
import argparse

import adjustbeta as utils

def pca(X, no_dims=50):
    """
    Runs PCA on the nxd array X in order to reduce its dimensionality to
    no_dims dimensions.

    Parameters
    ----------
    X : numpy.ndarray
        data input array with dimension (n,d)
    no_dims : int
        number of dimensions that PCA reduce to

    Returns
    -------
    Y : numpy.ndarray
        low-dimensional representation of input X
    """
    n, d = X.shape
    X = X - X.mean(axis=0)[None, :]
    _, M = np.linalg.eig(np.dot(X.T, X))
    Y = np.real(np.dot(X, M[:, :no_dims]))
    return Y


def compute_q_ij(Y):
    """
    computes probability distribution (t-distribution) based on low-dimension embedding Y

    Parameters
    ----------
    Y : numpy.ndarray
        low-dimensional representation of Y

    Returns
    -------
    num : numpy.ndarray
        inverse distance matrix of Y, to be used for gradient calculations
    
    q_ij_clipped : numpy.ndarray
        clipped Q_ij with minimum val = 1e-12    
    """
    
    # compute nxn distance matrix and inverse dist matrix
    D_y = utils.compute_D(Y)
    num = 1 / (1+D_y)

    # set diaonal to zero
    num[np.arange(num.shape[0]), np.arange(num.shape[0])] = 0
    denom = np.sum(num)

    # calculate Q_ij
    q_ij = num / denom
    q_ij_clipped = np.clip(q_ij, a_min=1e-12, a_max=np.inf)

    return num, q_ij_clipped

def compute_gradient(P, Q, Y, inv_D):
    """computes the gradient w.r.t. to Y

    Parameters
    ----------
    P : numpy.ndarray
        P_ij, pairwise and symmetrical to P_j|i and P_i|j
    Q : numpy.ndarray
        Q_ij, probability distribution
    Y : numpy.ndarray
        low-dimension representation of Y
    inv_D : numpy.ndarray
        inverse distance matrix of Y, 1st output from compute_q_ij()

    Returns
    -------
    numpy.ndarray
        gradient of loss funciton w.r.t. to Y
    """

    # initialize dY
    n, d = Y.shape
    dY = np.zeros_like(Y)

    # diff of P_ij and Q_ij, shape (n,n)
    pq_diff = P - Q
    
    for row in range(n):
        y_diff_term = Y[row,:][None,:]-Y # shape (n,d)
        pq_diff_term = pq_diff[row,:,None] # shape (n,1)
        inv_D_term = inv_D[row,:,None] # shape (n,1)

        weights = pq_diff_term * y_diff_term * inv_D_term # shape (n,d)
        dY[row] = np.sum(weights, axis=0) # shape (d,)

    return dY

def tsne(X, no_dims, perplexity, initial_momentum=0.5, final_momentum=0.8, eta=500, min_gain=0.01, T=1000, print_all=True):
    """implementation of tsne from scratch

    Parameters
    ----------
    X : numpy.adarray
        nxd input of matrix X, either before or after PCA
    no_dims : int
        number of dimensions to visualize tsne
    perplexity : int
        perplexity of tsne, used for precision adjustment
    initial_momentum : float, optional
        determines changes in Y in the initial stage, by default 0.5
    final_momentum : float, optional
        determines changes in Y after the initial stage, by default 0.8
    eta : int, optional
        eta-another component that determines delta Y, by default 500
    min_gain : float, optional
        minimum gains used for clipping during optimization, by default 0.01
    T : int, optional
        number of timesteps, by default 1000

    Returns
    -------
    Y
        low-dimension t-sne representation of input X
    """

    (n, d) = X.shape

    # precision adjustment by perplexity
    cond_p_ji, beta = utils.adjustbeta(X, tol=1e-5, perplexity=perplexity)

    # calculate pairwise p(i,j)
    # here assuming P(j|i) and P(i|j) are symmetrical
    p_ij = (cond_p_ji + cond_p_ji.T) / np.sum(cond_p_ji + cond_p_ji.T)
    
    # early exaggerate and clipping
    p_ij = np.clip(4*p_ij, a_min=1e-12, a_max=np.inf)

    # initialize Y, deltaY, and gains
    Y = pca(X, no_dims=no_dims)
    delta_y = np.zeros((n, no_dims))
    gains = np.ones((n, no_dims))
    
    # loop through time points to update Y
    for t in range(T):

        # computes Q_ij then gradient in each iteration
        inv_D, q_ij = compute_q_ij(Y)
        dY = compute_gradient(p_ij, q_ij, Y, inv_D)

        # monitor KL divergence for debugging
        kl_divergence = np.sum(p_ij * np.log(p_ij / q_ij))
        if (t % 10 == 0) and print_all:
            print(f"[t={t}] KL: {kl_divergence:.4f}, grad: {np.linalg.norm(dY):.2e}, Y: {np.linalg.norm(Y):.2e}")

        # determines momentum based on iteration progress
        momentum = initial_momentum if t < 20 else final_momentum

        # calculates gains in each step
        gains_02_logic = (dY > 0) != (delta_y > 0)
        gains_08_logic = (dY > 0) == (delta_y > 0)
        gains = (gains+0.2)*gains_02_logic + gains*0.8*gains_08_logic
        
        # clip to the minimum gain value
        gains = np.clip(gains, a_min=min_gain, a_max=np.inf)

        # final determination of delta Y
        delta_y = momentum*delta_y - eta * (gains*dY)
        Y += delta_y

        # removing early exaggeration
        if t==100:
            p_ij  /= 4

    return Y

def get_non_default_args(args, default_args):
    non_defaults = {}
    for key, value in vars(args).items():
        default_value = getattr(default_args, key)
        if value != default_value:
            non_defaults[key] = value
    return non_defaults


def parse_tsne_args():
    argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    # file path args
    argparser.add_argument("--path_to_X", type=str, default="", help="path to the input file")
    argparser.add_argument("--path_to_labels", type=str, default="../data/mnist2500_labels.txt", help="path to the labels file")

    # tsne args
    argparser.add_argument("--no_dims", type=int, default=2, help="number of dimensions")
    argparser.add_argument("--perplexity", type=int, default=30, help="perplexity of tsne, used for precision adjustment")
    argparser.add_argument("--T", type=int, default=1000, help="number of time steps")
    argparser.add_argument("--initial_momentum", type=int, default=0.5, help="initial momentum during early stage")
    argparser.add_argument("--final_momentum", type=int, default=0.8, help="momentum after early stage")
    argparser.add_argument("--eta", type=int, default=500, help="another component that determines delta Y")    
    argparser.add_argument("--min_gain", type=int, default=0.01, help="minimum gains used for clipping during optimization")

    # verbose indicators
    argparser.add_argument("--print_all", action="store_true", help="Print t-sne progress during code execution")   

    return argparser


def main():
    print("Run Y = tsne(X, no_dims, perplexity) to perform t-SNE on your dataset.")
    print("Running example on 2,500 MNIST digits...")

    # parses commandline args or set to default
    parser = parse_tsne_args()
    args = parser.parse_args()

    # set the default arg list and compare with CLI input
    default_args = parser.parse_args([])
    new_args = get_non_default_args(args, default_args)

    # inform the user of non-default arguments if any
    if new_args:
        print("Non-default arguments passed via CLI:")
        for key, value in new_args.items():
            print(f"  --{key}={value}")

    # load input and label files
    X = np.loadtxt(args.path_to_X)
    labels = np.loadtxt(args.path_to_labels)

    # preprocess with PCA
    X = pca(X, 50)

    # extract tsne args
    tsne_args = {
        "no_dims": args.no_dims,
        "perplexity": args.perplexity,
        "T": args.T,
        "initial_momentum": args.initial_momentum,
        "final_momentum": args.final_momentum,
        "eta": args.eta,
        "min_gain": args.min_gain,
        "print_all": args.print_all
    }

    # calls tsne()
    Y = tsne(X, **tsne_args)

    # visualization and save
    plt.scatter(Y[:, 0], Y[:, 1], s=20, c=labels, cmap='Paired')
    plt.colorbar()
    plt.savefig("mnist_tsne.png")

if __name__ == "__main__":
    main()
    