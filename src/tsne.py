import numpy as np
import matplotlib.pyplot as plt
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

    # early_clip = p_ij
    pq_diff = P - Q
    
    for row in range(n): #
        y_diff_term = Y[row,:][None,:]-Y # shape (n,d)
        pq_diff_term = pq_diff[row,:,None] # shape (n,1)
        inv_D_term = inv_D[row,:,None] # shape (n,1)

        weights = pq_diff_term * y_diff_term * inv_D_term # shape (n,d)
        dY[row] = np.sum(weights, axis=0) # shape (d,)

    return dY

def tsne(X, no_dims, perplexity, initial_momentum=0.5, final_momentum=0.8, eta=500, min_gain=0.01, T=1000):
    """_summary_

    Parameters
    ----------
    X : _type_
        _description_
    no_dims : _type_
        _description_
    perplexity : _type_
        _description_
    initial_momentum : float, optional
        _description_, by default 0.5
    final_momentum : float, optional
        _description_, by default 0.8
    eta : int, optional
        _description_, by default 500
    min_gain : float, optional
        _description_, by default 0.01
    T : int, optional
        _description_, by default 1000

    Returns
    -------
    _type_
        _description_
    """

    (n, d) = X.shape

    # # preprocessing with pca
    # X_pca = pca(X)

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

        inv_D, q_ij = compute_q_ij(Y)
        dY = compute_gradient(p_ij, q_ij, Y, inv_D)

        kl_divergence = np.sum(p_ij * np.log(p_ij / q_ij))
        if t % 10 == 0:
            # print(f"Iteration {t}: KL divergence = {kl_divergence:.5f}")
            print(f"[{t}] KL: {kl_divergence:.4f}, ||grad||: {np.linalg.norm(dY):.2e}, ||Y||: {np.linalg.norm(Y):.2e}")


        

        momentum = initial_momentum if t < 20 else final_momentum

        gains_02_logic = (dY > 0) != (delta_y > 0)
        gains_08_logic = (dY > 0) == (delta_y > 0)
        gains = (gains+0.2)*gains_02_logic + gains*0.8*gains_08_logic
        
        # clip to min gain value
        gains = np.clip(gains, a_min=min_gain, a_max=np.inf)

        delta_y = momentum*delta_y - eta * (gains*dY)

        Y += delta_y

        if t==100:
            p_ij  /= 4


    return Y


if __name__ == "__main__":
    print("Run Y = tsne(X, no_dims, perplexity) to perform t-SNE on your dataset.")
    print("Running example on 2,500 MNIST digits...")
    X = np.loadtxt("mnist2500_X.txt")
    X = pca(X, 50)
    labels = np.loadtxt("mnist2500_labels.txt")
    Y = tsne(X=X, no_dims=2, perplexity=30, T=1000)
    plt.scatter(Y[:, 0], Y[:, 1], s=20, c=labels, cmap='Paired')
    plt.colorbar()
    plt.savefig("mnist_tsne.png")
