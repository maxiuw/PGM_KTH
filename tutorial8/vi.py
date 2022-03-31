# author Olga Mikheeva olgamik@kth.se
# PGM tutorial on Variational Inference
# Bayesian Mixture of Gaussians

from cmath import sqrt
import numpy as np
import matplotlib.pyplot as plt
import math
import warnings


def generate_data(std, k, n, dim=1):
    means = np.random.normal(0.0, std, size=(k, dim))
    data = []
    categories = []
    for i in range(n):
        cat = np.random.choice(k)  # sample component assignment
        categories.append(cat)
        data.append(np.random.multivariate_normal(means[cat, :], np.eye(dim)))  # sample data point from the Gaussian
    return np.stack(data), categories, means


def plot(x, y, c, means, title):
    plt.scatter(x, y, c=c)
    plt.scatter(means[:, 0], means[:, 1], c='r')
    plt.title(title)
    plt.show()


def plot_elbo(elbo):
    plt.plot(range(len(elbo)),elbo)
    plt.title('ELBO')
    plt.show()


def compute_elbo(data, psi, m, s2, sigma2, mu0,  lmbd = 1):
    """ Computes ELBO """
    n, p = data.shape
    k = m.shape[0]

    elbo = 0

    # TODO: compute ELBO
    # expected log prior over mixture assignments
    # summing over k
    # the first 3 terms are added and 4th and 5th terms are subtracted from the elbo
    for i in range(k):
        elbo += - (1/(2*sigma2)) * (s2[i] + m[i].T @
                                    m[i] - m[i].T @ mu0 - mu0.T @ m[i] + mu0.T @ mu0)
    elbo += k * np.log(1/((2*math.pi)**p*sigma2))
    # expected log prior over mixture locations
    elbo += - n * np.log(k)
    # expected log likelihood
    # summing over n and k psi shape (n,k) - (500,5)
    for j in range(n):
        for i in range(k):
            elbo += psi[j, i] * (np.log(1/np.sqrt((2*math.pi)**p * lmbd)) - (1/(2*lmbd**2))
                                 * (data[j].T@data[j] - data[j].T@m[i] - m[i].T@data[j] + m[i].T @ m[i] + s2[i]))
    # elbo += np.sum(psi * (np.math.log(-np.math.sqrt(-2*math.pi * lmbd)) - (1/(2*lmbd)) * data.T@data + data.T@m -m.T@data +m.T @ m +s2))
    # entropy of variational location posterior

    for i in range(k):

        elbo -= np.log(1/np.sqrt(s2[i] * (2*math.pi)**p)) + 1
    # entropy of the variational assignment posterior
    # for j in range(n):
    # print(psi[:,i].reshape((n,)).shape)
    # for i in range(k):
    elbo -= np.sum(psi * np.log(psi))
    print(elbo)
    return elbo


def cavi(data, k, sigma2, m0, eps=1e-15):
    """ Coordinate ascent Variational Inference for Bayesian Mixture of Gaussians
    :param data: data
    :param k: number of components
    :param sigma2: prior variance
    :param m0: prior mean
    :param eps: stopping condition
    :return (m_k, s2_k, psi_i)
    """
    n, p = data.shape # 500, 2 
    # initialize randomly
    m = np.random.normal(0., 1., size=(k, p)) # 5 x 2 
    s2 = np.square(np.random.normal(0., 1., size=(k, 1))) # 5 x 1 
    psi = np.random.dirichlet(np.ones(k), size=n) # 500 x 5 
    lmbd = 1
    # compute ELBO
    elbo = [compute_elbo(data, psi, m, s2, sigma2, m0)]
    convergence = 1.
    step = 0 # forcing to go through at least x steps by setting its value 
    while (convergence > eps) or (step > 0):  # while ELBO fig

        # TODO: update categorical
        for j in range(n):
            for i in range(k):
                psi[j, i] = np.exp((data[j].T @ m[i])/lmbd**2- (m[i].T @ m[i]) / (2 * lmbd**2))
                psi[j] /= np.sum(psi[j])

        # TODO: update posterior parameters for the component means
        s2 = 1/(2 * np.sum(psi,axis=0)/(2 * lmbd**2) + 1/sigma2)
        s2 = s2.reshape((5,1))
        m = s2 *(psi.T@data)/lmbd**2
       
        elbo.append(compute_elbo(data, psi, m, s2, sigma2, m0))
        convergence = elbo[-1] - elbo[-2]
        step -= 1
    return m, s2, psi, elbo


def main():
    # parameters
    p = 2
    k = 5
    sigma = 5.

    data, categories, means = generate_data(std=sigma, k=k, n=500, dim=p)
    m = list()
    s2 = list()
    psi = list()
    elbo = list()
    best_i = 0
    for i in range(10):
        m_i, s2_i, psi_i, elbo_i = cavi(data, k=k, sigma2=sigma, m0=np.zeros(p))
        m.append(m_i)
        s2.append(s2_i)
        psi.append(psi_i)
        elbo.append(elbo_i)
        if i > 0 and elbo[-1][-1] > elbo[best_i][-1]:
            best_i = i
    class_pred = np.argmax(psi[best_i], axis=1)
    fig, ax = plt.subplots(ncols=2)

    ax[0].scatter(data[:, 0], data[:, 1], c=categories)
    ax[0].scatter(means[:, 0], means[:, 1], c='r')
    ax[0].set_title('true data')

    ax[1].scatter(data[:, 0], data[:, 1], c=class_pred)
    ax[1].scatter(m[best_i][:, 0], m[best_i][:, 1], c='r')
    ax[1].set_title('posterior')
    plt.show()


    # plot(, , categories, means, title='true data')
    # plot(data[:, 0], data[:, 1], class_pred, m[best_i], title='posterior')
    plot_elbo(elbo[best_i])

if __name__ == '__main__':
    main()
