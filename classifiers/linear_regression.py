""" Bayesian linear regression """

import numpy as np
from numpy import linalg


def add_intercept(X):
    X_new = np.ones((X.shape[0], X.shape[1] + 1), dtype=X.dtype)
    X_new[:, 1:] = X[:, :]
    return X_new


class BayesianLinearRegression:
    """
    Linear regression model: y = z beta[1] + beta[0]
    beta ~ N(0,Lambda)
    Lambda = I * lambda
    P(y|x,beta) ~ N(y|x.dot(beta),sigma**2)
    https://wikimedia.org/api/rest_v1/media/math/render/svg/8131876ebb4646bacc6cf315155856e5d7886483
    """

    def __init__(self, lamb=20., beta_mu=0, sigma=5, fit_intercept=True):
        """
        lamb: variance of the prior for each of the feature dimensions.
        beta_mu: the mean of the prior
        sigma: variance of the prediction error.
        """
        if not np.isscalar(lamb):
            self.inv_lamb = 1. / np.asarray(lamb)
        else:
            self.inv_lamb = 1. / float(lamb)
        if not np.isscalar(beta_mu):
            self.beta_mu = np.asarray(beta_mu)
        else:
            self.beta_mu = float(beta_mu)

        self.sigma = sigma
        self.fit_intercept = fit_intercept
        self.beta = None

    def fit_ml(self, X, y):
        """
          Fit a Maximum Likelihood estimate. (not Bayesian)
          y = mx + b
          X: features, n_samples by n_features nd-array
          y: target values, n_samples array
        """
        if self.fit_intercept:
            X = add_intercept(X)
        self.beta = linalg.inv(X.T.dot(X)).dot(X.T.dot(y))

    def fit_map(self, X, y):
        """
          Fit a MAP estimate
          X: features, n_samples by n_features nd-array
          y: target values, n_samples array
        """
        if self.fit_intercept:
            X = add_intercept(X)
        # data setup
        f_dim = X.shape[1]
        if np.isscalar(self.inv_lamb):
            inv_lamb = np.diagflat(np.repeat(self.inv_lamb, f_dim))
        else:
            inv_lamb = np.diagflat(self.inv_lamb)
        if np.isscalar(self.beta_mu):
            beta_mu = np.repeat(self.beta_mu, f_dim)
        else:
            beta_mu = self.beta_mu

        sigma = self.sigma
        # let the actual calculation begin
        l = sigma ** 2 * inv_lamb
        s = linalg.inv(X.T.dot(X) + l)
        # adding in the mean of the prior
        b0 = sigma ** 2 * inv_lamb.dot(beta_mu)

        self.beta = s.dot(X.T.dot(y) + b0)

    def predict(self, X):
        """ Prediction """
        if self.fit_intercept:
            X = add_intercept(X)
        return X.dot(self.beta)
