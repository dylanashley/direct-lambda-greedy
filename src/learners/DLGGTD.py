# -*- coding: ascii -*-

import numpy as np

from .GTD import GTD


class DLGGTD:

    EPS = 1e-3

    def __init__(self, initial_x, initial_gamma, max_reward):
        assert (len(initial_x.shape) == 1)
        n = len(initial_x)
        self._last_gamma = initial_gamma
        self._GTD = GTD(initial_x)
        self.w_err = np.ones(n) * max_reward / (
            max(DLGGTD.EPS, 1 - initial_gamma))
        self.w_var = np.zeros(n)
        self.e_bar = np.zeros(n)
        self.z_bar = np.zeros(n)
        self._lambda = 1
        self._gamma_bar = initial_gamma

    def predict(self, x):
        """Return the current prediction for a given set of features x."""
        return np.dot(self._GTD.w, x)

    def update(self, reward, gamma, x, alpha, eta, kappa, rho=1):
        self._lambda, self._gamma_bar = DLGGTD.lambda_greedy(
            self._last_gamma, self._GTD._last_x, reward, gamma, x, rho,
            self.w_err, self.w_var, self._GTD.w, self.e_bar, self.z_bar, alpha,
            self._lambda, self._gamma_bar, kappa)
        self._GTD.update(reward, gamma, x, alpha, eta, self._lambda, rho)

    @staticmethod
    def lambda_greedy(gamma, x, next_reward, next_gamma, next_x, rho, w_err,
                      w_var, w, last_e_bar, last_z_bar, alpha,
                      next_lambda_estimate, gamma_bar, kappa_bar):
        # use GTD to update w_err
        delta_err = next_reward + next_gamma * np.dot(next_x, w_err) - np.dot(
            x, w_err)
        last_e_bar *= gamma
        last_e_bar += x
        last_e_bar *= rho
        w_err += alpha * delta_err * last_e_bar

        # use GTD to update w_var
        delta = next_reward + next_gamma * np.dot(next_x, w) - np.dot(x, w)
        next_reward_bar = delta**2
        next_gamma_bar = (next_gamma * next_lambda_estimate)**2
        delta_bar = next_reward_bar + next_gamma_bar * np.dot(
            next_x, w_var) - np.dot(x, w_var)
        last_z_bar *= kappa_bar * gamma_bar
        last_z_bar += x
        last_z_bar *= rho
        w_var += alpha * delta_bar * last_z_bar

        # compute lambda estimate
        errsq = (np.dot(next_x, w_err) - np.dot(next_x, w))**2
        varg = max(0, np.dot(next_x, w_var))
        return errsq / max(DLGGTD.EPS, varg + errsq), next_gamma_bar
