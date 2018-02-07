# -*- coding: ascii -*-

import numpy as np

from .GTD import GTD


class LGGTD:

    EPS = 1e-3

    def __init__(self, initial_x, initial_gamma, max_reward):
        assert (len(initial_x.shape) == 1)
        n = len(initial_x)
        self._last_gamma = initial_gamma
        self._GTD = GTD(initial_x)
        self.w_err = np.ones(n) * max_reward / (
            max(LGGTD.EPS, 1 - initial_gamma))
        self.w_sq = np.zeros(n)
        self.e_bar = np.zeros(n)
        self.z_bar = np.zeros(n)

    def predict(self, x):
        """Return the current prediction for a given set of features x."""
        return np.dot(self._GTD.w, x)

    def update(self, reward, gamma, x, alpha, eta, rho=1):
        lambda_ = LGGTD.lambda_greedy(
            self._last_gamma, self._GTD._last_x, reward, gamma, x, rho,
            self.w_err, self.w_sq, self._GTD.w, self.e_bar, self.z_bar, alpha)
        self._GTD.update(reward, gamma, x, alpha, eta, lambda_, rho)
        self._last_gamma = gamma
        return lambda_

    @staticmethod
    def lambda_greedy(gamma, x, next_reward, next_gamma, next_x, rho, w_err,
                      w_sq, w, last_e_bar, last_z_bar, alpha):
        # use GTD to update w_err
        next_g_bar = np.dot(next_x, w_err)
        delta = next_reward + next_gamma * next_g_bar - np.dot(x, w_err)
        last_e_bar *= gamma
        last_e_bar += x
        last_e_bar *= rho
        w_err += alpha * delta * last_e_bar

        # use VTD to update w_sq
        next_reward_bar = (rho * next_reward)**2 + 2 * (
            rho**2) * next_gamma * next_reward * next_g_bar
        next_gamma_bar = (rho * next_gamma)**2
        delta_bar = next_reward_bar + next_gamma_bar * np.dot(
            next_x, w_sq) - np.dot(x, w_sq)
        last_z_bar *= next_gamma_bar
        last_z_bar += x
        w_sq += alpha * delta_bar * last_z_bar

        # compute lambda estimate
        errsq = (next_g_bar - np.dot(next_x, w))**2
        varg = max(0, np.dot(next_x, w_sq) - (next_g_bar)**2)
        return errsq / max(LGGTD.EPS, varg + errsq)
