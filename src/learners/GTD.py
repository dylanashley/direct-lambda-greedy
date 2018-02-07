# -*- coding: ascii -*-

import numpy as np


class GTD:
    def __init__(self, initial_x):
        assert (len(initial_x.shape) == 1)
        n = len(initial_x)
        self._last_x = np.copy(initial_x)
        self.e = np.zeros(n)
        self.h = np.zeros(n)
        self.w = np.zeros(n)

    def predict(self, x):
        """Return the current prediction for a given set of features x."""
        return np.dot(self.w, x)

    def update(self, reward, gamma, x, alpha, eta, lambda_, rho=1):
        delta = reward + gamma * self.predict(x) - self.predict(self._last_x)
        self.w += alpha * (delta * self.e - gamma *
                           (1 - lambda_) * x * np.dot(self.e, self.h))
        self.h += alpha * eta * (
            delta * self.e - np.dot(self.h, self._last_x) * self._last_x)
        self.e *= lambda_ * gamma
        self.e += x
        self.e *= rho
        np.copyto(self._last_x, x)
