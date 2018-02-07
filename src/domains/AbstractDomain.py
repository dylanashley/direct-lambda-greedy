# -*- coding: ascii -*-

import collections
import numpy as np

Transition = collections.namedtuple(
    'Transition', ['last_state', 'action', 'reward', 'gamma', 'state'])


class AbstractDomain:
    def __next__(self):
        raise NotImplementedError()

    def rmse(self, learner):
        raise NotImplementedError()

    def expected_return(self, start_state=None):
        raise NotImplementedError()

    def state_to_features(self, state):
        raise NotImplementedError()


class AbstractTabularDomain(AbstractDomain):
    def rmse(self, learner):
        mse = 0
        for state in range(self.number_of_states):
            x = self.state_to_features(state)
            prediction = learner.predict(x)
            error = prediction - self.expected_return(state)
            mse += (error**2) / self.number_of_states
        return np.sqrt(mse)

    def state_to_features(self, state):
        rv = np.zeros(self.number_of_states)
        rv[state] = 1
        return rv
