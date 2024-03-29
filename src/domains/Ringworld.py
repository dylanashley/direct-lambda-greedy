# -*- coding: ascii -*-

import collections
import numpy as np

Transition = collections.namedtuple(
    'Transition', ['last_state', 'action', 'reward', 'gamma', 'state'])


class Ringworld:

    LEFT = 0
    RIGHT = 1
    MAX_REWARD = 1
    MAX_GAMMA = 0.99

    def __init__(self,
                 number_of_states,
                 left_probability=0.05,
                 random_generator=np.random):
        assert (number_of_states > 4)
        self.random_generator = random_generator
        self.number_of_states = number_of_states
        self.left_probability = left_probability
        self._left_state = 0
        if number_of_states % 2:
            self._center_states = [self.number_of_states // 2]
        else:
            self._center_states = [
                self.number_of_states // 2, self.number_of_states // 2 + 1
            ]
        self._right_state = self.number_of_states - 1
        self.current_state = self.random_generator.choice(self._center_states)

    def __next__(self):
        last_state = self.current_state
        action = int(self.random_generator.rand() > self.left_probability)
        state = self.current_state + (2 * action - 1)
        if state < self._left_state:
            state = self.random_generator.choice(self._center_states)
            reward = -1
            gamma = 0
        elif state > self._right_state:
            state = self.random_generator.choice(self._center_states)
            reward = 1
            gamma = 0
        else:
            reward = 0
            gamma = 0.99
        self.current_state = state
        return Transition(last_state, action, reward, gamma, state)

    def expected_return(self, start_state=None, left_probability=None):
        if start_state is None:
            current_state = self.current_state
        else:
            current_state = start_state

        if left_probability is None:
            left_probability = self.left_probability

        # get expected return from rest of episode
        return 2 * (left_probability *
                    (current_state - self.number_of_states) +
                    (1 - left_probability) *
                    (current_state + 1)) / (self.number_of_states + 1)

    def msve(self, learner, left_probability=None):
        if left_probability is None:
            left_probability = self.left_probability

        msve = 0
        for state in range(self.number_of_states):
            x = self.state_to_features(state)
            prediction = learner.predict(x)
            error = prediction - self.expected_return(
                state, left_probability=left_probability)
            msve += (error**2) / self.number_of_states
        return msve

    def state_to_features(self, state):
        rv = np.zeros(self.number_of_states)
        rv[state] = 1
        return rv
