# -*- coding: ascii -*-

import numpy as np

from .AbstractDomain import AbstractTabularDomain, Transition


class Ringworld(AbstractTabularDomain):

    LEFT = 0
    RIGHT = 1
    MAX_REWARD = 1
    MAX_GAMMA = 0.99

    def __init__(self, number_of_states, random_generator=np.random):
        assert (number_of_states > 4)
        self.random_generator = random_generator
        self.number_of_states = number_of_states
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
        action = int(self.random_generator.rand() > 0.05)
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

    def expected_return(self, start_state=None):
        if start_state is None:
            current_state = self.current_state
        else:
            current_state = start_state

        # get expected return from rest of episode
        return 2 * (0.05 * (current_state - self.number_of_states) + 0.95 *
                    (current_state + 1)) / (self.number_of_states + 1)
