
import numpy as np

from envs import Environment


class Toy(Environment):
    """Environment for testing of learning and
    function approximation algorithms.

    Observations:
        position

    Actions:
        move in positive or negative direction
    """

    ACTION_RANGE = [-5, 5]
    POSITION_RANGE = [-50, 50]

    @property
    def action_range(self):
        return self.ACTION_RANGE

    @property
    def observation_range(self):
        return self.POSITION_RANGE

    @property
    def reward_range(self):
        # reward range depends on goal position
        # get lowest reward possible (furthest from goal)
        return [
            np.minimum(
                self._get_reward(self.POSITION_RANGE[0]),
                self._get_reward(self.POSITION_RANGE[1])),
            0.0]

    def __init__(self, init_position, goal_position, num_steps_max=0):

        self.init_position = init_position
        self.goal_position = goal_position
        self.num_steps_max = num_steps_max
        self.num_steps = 0
        self.position = self.init_position

    def reset(self, random=False):

        if random:
            self.position = self.random.integers(
                self.POSITION_RANGE[0], self.POSITION_RANGE[1], endpoint=True)
        else:
            self.position = self.init_position
        self.num_steps = 0

        return self.position

    def step(self, action):

        self.num_steps += 1
        action = self.check_action_range(action)
        reward = self._get_reward(self.position)

        self.position, outside_range = self.check_pos_range(
            self.position + action)

        # Episode terminates if cart reaches limit of allowed range
        # or if training reaches maximum number of steps
        episode_done = (
            (self.num_steps >= self.num_steps_max)
            or outside_range)

        return (self.position, reward, episode_done)

    def render(self, mode="text"):
        print(f"pos={self.position:.2f}")

    def close(self):
        pass

    def _get_reward(self, state):
        return -np.abs(self.goal_position - state)

    def check_pos_range(self, pos):

        outside_range = (
            pos < self.POSITION_RANGE[0] or
            pos > self.POSITION_RANGE[1])

        return (np.clip(pos, *self.POSITION_RANGE), outside_range)

    def check_action_range(self, action):

        # continuous action
        return np.clip(action, *self.ACTION_RANGE)

        # map actions to [-1, 0, 1] (move left, don't move, more right)
        # return np.sign(action)
