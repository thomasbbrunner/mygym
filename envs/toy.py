
import numpy as np

from envs import Environment


class Toy(Environment):
    """Environment for testing of learning and 
    function approximation algorithms.
    """

    ACTIONS_RANGE = [-5, 5]
    POSITION_RANGE = [-50, 50]

    def __init__(self, init_position, goal_position, num_steps_max=100):

        self.init_position = init_position
        self.goal_position = goal_position
        self.num_steps_max = num_steps_max
        self.num_steps = 0
        self.position = self.init_position

        # random number generator
        self.random = np.random.default_rng(seed=5555)

    def reset(self, random=False):

        if random:
            self.position = self.random.integers(
                self.POSITION_RANGE[0], self.POSITION_RANGE[1], endpoint=True)
        else:
            self.position = self.init_position
        self.num_steps = 0

        return self.position

    def step(self, action):

        action = self.check_action_range(action)

        # TODO maybe after state transition?
        reward = self.__get_reward(self.position)

        self.position, outside_range = self.check_pos_range(
            self.position + action)

        # Episode terminates if cart reaches limit of allowed range
        # or if training reaches maximum number of steps
        episode_done = (
            (self.num_steps >= self.num_steps_max - 1)
            or outside_range)

        self.num_steps += 1

        return (reward, self.position, episode_done)

    def get_state(self):
        return self.position

    def __get_reward(self, state):
        return - np.abs(self.goal_position - state)

    def check_pos_range(self, pos):

        outside_range = (
            pos < self.POSITION_RANGE[0] or
            pos > self.POSITION_RANGE[1])

        return (np.clip(pos, *self.POSITION_RANGE), outside_range)

    def check_action_range(self, action):
        """Map actions to [-1, 0, 1] (move left, don't move, more right).
        """
        return np.clip(action, *self.ACTIONS_RANGE)
        # return np.sign(action)

    def get_valid_actions(self):
        pass

    def get_action_space(self):
        pass

    def get_state_space(self):
        pass


if __name__ == "__main__":

    toy = Toy(0, 10, [-50, 50], num_steps_max=100)
    episode_done = False

    while not episode_done:
        print("Step: {}".format(toy.num_steps))

        reward, position, episode_done = toy.step(np.random.rand()-0.5)

        print("  pos: {}".format(position))
        print("  reward: {}".format(reward))
        print("  done: {}".format(episode_done))
