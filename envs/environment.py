
import abc
import numpy as np


class Environment(abc.ABC):
    """Base class for environments.

    Environments must have the following properties:
        action_space
        observation_space
        reward_range

    Environments must have to implement the methods:
        step
        reset
        render
        seed
        close

    """

    # random number generator
    random = np.random.default_rng(seed=5555)

    @property
    @abc.abstractmethod
    def action_range(self):
        """Range of values of actions as (nested) list(s).
        Example: [[x_min, x_max], [y_min, y_max]].
        """

    @property
    @abc.abstractmethod
    def observation_range(self):
        """Range of values of observations as (nested) list(s).
        Example: [[x_min, x_max], [y_min, y_max]].
        """

    @property
    @abc.abstractmethod
    def reward_range(self):
        """Range of values of the reward as list.
        Example: [min, max].
        """

    @abc.abstractmethod
    def reset(self, random):
        """Resets environment to initial observation.

        Args:
            random: random initial observation.

        Returns:
            observation: current observation of environment.
        """

    @abc.abstractmethod
    def step(self, action):
        """Executes step in environment.

        Args:
            action: action to execute in environment.

        Returns:
            observation: current observation of environment.
            reward: reward of action and observation.
            done: whether episode has ended.
        """

    @abc.abstractmethod
    def render(self, mode):
        """Render the environment.

        Accepts different modes = ["text", "graphic"]

        Graphical rendering should be implemented
        in a separate class.

        Args:
            mode: sets mode of rendering
        """

    def seed(self, seed):
        """Set the seed of the random number generator.

        Args:
            seed: seed of the random number generator
        """

        self.random = np.random.default_rng(seed=seed)

    @abc.abstractmethod
    def close(self):
        """Close and cleanup environment.
        """


class RenderEnvironment(abc.ABC):
    pass
