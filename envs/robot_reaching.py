
import numpy as np

from envs import Environment
import robotsim


class RobotReaching(Environment):
    """Environment for robot's end-effector to reach a specific point in the 
    workspace.

    State:
        joint states
        end-effector position

    Observations:
        joint states
        end-effector position
    
    Actions:
        joint angular velocities
    
    Reward:
        sparse reward (1 if goal position is reached, 0 otherwise)
    """

    DT = 0.01  # delta time between simulation steps

    @property
    def action_range(self):
        return np.repeat([[-np.pi/3, np.pi/3]], self.num_dof, axis=0)

    @property
    def observation_range(self):
        return np.repeat([[-np.pi, np.pi]], self.num_dof, axis=0)

    @property
    def reward_range(self):
        return np.array([0, 1])

    def __init__(self, num_dof, precision, max_num_steps=100):

        self._num_dof = num_dof
        self._precision = precision
        self._num_steps = 0
        self._max_num_steps = max_num_steps
        self._simulation = robotsim.Planar(self.num_dof, np.repeat(1, self.num_dof))

        # state is composed of (joint states, end-effector position)
        self._state = (np.zeros(self._num_dof), self._simulation.forward(np.zeros(self._num_dof)))
    
    def reset(self, random):

        if random:
            state = self.random.uniform(
                self.observation_range[:,0], self.observation_range[:,1], 
                self.observation_range.shape)
            self._state = (state, self._simulation.forward(state))

        else:
            self._state = (np.zeros(self._num_dof), self._simulation.forward(np.zeros(self._num_dof)))
        
        return (np.concatenate((self._state[0], self._state[1])))
    
    def step(self, action):

        self._state[0] = self._state[0] + self.DT*action
        self._state[1] = self._simulation.forward(self._state[0])

        reward = np.linalg.norm(self._goal - self._state[1], 2) <= self._precision
        done = reward or self._num_steps >= self._max_num_steps

        return (np.concatenate((self._state[0], self._state[1])), int(reward), done)

    def render(self):
        pass

    def close(self):
        pass
