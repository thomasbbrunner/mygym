
import numpy as np
import pygame

from envs import Environment


class Pendulum(Environment):
    """
    Pendulum simulation environment.

    Coordinate system:
        angle: positive direction is anti-clockwise,
            with zero position when pendulum is pointing up.

    Observations:
        angle
        angular velocity

    Actions:
        torque
    """

    # Physics parameters
    DT = 0.001
    G = 9.81
    MASS = 1.0
    POLE_LENGTH = 1.0
    MUE = 0.01

    # Ranges
    ANG_RANGE = [-np.pi, np.pi]
    VEL_RANGE = [-2*np.pi, 2*np.pi]
    TORQUE_RANGE = [-5.0, 5.0]

    # Rendering parameters
    RED = (255, 0, 0)
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)
    RADIUS = 20
    WINDOW_HEIGHT = 400
    WINDOW_WIDTH = 400
    CENTER = (WINDOW_WIDTH/2, WINDOW_HEIGHT/2)
    POLE_RENDER_LENGTH = 0.4 * min(WINDOW_WIDTH, WINDOW_HEIGHT)

    @property
    def action_range(self):
        return self.TORQUE_RANGE

    @property
    def observation_range(self):
        return [self.ANG_RANGE, self.VEL_RANGE]

    @property
    def reward_range(self):
        # minimum is when pendulum is down
        # maximum is when pendulum up
        return [
            self._get_reward(self.ANG_RANGE[0]),
            self._get_reward(0.0)]

    def __init__(
            self, init_ang=np.pi, init_vel=0.0,
            num_steps_max=0):

        self._angle = self.check_angle_range(init_ang)
        self._velocity = self.check_velocity_range(init_vel)
        self.num_steps_max = num_steps_max
        self.num_steps = 0

        self.init_obs = (self._angle, self._velocity)

    def reset(self, random=False):

        self.num_steps = 0
        self._angle, self._velocity = self.init_obs

        if random:
            self._angle = self.check_angle_range(
                self.random.uniform(self.ANG_RANGE))

        return (self._angle, self._velocity)

    def step(self, action):

        self.num_steps += 1
        torque = self.check_torque_range(action)
        reward = self._get_reward(self._angle)

        # dynamics equation:
        # m*l**2*acc = -mu*vel + m*g*l*sin(angle) + action
        # acc = -C1*vel + C2*sin(angle) + C3*action
        acceleration = (
            -self.MUE/(self.MASS*self.POLE_LENGTH**2)*self._velocity +
            self.G/(self.POLE_LENGTH)*np.sin(self._angle) +
            1/(self.MASS*self.POLE_LENGTH**2)*torque)

        self._velocity = self.check_velocity_range(
            self._velocity + self.DT*acceleration)
        self._angle = self.check_angle_range(
            self._angle + self.DT*self._velocity + 0.5*self.DT**2*acceleration)

        # Episode terminates if maximum number of steps is reached
        episode_done = self.num_steps >= self.num_steps_max

        return ((self._angle, self._velocity), reward, episode_done)

    def render(self, mode="graphic"):

        if mode == "graphic":

            # Check if rendering was already initialized
            if not pygame.get_init():
                pygame.init()
                self.background = pygame.display.set_mode(
                    (self.WINDOW_WIDTH, self.WINDOW_HEIGHT))

            # avoid unresponsive window
            pygame.event.get()

            # Pygame has coordinates starting on top left corner of the screen
            # with positive directions going right (x) and down (y)
            x = self.CENTER[0] - np.sin(self._angle)*self.POLE_RENDER_LENGTH
            y = self.CENTER[1] - np.cos(self._angle)*self.POLE_RENDER_LENGTH

            # Draw blank page
            self.background.fill(self.WHITE)

            # Draw pendulum
            pygame.draw.lines(
                self.background, self.BLACK, False, [self.CENTER, (x, y)], 2)
            pygame.draw.circle(
                self.background, self.BLACK, (x, y), self.RADIUS)
            pygame.draw.circle(
                self.background, self.RED, (x, y), self.RADIUS - 2)
            pygame.draw.circle(
                self.background, self.BLACK, (self.CENTER[0], self.CENTER[1]), 5)
            pygame.display.update()

        else:
            print(f"ang={self._angle:.2f}, vel={self._velocity:.2f}")

    def close(self):
        pygame.quit()

    def check_torque_range(self, torque):
        """Method checks if torque is an allowed torque range
        """
        return np.clip(torque, *self.TORQUE_RANGE)

    def check_velocity_range(self, velocity):
        """Clips velocity to allowed range.
        Attention: upper bound not included.
        """
        return np.clip(velocity, *self.VEL_RANGE)

    def check_angle_range(self, angle):
        """Method rescales the angle between [-PI, PI)
        Attention: upper bound not included.
        """
        return (angle + np.pi) % (2*np.pi) - np.pi

    def _get_reward(self, angle):
        return -np.abs(angle)
