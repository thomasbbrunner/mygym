
import numpy as np
import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame

class InvertedPendulum():
    """Inverted pendulum enviroment.
    """

    # parameters
    DT = 0.001
    G = 9.81
    MASS = 1.0
    POLE_LENGTH = 1.0
    MUE = 0.01

    # pre-computed coefficients to speed up calculation
    C1 = MUE/(MASS*POLE_LENGTH*POLE_LENGTH)
    C2 = G/(POLE_LENGTH)
    C3 = 1/(MASS*POLE_LENGTH*POLE_LENGTH)
    DTsd = DT*DT

    # ranges
    TORQUE_RANGE = [-5.0, 5.0]
    VEL_RANGE = [-2*np.pi, 2*np.pi]
    ANGLE_RANGE = [-np.pi, np.pi]

    # rendering colors
    RED = (255, 0, 0)
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)
    RADIUS = 20

    # window properties
    WINDOW_HEIGHT = 400
    WINDOW_WIDTH = 400
    CENTER = (WINDOW_WIDTH/2, WINDOW_HEIGHT/2)
    POLE_RENDER_LENGTH = 0.4 * min(WINDOW_WIDTH, WINDOW_HEIGHT)

    def __init__(
            self, start_angle, start_velocity,
            num_steps_max=500, do_rendering=True):

        self.do_rendering = do_rendering
        self.renderer_initialized = False
        self.start_angle = self.check_angle_range(start_angle)
        self.start_velocity = self.check_velocity_range(start_velocity)
        self.num_steps_max = num_steps_max
        self.num_steps = 0

        # random number generator
        self.random = np.random.default_rng(seed=5555)

        self.reset()

    def reset(self, random=False):
        '''Method that resets the physical components after every episode
        '''
        if random:
            angle = self.random.uniform(*self.ANGLE_RANGE)
            velocity = self.start_velocity
        else:
            angle = self.start_angle
            velocity = self.start_velocity

        self._angle = self.check_angle_range(angle)
        self._velocity = self.check_velocity_range(velocity)

        if self.do_rendering:
            self.render()

        self.num_steps = 0

        return (self._angle, self._velocity)

    def step(self, torque):
        '''Method executes one step in the enviroment.

        Calculations use the right-handed coordinate system.

        Args:
            torque
        Return:
            Next state as (angle, velocity)
        '''

        torque = self.check_torque_range(torque)

        reward = self._get_reward(self._angle)

        # We need to take 100 steps to simulate 0.1 seconds
        for i in range(100):

            # dynamics equation:
            # m*l**2*acc = -mu*vel + m*g*l*sin(angle) + action
            # acc = -C1*vel + C2*sin(angle) + C3*action
            acceleration = (
                -self.C1*self._velocity + self.C2*np.sin(self._angle) + self.C3*torque)

            self._velocity = self.check_velocity_range(
                self._velocity + self.DT*acceleration)
            self._angle = self.check_angle_range(
                self._angle + self.DT*self._velocity + 0.5*self.DTsd*acceleration)

            if not i % 2 and self.do_rendering:
                self.render()

        # Episode terminates if maximum number of steps is reached
        episode_done = self.num_steps >= self.num_steps_max - 1

        self.num_steps += 1

        return reward, [self._angle, self._velocity], episode_done

    def render(self):
        '''Method renders the enviroment.
        '''

        if not self.renderer_initialized:
            pygame.init()
            self.clock = pygame.time.Clock()
            self.background = pygame.display.set_mode(
                (self.WINDOW_WIDTH, self.WINDOW_HEIGHT))

        # make rendering real-time
        self.clock.tick(1000)

        # avoid 'not-responding' error
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

    def check_torque_range(self, torque):
        '''Method checks if torque is an allowed torque range
        '''
        return np.clip(torque, *self.TORQUE_RANGE)

    def check_velocity_range(self, velocity):
        '''Clips velocity to allowed range.
        Attention: upper bound not included.
        '''
        return np.clip(velocity, *self.VEL_RANGE)

    def check_angle_range(self, angle):
        '''Method rescales the angle between [-PI, PI)
        Attention: upper bound not included.
        '''
        return (angle + np.pi) % (2*np.pi) - np.pi

    def get_state(self):
        return (self._angle, self._velocity)

    def _get_reward(self, angle):
        return -np.abs(angle)


if __name__ == "__main__":

    pendulum = InvertedPendulum(np.pi, 0.0, do_rendering=True)

    pendulum.reset(np.pi, 0.0)

    # simulate 10 seconds
    for _ in range(100):
        pendulum.step(5)

    pendulum.step(5)

    # simulate 10 seconds
    for _ in range(100):
        pendulum.step(0)
