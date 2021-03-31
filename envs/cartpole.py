
import numpy as np
import pygame

from environment import Environment


class Cartpole(Environment):
    """Cartpole simulation environment.

    Can be used for balance (pole starts facing up, has to be balanced) and
    swing-up (pole starts facing down, has to be swing-up) tasks.

    Coordinate system:
        x-axis: positive direction going right
        y-axis: positive direction going up
        angle: positive direction is anti-clockwise,
            with zero position when pendulum is pointing downwards

    Observations:
        position of cart,
        velocity of cart,
        angle of pole, and
        angular velocity of pole.

    Actions:
        apply horizontal force to cart (direction of x-axis).

    Equations of motion and parameters from:
    Appendix C.2, Efficient Reinforcement Learning using
    Gaussian Processes, (Deisenroth, 2010)
    """

    # Physics properties
    L = 0.6    # Length of the pole
    Mc = 0.5   # Mass of the cartpole
    Mp = 0.5   # Mass of the pendulum
    B = 0.1    # Friction coefficients
    G = 9.81   # Gravity constant
    DT = 0.01  # Delta time

    # Enviroment constrains
    POS_RANGE = [-6, 6]
    VEL_RANGE = [-10, 10]
    ANG_RANGE = [-np.pi, np.pi]
    ANG_VEL_RANGE = [-10, 10]
    FORCE_RANGE = [-10, 10]

    # Reward
    # target reward for pendulum in upright position
    # ang = pi and pos = 0
    # target = [x, sin(ang), cos(ang)]
    J_TARGET = np.array([0, 0, -1])
    A = 1
    T_INV = A**2 * np.array([
        [1, L, 0],
        [L, L**2, 0],
        [0, 0, L**2]])

    # RENDERING
    # COLORS
    RED = (255, 0, 0)
    CAR = (122, 122, 0)
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)
    RADIUS = 5
    WINDOW_HEIGHT = 600
    WINDOW_WIDTH = 600
    CENTER = (WINDOW_WIDTH/2, WINDOW_HEIGHT/2)
    POLE_RENDERLENGTH = 0.2 * min(WINDOW_WIDTH, WINDOW_HEIGHT)

    def __init__(
            self, start_pos=0.0, start_vel=0.0,
            start_ang=0.0, start_ang_vel=0.0,
            num_steps_max=0, augmented_obs=False):
        """
        Args:
            num_steps_max: maximum number of steps per episode
                no limit if set to 0
        """

        self.pos, _ = self.check_pos_range(start_pos)
        self.vel = self.check_vel_range(start_vel)
        self.ang = self.check_ang_range(start_ang)
        self.ang_vel = self.check_ang_vel_range(start_ang_vel)

        # start state contains (position, velocity, angle, angular velocity)
        self.start_obs = (self.pos, self.vel, self.ang, self.ang_vel)

        self.num_steps_max = num_steps_max
        self.num_steps = 0
        self.augmented_obs = augmented_obs

    def reset(self, random=False):

        self.num_steps = 0
        self.pos, self.vel, self.ang, self.ang_vel = self.start_obs

        if random:
            self.ang = self.check_ang_range(
                self.random.uniform(self.ANG_RANGE))

        if self.augmented_obs:
            return (self.pos, self.vel, np.sin(self.ang), np.cos(self.ang), self.ang_vel)
        else:
            return (self.pos, self.vel, self.ang, self.ang_vel)

    def step(self, action):

        action = self.check_action_range(action)

        # reward for current state and chosen action
        reward = self._get_reward(self.pos, self.ang)

        # take 10 steps to simulate 0.1 seconds
        for _ in range(10):

            divisor = (
                4*(self.Mc + self.Mp) -
                3*self.Mp*np.cos(self.ang)**2)
            acc = (
                np.sin(self.ang)*(
                    2*self.Mp*self.L*self.ang_vel**2 + 3*self.Mp*self.G*np.cos(self.ang))
                + 4*action - 4*self.B*self.vel) / divisor
            ang_acc = (
                np.sin(self.ang)*(
                    -3*self.Mp*self.L*self.ang_vel**2*np.cos(self.ang) +
                    -6*(self.Mc + self.Mp)*self.G) +
                -6*(action - self.B*self.vel)*np.cos(self.ang)) / (divisor*self.L)

            # Update state
            # Attention: velocity is not clipped according to position.
            # Cart can reach limit of range and stay still, but velocity will
            # stil be non-zero.
            # However, this is not relevant as the episode ends
            # if cart reaches the limits.
            self.ang_vel = self.check_ang_vel_range(
                self.ang_vel + self.DT*ang_acc)
            self.ang = self.check_ang_range(
                self.ang + self.DT*self.ang_vel + 0.5*(self.DT**2)*ang_acc)
            self.vel = self.check_vel_range(
                self.vel + self.DT*acc)
            self.pos, outside_range = self.check_pos_range(
                self.pos + self.DT*self.vel + 0.5*(self.DT**2.0)*acc)

        # Episode terminates if cart reaches limit of allowed range
        # or if training reaches maximum number of steps
        if ((self.num_steps >= self.num_steps_max - 1 and self.num_steps_max != 0)
                or outside_range):
            episode_done = True

        # punitive reward for going beyond allowed range
        # if outside_range:
        #     reward = - (self.num_steps_max - self.num_steps)

        self.num_steps += 1

        if self.augmented_obs:
            obs = (
                self.pos, self.vel, np.sin(self.ang), np.cos(self.ang), self.ang_vel)
        else:
            obs = (
                self.pos, self.vel, self.ang, self.ang_vel)

        return (obs, reward, episode_done)

    def render(self, mode):

        if mode == "graphic":
            # TODO fix this
            # keep disance of 100 to border
            buffer = 100
            half_win_len = (self.WINDOW_WIDTH - buffer)/2

            # Determine cart center
            scaler = (self.POS_RANGE[1] - self.POS_RANGE[0])/2

            x_cart = self.CENTER[0] + half_win_len/scaler*self.pos
            y_cart = self.CENTER[1]

            # Determine the coordinates of the pole end
            # Pygame has coordinates starting on top left corner of the screen
            # with positive directions going right (x) and down (y)
            x_pole_end = x_cart + np.sin(self.ang)*self.POLE_RENDER_LENGTH
            y_pole_end = y_cart + np.cos(self.ang)*self.POLE_RENDER_LENGTH

            # Check if rendering was already initialized
            if self.background is None:
                pygame.init()  # pylint: disable=no-member
                self.background = pygame.display.set_mode(
                    (self.WINDOW_WIDTH, self.WINDOW_HEIGHT))
                self.clock = pygame.time.Clock()

            # make rendering real-time
            self.clock.tick(500)

            # avoid "not-responding" error
            pygame.event.get()

            # Draw blank page
            self.background.fill(self.WHITE)

            # Draw pendulum
            pygame.draw.circle(
                self.background, self.BLACK, (x_pole_end, y_pole_end), self.RADIUS)
            pygame.draw.circle(
                self.background, self.RED, (x_pole_end, y_pole_end), self.RADIUS - 2)
            pygame.draw.circle(
                self.background, self.CAR, (x_cart, y_cart), 15)
            pygame.draw.lines(
                self.background, self.BLACK, False,
                [(x_cart, y_cart), (x_pole_end, y_pole_end)], 2)

            pygame.display.update()
        else:
            print(f"{self.pos=}")

    @staticmethod
    def check_action_range(action):
        """Clips input value to allowed range.
        """
        return np.clip(action, *Cartpole.FORCE_RANGE)

    @staticmethod
    def check_pos_range(pos):
        """Clips input value to allowed range and
        indicates if cart reached the limits of the 
        allowed range.
        """
        outside_range = (
            pos < Cartpole.POS_RANGE[0] or
            pos > Cartpole.POS_RANGE[1])
        return np.clip(pos, *Cartpole.POS_RANGE), outside_range

    @staticmethod
    def check_vel_range(vel):
        """Clips input value to allowed range.
        """
        return np.clip(vel, *Cartpole.VEL_RANGE)

    @staticmethod
    def check_ang_range(ang):
        """Clips input value to allowed range.
        """
        return (ang + np.pi) % (2*np.pi) - np.pi

    @staticmethod
    def check_ang_vel_range(ang_vel):
        """Clips input value to allowed range.
        """
        return np.clip(ang_vel, *Cartpole.ANG_VEL_RANGE)

    @staticmethod
    def _get_reward(pos, ang):
        """Returns reward for current state.
        """
        j = np.array([pos, np.sin(ang), np.cos(ang)])

        j_diff = j - Cartpole.J_TARGET
        reward = -(1 - np.exp(-0.5*(j_diff @ Cartpole.T_INV @ j_diff)))

        return reward
