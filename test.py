
import numpy as np

from envs import Cartpole, Pendulum, Toy


if __name__ == "__main__":

    # Cartpole
    cartpole = Cartpole(
        init_pos=0.0,
        init_ang=np.pi+0.1,)

    for _ in range(10000):
        cartpole.step(0)
        cartpole.render()

    for _ in range(10000):
        cartpole.step(0.5)
        cartpole.render()

    cartpole.close()

    # Pendulum
    pendulum = Pendulum(
        init_ang=0.0+0.1)

    for _ in range(10000):
        pendulum.step(0)
        pendulum.render()

    for _ in range(10000):
        pendulum.step(4)
        pendulum.render()

    pendulum.close()

    # Toy
    toy = Toy(0, 10)

    for _ in range(100):
        toy.step(np.random.rand()-0.5)
        toy.render()
