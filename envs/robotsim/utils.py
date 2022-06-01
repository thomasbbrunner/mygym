
import numpy as np


def dh_transformation(alpha, a, d, theta, squeeze=True):
    """
    Returns transformation matrix between two frames
    according to the Denavit-Hartenberg convention presented
    in 'Introduction to Robotics' by Craig.

    Also accepts batch processing of several joint states (d or theta).

    Transformation from frame i to frame i-1:
    alpha:  alpha_{i-1}
    a:      a_{i-1}
    d:      d_i     (variable in prismatic joints)
    theta:  theta_i (variable in revolute joints)
    """

    d = np.atleast_1d(d)
    theta = np.atleast_1d(theta)

    if d.shape[0] > 1 and theta.shape[0] > 1:
        raise RuntimeError(
            "Only one variable joint state is allowed.")

    desired_shape = np.maximum(d.shape[0], theta.shape[0])

    alpha = np.resize(alpha, desired_shape)
    a = np.resize(a, desired_shape)
    d = np.resize(d, desired_shape)
    theta = np.resize(theta, desired_shape)
    zeros = np.zeros(desired_shape)
    ones = np.ones(desired_shape)

    sin = np.sin
    cos = np.cos
    th = theta
    al = alpha

    transformation = np.array([
        [cos(th),           -sin(th),           zeros,          a],
        [sin(th)*cos(al),   cos(th) * cos(al),  -sin(al),   -sin(al)*d],
        [sin(th)*sin(al),   cos(th) * sin(al),  cos(al),    cos(al)*d],
        [zeros,             zeros,              zeros,      ones]
    ])

    # fix dimensions
    transformation = np.rollaxis(transformation, 2)

    if squeeze:
        transformation = np.squeeze(transformation)

    return transformation


def wrap(angles):
    """Wraps angles to [-pi, pi) range."""
    return (angles + np.pi) % (2*np.pi) - np.pi
