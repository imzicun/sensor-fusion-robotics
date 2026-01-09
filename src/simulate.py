import numpy as np

def simulate_motion(T=12.0, dt=0.01):
    """
    Generate a smooth ground-truth motion: theta(t), omega(t).
    theta = angle, omega = angular velocity.
    """
    t = np.arange(0.0, T, dt)
    theta = 0.8 * np.sin(0.6 * t) + 0.3 * np.sin(1.8 * t)
    omega = np.gradient(theta, dt)
    return t, theta, omega
