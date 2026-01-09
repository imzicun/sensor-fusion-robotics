import numpy as np

def gyro_measurement(true_rate, bias, noise_std):
    """Gyro: true_rate + constant bias + noise."""
    return true_rate + bias + np.random.normal(0.0, noise_std)

def accel_measurement(true_theta, noise_std):
    """Accelerometer: angle with noise."""
    return true_theta + np.random.normal(0.0, noise_std)

def encoder_measurement(true_rate, noise_std):
    """Encoder: rate with noise."""
    return true_rate + np.random.normal(0.0, noise_std)
