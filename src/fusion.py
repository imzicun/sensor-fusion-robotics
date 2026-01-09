import numpy as np

def fuse_rate(gyro_rate, encoder_rate, w_gyro=0.7):
    """
    Fuse gyro and encoder angular velocity.
    If encoder is missing or NaN, fall back to gyro only.
    """
    if encoder_rate is None:
        return gyro_rate
    if isinstance(encoder_rate, float) and np.isnan(encoder_rate):
        return gyro_rate
    if np.isnan(encoder_rate):
        return gyro_rate
    return w_gyro * gyro_rate + (1.0 - w_gyro) * encoder_rate
