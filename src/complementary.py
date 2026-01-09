def complementary_filter(theta_prev, rate, accel_theta, dt, alpha=0.98):
    """
    Complementary filter:
      predict angle by integrating rate,
      then blend with accelerometer angle.
    """
    pred = theta_prev + rate * dt
    return alpha * pred + (1.0 - alpha) * accel_theta

