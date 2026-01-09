import numpy as np

class KalmanFilterAngleBias:
    """
    State x = [theta, bias]^T
    - Predict using (gyro - bias)
    - Correct using accelerometer angle
    """
    def __init__(self, q_theta=1e-3, q_bias=1e-5, r_accel=3e-2):
        self.x = np.array([[0.0], [0.0]])  # theta, bias
        self.P = np.eye(2) * 0.1

        self.Q = np.diag([q_theta, q_bias])  # process noise
        self.R = np.array([[r_accel]])       # measurement noise
        self.H = np.array([[1.0, 0.0]])      # measure theta only

    def predict(self, gyro, dt):
        # x_k = F x_{k-1} + u, with:
        # theta_k = theta_{k-1} + (gyro - bias)*dt
        F = np.array([[1.0, -dt],
                      [0.0,  1.0]])
        u = np.array([[gyro * dt],
                      [0.0]])

        self.x = F @ self.x + u
        self.P = F @ self.P @ F.T + self.Q

    def update(self, accel_theta):
        z = np.array([[accel_theta]])   # measurement
        y = z - (self.H @ self.x)       # innovation
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y
        self.P = (np.eye(2) - K @ self.H) @ self.P

    @property
    def theta(self):
        return float(self.x[0, 0])

    @property
    def bias(self):
        return float(self.x[1, 0])
