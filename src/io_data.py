import csv
import numpy as np

def load_csv(path):
    """
    Load CSV with columns:
      t, gyro, accel_angle, (optional) encoder
    Returns: t, dt, gyro, accel_angle, encoder_array
    """
    rows = []
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    if not rows:
        raise ValueError("CSV is empty")

    def col(name, default=None):
        if name not in rows[0]:
            if default is None:
                raise ValueError(f"Missing required column: {name}")
            return np.array([default] * len(rows), dtype=float)
        return np.array([float(r[name]) for r in rows], dtype=float)

    t = col("t")
    gyro = col("gyro")
    accel_angle = col("accel_angle")
    encoder = col("encoder", default=np.nan)

    if len(t) < 2:
        raise ValueError("Need at least 2 time samples")

    dt = float(np.median(np.diff(t)))
    return t, dt, gyro, accel_angle, encoder
