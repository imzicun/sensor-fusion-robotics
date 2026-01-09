import argparse
import numpy as np
import matplotlib.pyplot as plt

from simulate import simulate_motion
from sensors import gyro_measurement, accel_measurement, encoder_measurement
from io_data import load_csv
from complementary import complementary_filter
from kalman import KalmanFilterAngleBias
from fusion import fuse_rate
from metrics import rmse

def run_simulation(args):
    t, theta_true, omega_true = simulate_motion(T=args.T, dt=args.dt)
    dt = t[1] - t[0]

    gyro_bias = args.gyro_bias
    comp_angle = 0.0
    kf = KalmanFilterAngleBias(q_theta=args.q_theta,
                               q_bias=args.q_bias,
                               r_accel=args.r_accel)

    comp_est = []
    kf_est = []

    for i in range(len(t)):
        gyro = gyro_measurement(omega_true[i], gyro_bias, args.gyro_noise)
        accel = accel_measurement(theta_true[i], args.accel_noise)
        enc = encoder_measurement(omega_true[i], args.encoder_noise) if args.use_encoder else np.nan

        rate = fuse_rate(gyro, enc, w_gyro=args.w_gyro)

        comp_angle = complementary_filter(comp_angle, rate, accel, dt, alpha=args.alpha)
        kf.predict(rate, dt)
        kf.update(accel)

        comp_est.append(comp_angle)
        kf_est.append(kf.theta)

    return t, theta_true, np.array(comp_est), np.array(kf_est)

def run_dataset(args):
    t, dt, gyro, accel_angle, encoder = load_csv(args.csv)

    comp_angle = 0.0
    kf = KalmanFilterAngleBias(q_theta=args.q_theta,
                               q_bias=args.q_bias,
                               r_accel=args.r_accel)

    comp_est = []
    kf_est = []

    for i in range(len(t)):
        rate = fuse_rate(gyro[i], encoder[i], w_gyro=args.w_gyro)
        comp_angle = complementary_filter(comp_angle, rate, accel_angle[i], dt, alpha=args.alpha)

        kf.predict(rate, dt)
        kf.update(accel_angle[i])

        comp_est.append(comp_angle)
        kf_est.append(kf.theta)

    # No true angle necessarily in real data
    return t, None, np.array(comp_est), np.array(kf_est)

def plot_and_save(t, truth, comp, kf, out_path):
    plt.figure()
    if truth is not None:
        plt.plot(t, truth, label="True angle")
    plt.plot(t, comp, label="Complementary")
    plt.plot(t, kf, label="Kalman")
    plt.xlabel("Time [s]")
    plt.ylabel("Angle [rad]")
    plt.title("Sensor Fusion: Complementary vs Kalman")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.show()

def main():
    p = argparse.ArgumentParser(description="Sensor fusion demo (Complementary + Kalman).")
    p.add_argument("--mode", choices=["sim", "csv"], default="sim")
    p.add_argument("--csv", type=str, default=None, help="CSV path when using mode=csv")

    # Simulation params
    p.add_argument("--T", type=float, default=12.0)
    p.add_argument("--dt", type=float, default=0.01)
    p.add_argument("--gyro_bias", type=float, default=0.02)
    p.add_argument("--gyro_noise", type=float, default=0.01)
    p.add_argument("--accel_noise", type=float, default=0.05)
    p.add_argument("--encoder_noise", type=float, default=0.02)
    p.add_argument("--use_encoder", action="store_true")

    # Fusion params
    p.add_argument("--alpha", type=float, default=0.98)
    p.add_argument("--w_gyro", type=float, default=0.7)

    # Kalman params
    p.add_argument("--q_theta", type=float, default=1e-3)
    p.add_argument("--q_bias", type=float, default=1e-5)
    p.add_argument("--r_accel", type=float, default=3e-2)

    p.add_argument("--out", type=str, default="results/example_output.png")

    args = p.parse_args()

    if args.mode == "sim":
        t, truth, comp, kf = run_simulation(args)
        print("RMSE (Kalman vs True):", rmse(truth, kf))
        print("RMSE (Complementary vs True):", rmse(truth, comp))
    else:
        if not args.csv:
            raise ValueError("In csv mode, you must provide --csv path/to/file.csv")
        t, truth, comp, kf = run_dataset(args)
        print("CSV mode complete (no ground truth).")

    plot_and_save(t, truth, comp, kf, args.out)

if __name__ == "__main__":
    main()
