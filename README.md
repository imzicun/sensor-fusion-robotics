# Sensor Fusion for Robotics  
### Comparing Complementary and Kalman Filters for Orientation Estimation

This project implements and compares two sensor fusion techniques – a **Complementary Filter** and a **Kalman Filter** – to estimate the orientation of a robot using noisy measurements from simulated sensors (gyro, accelerometer, and optionally wheel encoder).  

The goal is to demonstrate how sensor fusion improves state estimation under noise, drift, and uncertainty — a key skill in robotics, mechatronics, and embedded systems.

---

## 🚀 Overview

Robots rely on multiple imperfect sensors to estimate their orientation.  
In this project:

- **Gyroscope** gives angular velocity (good short-term, but drifts)
- **Accelerometer** gives absolute angle (very noisy)
- **Encoder** gives an additional velocity estimate (optional)
- **Complementary Filter** blends fast + slow signals
- **Kalman Filter** uses a probabilistic model to estimate angle + gyro bias

This project:

✔ Generates synthetic robot motion  
✔ Simulates sensor noise realistically  
✔ Applies both filters  
✔ Plots and compares their performance  
✔ Computes RMSE (error)  

---

## 🧠 System Model

We estimate:

- **θ** (robot orientation)
- **b** (gyro bias)

State vector:

