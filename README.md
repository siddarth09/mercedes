# Mercedes - F1/10th Car Project

This repository contains all ROS 2 packages and code for controlling and simulating the **Mercedes** F1/10th autonomous racecar. The project leverages Ackermann steering control, state estimation, and path tracking using algorithms like Pure Pursuit and PID-based wall following. The system is designed for both real-world deployment and simulation using ROS 2 and F1TENTH Gym environments.

---

## 🚗 Project Features

### ✅ ROS 2 Packages

* `pure_pursuit`: Implements dynamic lookahead Pure Pursuit algorithm for path tracking.
* `pid_controller`: Controls wall-following behavior using two LIDAR beams.
* `ekf_filter_node`: Extended Kalman Filter (via `robot_localization`) for fusing odometry and IMU data.
* `joy_teleop`: Manual control via joystick.

### ✅ Core Topics

* `/reference_trajectory`: Path to follow (e.g., from planner or mapped global path).
* `/odometry/filtered`: EKF-fused odometry.
* `/drive`: AckermannDriveStamped output.

---

## 🔧 Parameter Configuration (YAML)

```yaml
pure_pursuit:
  ros__parameters:
    Kdd: 3.0
    min_ld: 0.3
    max_ld: 1.2
    max_steering_angle: 1.0
    wheel_base: 0.325

pid_controller:
  ros__parameters:
    desired_distance: 1.2
    lookahead: 1.0
    theta_deg: 45.0
    Kp: 0.9
    Ki: 0.4
    Kd: 0.09

ekf_filter_node:
  ros__parameters:
    frequency: 30.0
    sensor_timeout: 0.1
    two_d_mode: true
    odom0: "ego_racecar/odom"
    odom0_config: [true, true, false,
                   false, false, true,
                   false, false, false,
                   false, false, false,
                   false, false, false]
    base_link_frame: "ego_racecar/base_link"
    map_frame: "map"
    world_frame: "map"
    publish_tf: true
    publish_acceleration: false
```

---

## 🧠 Algorithms Implemented

### Pure Pursuit (Dynamic Lookahead)

* Dynamically adjusts lookahead based on vehicle speed.
* Computes steering angle using:

  ```math
  \delta = \arctan\left(\frac{2L \sin(\alpha)}{L_d}\right)
  ```
* ROS parameters: `Kdd`, `min_ld`, `max_ld`, `max_steering_angle`, `wheel_base`

### PID Wall Follower

* Uses LIDAR beam projection at angle `theta` to estimate wall distance and angle.
* Applies PID control to maintain `desired_distance`.
* Parameters: `Kp`, `Ki`, `Kd`, `lookahead`, `theta_deg`

---

## 🧪 Testing

* Logged steering angles and path-following error in Pure Pursuit.
* Debug logs display lookahead distances, goal points, and steering values.
* Real-time visualization supported via `rqt_graph`, `rqt_plot`, and simulation tools.

---

## 📦 Running the System

### Using Parameter File:

```bash
ros2 run mercedes pure_pursuit --ros-args --params-file config/params.yaml
```

### With Direct Arguments:

```bash
ros2 run mercedes pure_pursuit --ros-args -p Kdd:=3.0 -p min_ld:=0.3 -p max_ld:=1.2 -p max_steering_angle:=1.0
```

---

## 📍 Future Work

* Implement LQR/MPC for more advanced tracking.
* Integrate perception-driven planning.
* STL-based safety monitoring integration.
* Real-world testing and fine-tuning.

---

## 🧠 Authors

* Siddarth Dayasagar


---

## 📜 License

MIT License.
