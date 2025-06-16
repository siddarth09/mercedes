


# Hierarchical MPC Controller for F1TENTH

This repository implements a **hierarchical Model Predictive Control (MPC)** system for a F1TENTH autonomous racecar. The system is designed to follow a dynamically generated trajectory while satisfying physical constraints and optimizing performance.

---

##  Overview

The control system is built as a two-level hierarchy:

### 1. **High-Level Planner (Dynamic Trajectory Generator)**
- Generates a **time-varying reference trajectory** using a rollout method (e.g., forward integration of a kinematic model).
- Trajectory is published to `/dynamic_trajectory` as a `nav_msgs/Path`.

### 2. **Low-Level Controller (CasADi-based MPC)**
- Uses a **nonlinear kinematic bicycle model** with symbolic optimization in CasADi.
- Optimizes control inputs (`v`, `delta`) over a horizon `N`.
- Tracks the reference trajectory while minimizing tracking error and control effort.
- Outputs control commands to `/drive` as `AckermannDriveStamped`.

---

##  Robot Model

We use a **kinematic bicycle model** with:
- States: `[x, y, ψ, v]`
- Controls: `[v_cmd, δ]` (speed and steering)
- Dynamics include slip angle `β = atan(Lr * tan(δ) / L)`

The motion update:
```

x\_{k+1} = x\_k + v \* cos(ψ + β) \* dt
y\_{k+1} = y\_k + v \* sin(ψ + β) \* dt
ψ\_{k+1} = ψ\_k + (v \* cos(β) \* tan(δ) / L) \* dt
v\_{k+1} = v\_k   (or optionally v\_cmd for variable speed)

````

---

### Topics:

| Topic                  | Type                       | Description                        |
|------------------------|----------------------------|------------------------------------|
| `/ego_racecar/odom`    | `nav_msgs/Odometry`         | Vehicle state (used in MPC)        |
| `/dynamic_trajectory`  | `nav_msgs/Path`             | Reference trajectory (from planner)|
| `/drive`               | `ackermann_msgs/AckermannDriveStamped` | Control output             |

---

##  Parameters

These are declared in the MPC node:

```yaml
mpc_node:
  ros__parameters:
    N: 15                       # Horizon length
    wheelbase: 0.4              # Total wheelbase L
    State_Weight: [10.0, 10.0, 5.0, 1.0]         # [x, y, ψ, v]
    Control_Weight: [5.0, 3.5]                   # [v, δ]
    Terminal_Weight: [5.0, 5.0, 0.5, 1.0]        # [x, y, ψ, v] terminal cost
    v_min: 0.5
    v_max: 3.0
    delta_min: -0.5
    delta_max: 0.5
````

---

## Solver Setup

* Uses **CasADi** with `ipopt` backend for nonlinear optimization.
* Cost function:

  ```
  cost = Σ [(x_k - x_ref_k)' Q (x_k - x_ref_k) + u_k' R u_k] + (x_N - x_ref_N)' Qf (x_N - x_ref_N)
  ```
* Constraints:

  * Dynamics equality constraints
  * Input bounds: `v ∈ [v_min, v_max]`, `δ ∈ [δ_min, δ_max]`

---


## 🚀 Run Instructions

1. Start ROS 2 (if robot):

   ```bash
   ros2 launch f1tenth_stack bringup_launch.py
   ```
   or
    
   ```bash
   ros2 launch f1tenth_stack bringup_launch.py
   ```
 
   

2. Run the MPC controller:

   ```bash
   ros2 launch mercedes mpc.launch.py
   ```

---


