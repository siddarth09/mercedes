


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

# 📘 MPCC Trajectory Tracker – Mathematical Formulation

This node implements **Model Predictive Contouring Control (MPCC)** for an Ackermann-steered vehicle (e.g., F1TENTH car) using the **CasADi** optimization framework. The controller follows a dynamic reference trajectory by minimizing **contouring** and **lag** errors while respecting vehicle dynamics and input constraints.

---

##  1. State and Control Definitions

### 🔹 Vehicle State Vector

$$
\mathbf{x}_k = \begin{bmatrix}
x_k \\\\ y_k \\\\ \psi_k \\\\ v_k
\end{bmatrix}
$$

Where:

* $x_k, y_k$: Position at step $k$
* $\psi_k$: Heading angle (yaw)
* $v_k$: Longitudinal velocity

---

### 🔹 Control Input Vector

$$
\mathbf{u}_k = \begin{bmatrix}
v_k^c \\\\ \delta_k
\end{bmatrix}
$$

Where:

* $v_k^c$: Commanded velocity
* $\delta_k$: Steering angle

---

##  2. Vehicle Dynamics Model

We use a **kinematic bicycle model**, discretized using forward Euler:

$$
\begin{aligned}
x_{k+1} &= x_k + v_k^c \cdot \cos(\psi_k) \cdot \Delta t \\\\
y_{k+1} &= y_k + v_k^c \cdot \sin(\psi_k) \cdot \Delta t \\\\
\psi_{k+1} &= \psi_k + \frac{v_k^c}{L} \cdot \tan(\delta_k) \cdot \Delta t \\\\
v_{k+1} &= v_k^c
\end{aligned}
$$

Where:

* $L$: Wheelbase of the vehicle
* $\Delta t$: Time step

This model is encoded in CasADi using:

```python
x_next = x + v_c * cos(psi) * dt
y_next = y + v_c * sin(psi) * dt
psi_next = psi + (v_c / L) * tan(delta) * dt
v_next = v_c
```

---

##  3. Contouring and Lag Error Formulation

We track a reference trajectory point $(x_r, y_r, \psi_r)$ at each step and compute:

### 🔹 Contouring Error (Lateral Deviation)

$$
e_c = -\sin(\psi_r)(x - x_r) + \cos(\psi_r)(y - y_r)
$$

### 🔹 Lag Error (Longitudinal Deviation)

$$
e_l = \cos(\psi_r)(x - x_r) + \sin(\psi_r)(y - y_r)
$$

These are derived by projecting the position error vector into the **local path frame**.

---

##  4. Cost Function

The MPCC cost function penalizes:

* Lateral deviation ($e_c$)
* Longitudinal deviation ($e_l$)
* Control effort ($\mathbf{u}$)

$$
J = \sum_{k=0}^{N-1} \left( q_c \cdot e_{c,k}^2 + q_l \cdot e_{l,k}^2 + \mathbf{u}_k^T R \mathbf{u}_k \right)
$$

Where:

* $q_c$: Contouring error weight
* $q_l$: Lag error weight
* $R$: Input cost matrix

---

##  5. Constraints

### 🔹 Dynamics Constraints:

$$
\mathbf{x}_{k+1} = f(\mathbf{x}_k, \mathbf{u}_k)
$$

### 🔹 State & Input Bounds:

$$
\begin{aligned}
v_{\min} &\leq v_k \leq v_{\max} \\\\
\delta_{\min} &\leq \delta_k \leq \delta_{\max}
\end{aligned}
$$

Implemented as CasADi `lbx`, `ubx` vectors in the NLP.

---

##  6. Solver Setup

The problem is solved using **CasADi’s `nlpsol` interface** with the IPOPT solver:

```python
solver = nlpsol('solver', 'ipopt', nlp, {
    'ipopt.print_level': 0,
    'print_time': 0,
    'ipopt.tol': 1e-3,
})
```

Initial guess and bounds are provided for all state and control variables.

---

##  7. Output

The solver returns:

* Optimal steering angle: $\delta_0^*$
* Optimal speed: $v_0^*$

These are sent to the `/drive` topic via `AckermannDriveStamped`.


