##  Mathematical Overview

This section describes the core math behind the **minimum curvature trajectory generation** used in the `raceline_optimizer` module.
The goal is to compute a smooth, dynamically feasible **racing line** from an occupancy grid map.

---

### 1️⃣  **Map to Centerline Extraction**

Given a binary occupancy grid map ( M(x, y) ), the first step is to extract the **track boundaries** and compute a **geometric centerline**.

1. **Binary thresholding and contour extraction:**
   The outer and inner walls of the track are extracted using OpenCV’s contour detection.
   [
   \text{walls} = {C_\text{outer}, C_\text{inner}} = \text{Contours}(M)
   ]

2. **Centerline computation:**
   Each point along the centerline is taken as the midpoint between the corresponding points on the two walls:
   [
   P_i = \frac{1}{2}\big(C_{\text{outer},i} + C_{\text{inner},i}\big)
   ]

3. **Spline smoothing:**
   A cubic B-spline is fit through all center points to obtain a smooth, differentiable path:
   [
   \mathbf{r}_c(s) = (x_c(s), y_c(s))
   ]
   where ( s ) is the arc length parameter.

This centerline serves as the **initial geometric reference** for optimization.

---

### 2️⃣  **Minimum Curvature Optimization**

We now refine the geometric centerline into the **optimal racing line**, minimizing the total curvature of the trajectory while staying within track boundaries.

#### Path parameterization:

Each point on the raceline is modeled as a small lateral offset (\delta(s)) along the local normal vector of the centerline:
[
\mathbf{r}(s) = \mathbf{r}_c(s) + \delta(s),\mathbf{n}(s)
]
where
[
\mathbf{n}(s) =
\frac{1}{|\mathbf{r}_c'(s)|}
\begin{bmatrix}
-,y_c'(s) [3pt]
x_c'(s)
\end{bmatrix}
]
is the unit normal to the centerline.

---

#### Curvature of the offset path:

For any planar curve ((x(s), y(s))), curvature is defined as:
[
\kappa(s) =
\frac{x'(s) y''(s) - y'(s) x''(s)}
{\big(x'(s)^2 + y'(s)^2\big)^{3/2}}
]

Substituting the offset form (\mathbf{r}(s)), curvature becomes a nonlinear function of (\delta(s)) and its derivatives.

---

#### Optimization objective:

We minimize the **integrated squared curvature** (smoothest path) while enforcing lateral constraints:
[
J(\delta) =
\int \kappa(s)^2,ds
\quad\text{s.t.}\quad
|\delta(s)| \leq w_{\text{max}}
]

In discrete form (using samples ( s_i )):
[
J(\delta) \approx
\sum_i \kappa_i^2

* \lambda_1 \sum_i (\delta''_i)^2
* \lambda_2 \sum_i \max(0, |\delta_i| - w_{\text{max}})^2
  ]

where:

* The **first term** penalizes curvature,
* The **second term** enforces **smoothness** in lateral offset,
* The **third term** keeps the path **within track limits** via a soft constraint.

---

### 3️⃣  **JAX-Based Gradient Optimization**

Instead of solving a symbolic QP, the objective (J(\delta)) is minimized directly using **JAX autodifferentiation**:

1. Compute gradients (\nabla_\delta J) via automatic differentiation.
2. Iteratively update (\delta \leftarrow \delta - \eta \nabla_\delta J) using the **Adam optimizer** with a decaying learning rate.
3. Apply clipping or projection to maintain (|\delta| \le w_{\text{max}}).

The result is a smooth, curvature-minimized path:
[
\mathbf{r}^*(s) = \mathbf{r}_c(s) + \delta^*(s),\mathbf{n}(s)
]

---

### 4️⃣  **Trajectory Output**

The optimized racing line is exported as:

```
out/racing_line_world.csv
```

with columns:

```
s_m, x_m, y_m, kappa
```

This can be directly published as a `nav_msgs/Path` message in ROS 2 and used by your controller (MPC, Pure Pursuit, or Stanley) as the **reference trajectory**.

---

### 🧩 Summary of the Full Pipeline

| Stage                          | Input             | Output                     | Tool             |
| ------------------------------ | ----------------- | -------------------------- | ---------------- |
| Map Preprocessing              | `.pgm`, `.yaml`   | Binary mask of track       | OpenCV           |
| Boundary Extraction            | Mask              | Inner/Outer contours       | cv2.findContours |
| Centerline Smoothing           | Contours          | Cubic spline centerline    | SciPy splprep    |
| Minimum-Curvature Optimization | Centerline spline | Optimal raceline           | JAX + Optax      |
| Export & Visualization         | Optimized path    | `.csv` + `/reference_path` | ROS 2 RViz       |



> NOTE: Please make sure you clone the repo and go inside mercedes package and follow the below steps to extract the optimized trajectory 


```bash 

cd mercedes/raceline_optimizer

python3 /home/siddarth/f1ws/src/mercedes/raceline_optimizer/centerline_extractor.py   --map /home/siddarth/f1ws/src/mercedes/maps/csc433_track.pgm   --yaml /home/siddarth/f1ws/src/mercedes/maps/csc433_track.yaml   --out /home/siddarth/f1ws/src/mercedes/raceline_optimizer/out   --show


python3 /home/siddarth/f1ws/src/mercedes/raceline_optimizer/min_curvature_optimizer.py

``` 

