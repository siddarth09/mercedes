#!/usr/bin/env python3
import numpy as np
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped
from mercedes_msgs.msg import Trajectory, TrajectoryPoint
from tf_transformations import euler_from_quaternion
from casadi import SX, vertcat, Function, nlpsol, diag, sin, cos, tan
from tf2_ros import Buffer, TransformListener, LookupException, ConnectivityException, ExtrapolationException
from rclpy.time import Time
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Float64


class MPCCTrajectoryTracker(Node):
    def __init__(self):
        super().__init__('mpcc_trajectory_tracker')

        # ---------------- Parameters ----------------
        self.declare_parameter('N', 15)
        self.declare_parameter('wheelbase', 0.34)
        self.declare_parameter('q_c', 20.0)
        self.declare_parameter('q_l', 1.0)
        self.declare_parameter('R_weights', [0.1, 0.1])
        self.declare_parameter('v_min', 0.5)
        self.declare_parameter('v_max', 8.0)
        self.declare_parameter('delta_min', -0.4)
        self.declare_parameter('delta_max', 0.4)
        self.declare_parameter('base_frame', 'ego_racecar/base_link')
        self.declare_parameter('Q_theta', 10.0)

        self.N = self.get_parameter('N').value
        self.dt = 0.05
        self.L = self.get_parameter('wheelbase').value
        self.q_c = self.get_parameter('q_c').value
        self.q_l = self.get_parameter('q_l').value
        self.R = diag(SX(np.array(self.get_parameter('R_weights').value)))
        self.v_min = self.get_parameter('v_min').value
        self.v_max = self.get_parameter('v_max').value
        self.delta_min = self.get_parameter('delta_min').value
        self.delta_max = self.get_parameter('delta_max').value
        self.Q_theta = self.get_parameter('Q_theta').value
        
        # ---------------- State & Trajectory ----------------
        self.current_state = None
        self.ref_centerline = []   # [(x_cl, y_cl, phi_cl), ...]
        self.velocity = 0.0

        # ---------------- ROS Interfaces ----------------
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.map_frame = "map"
        self.base_frame = "ego_racecar/base_link"

        self.create_subscription(Odometry, "/odom", self.odom_callback, 10)
        self.create_subscription(Trajectory, "/racing_trajectory", self.traj_callback, 10)
        self.drive_pub = self.create_publisher(AckermannDriveStamped, "/drive", 10)
        self.marker_pub = self.create_publisher(MarkerArray, "/mpcc_predicted_horizon", 10)
        self.timer = self.create_timer(self.dt, self.timer_callback)
        
        self.theta_pub = self.create_publisher(Float64, "/mpcc_progress", 10)
        self.get_logger().info(" MPCC Tracker Started and waiting for /racing_trajectory")

    # ---------------- Odometry ----------------
   
    def odom_callback(self, msg):
        self.velocity = msg.twist.twist.linear.x

    def update_state_from_tf(self):
        try:
            t = self.tf_buffer.lookup_transform(self.map_frame, self.base_frame, Time())
            trans = t.transform.translation
            rot = t.transform.rotation
            _, _, yaw = euler_from_quaternion([rot.x, rot.y, rot.z, rot.w])
            self.current_state = np.array([trans.x, trans.y, yaw, self.velocity])
        except (LookupException, ConnectivityException, ExtrapolationException) as e:
            self.get_logger().warn(f"TF Lookup failed: {e}")

    # ---------------- Reference Trajectory ----------------
    def traj_callback(self, msg: Trajectory):
        """Store the full reference racing line with arc-length."""
        self.ref_centerline = []
        for p in msg.points:
            self.ref_centerline.append(np.array([p.s, p.x, p.y, p.yaw]))
        self.get_logger().info(f"Received full racing trajectory with {len(self.ref_centerline)} points.")


    # ---------------- Error Computation (Eq. 8a–b) ----------------
    def compute_contour_lag_error(self, x, y, psi, ref):
        """
        Compute linearized contouring and lag errors as per Eq. (8a–b),
        including arc-length (s) for MPCC path progression.
        """
        s_cl, x_cl, y_cl, phi_cl = ref

        # Local position errors relative to the path centerline
        dx = x - x_cl
        dy = y - y_cl

        # Linearized contour and lag errors (Eq. 8a, 8b)
        eps_cont = sin(phi_cl) * dx - cos(phi_cl) * dy
        eps_lag  = -cos(phi_cl) * dx - sin(phi_cl) * dy

        return eps_cont, eps_lag, s_cl


    # ---------------- MPC Loop ----------------
    def timer_callback(self):
        """Main MPCC optimization loop."""
        # --- 1. Update current state ---
        self.update_state_from_tf()
        if self.current_state is None or len(self.ref_centerline) < self.N + 1:
            return

        # --- 2. Find nearest reference point along the centerline ---
        dists = [np.linalg.norm(self.current_state[:2] - ref[1:3]) for ref in self.ref_centerline]
        nearest_idx = int(np.argmin(dists))
        theta_s = self.ref_centerline[nearest_idx][0]   # starting arc-length (s)
        ref_window = self.ref_centerline[nearest_idx : nearest_idx + self.N + 1]
        if len(ref_window) < self.N + 1:
            return

        # --- 3. Define CasADi symbolic model ---
        x, y, psi, v, theta = SX.sym('x'), SX.sym('y'), SX.sym('psi'), SX.sym('v'), SX.sym('theta')
        x_state = vertcat(x, y, psi, v, theta)
        v_c, delta = SX.sym('v_c'), SX.sym('delta')
        u_ctrl = vertcat(v_c, delta)

        # dynamics (Eq. 7b–7c)
        x_next     = x + v_c * cos(psi) * self.dt
        y_next     = y + v_c * sin(psi) * self.dt
        psi_next   = psi + (v_c / self.L) * tan(delta) * self.dt
        v_next     = v_c
        theta_next = theta + v_c * self.dt              # θₖ₊₁ = θₖ + vₖ Δt
        next_state = vertcat(x_next, y_next, psi_next, v_next, theta_next)
        dynamics   = Function('dynamics', [x_state, u_ctrl], [next_state])

        # --- 4. Optimization variables ---
        X = [SX.sym(f'X_{k}', 5) for k in range(self.N + 1)]
        U = [SX.sym(f'U_{k}', 2) for k in range(self.N)]

        cost = 0
        constraints = [X[0] - vertcat(*self.current_state, theta_s)]

        # --- 5. MPCC cost and constraints (Eqs 7a–7f) ---
        for k in range(self.N):
            ref = ref_window[k + 1]
            s_cl, x_cl, y_cl, phi_cl = ref

            # contouring / lag error (Eqs 8a–8b)
            dx = X[k][0] - x_cl
            dy = X[k][1] - y_cl
            eps_c =  sin(phi_cl) * dx - cos(phi_cl) * dy
            eps_l = -cos(phi_cl) * dx - sin(phi_cl) * dy

            # cost: tracking + progress + smooth control
            cost += self.q_c * eps_c**2 + self.q_l * eps_l**2
            cost -= self.Q_theta * (X[k][4] - s_cl)         # encourage θ → s_cl
            cost += U[k].T @ self.R @ U[k]
            if k > 0:
                du = U[k] - U[k - 1]
                cost += du.T @ self.R @ du

            # dynamics equality constraint
            X_next = dynamics(X[k], U[k])
            constraints.append(X[k + 1] - X_next)

        # --- 6. Build NLP and solver ---
        z = vertcat(*X, *U)
        g = vertcat(*constraints)
        nlp = {'x': z, 'f': cost, 'g': g}

        solver = nlpsol('solver', 'ipopt', nlp, {
            'ipopt.print_level': 0,
            'ipopt.sb': 'yes',
            'print_time': False,
            'ipopt.tol': 1e-3,
        })

        # --- 7. Bounds per (7d–7f) ---
        lbx, ubx = [], []

        # state bounds: (x, y, ψ, v, θ)
        for _ in range(self.N + 1):
            lbx.extend([-np.inf, -np.inf, -np.inf, 0.0, theta_s])  # v∈[0, 8], θ∈[θₛ, 0]
            ubx.extend([ np.inf,  np.inf,  np.inf, 8.0, 0.0])

        # control bounds: (v_c, δ)
        for _ in range(self.N):
            lbx.extend([0.0,  -0.4])
            ubx.extend([8.0,   0.4])

        # equality constraint bounds (g = 0)
        lbg = [0.0] * (len(constraints) * 5)
        ubg = [0.0] * (len(constraints) * 5)

        # initial guess
        x0 = [*self.current_state, theta_s] * (self.N + 1) + [self.velocity, 0.0] * self.N

        # --- 8. Solve NLP ---
        try:
            sol = solver(x0=x0, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)
            z_opt = sol['x'].full().flatten()

            # extract first control action
            steer = float(z_opt[5 * (self.N + 1) + 1])
            speed = float(z_opt[5 * (self.N + 1)])

            # --- 9. Publish control ---
            msg = AckermannDriveStamped()
            msg.drive.steering_angle = np.clip(steer, -0.4, 0.4)
            msg.drive.speed = np.clip(speed, 0.0, 8.0)
            self.drive_pub.publish(msg)

            # --- 10. Publish predicted horizon and θ progress ---
            self.publish_predicted_horizon(z_opt)
            theta_final = float(z_opt[5 * self.N + 4])
            theta_msg = Float64()
            theta_msg.data = theta_final
            self.theta_pub.publish(theta_msg)

            self.get_logger().info(
                f"[MPCC] speed={speed:.2f} m/s, steer={steer:.3f} rad, θ={theta_final:.2f}"
            )

        except Exception as e:
            self.get_logger().warn(f"MPCC solver failed: {e}")


    def publish_predicted_horizon(self, z_opt):
        """Visualize predicted MPCC horizon as green spheres in RViz."""
        markers = MarkerArray()
        time_now = self.get_clock().now().to_msg()

        for k in range(self.N + 1):
            # Each state has [x, y, psi, v, theta]
            idx = 5 * k
            m = Marker()
            m.header.frame_id = self.map_frame
            m.header.stamp = time_now
            m.ns = "mpcc_predicted"
            m.id = k
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.scale.x = 0.2
            m.scale.y = 0.2
            m.scale.z = 0.2
            m.color.a = 1.0
            m.color.r = 0.1
            m.color.g = 0.9
            m.color.b = 0.1
            m.pose.position.x = float(z_opt[idx])
            m.pose.position.y = float(z_opt[idx + 1])
            m.pose.position.z = 0.05
            markers.markers.append(m)

        # Clear old markers beyond current horizon
        for k in range(self.N + 1, self.N + 5):
            m = Marker()
            m.header.frame_id = self.map_frame
            m.header.stamp = time_now
            m.ns = "mpcc_predicted"
            m.id = k
            m.action = Marker.DELETE
            markers.markers.append(m)

        self.marker_pub.publish(markers)


def main(args=None):
    rclpy.init(args=args)
    node = MPCCTrajectoryTracker()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
