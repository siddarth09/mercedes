#!/usr/bin/env python3
import os, math, glob
import numpy as np, pandas as pd, casadi as ca
import rclpy
import tf_transformations

from rclpy.node import Node
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped
from tf2_ros import Buffer, TransformListener, TransformException
from builtin_interfaces.msg import Time
from tf_transformations import euler_from_quaternion
from tf2_ros import Buffer, TransformListener, LookupException, ConnectivityException, ExtrapolationException
from scipy.interpolate import CubicSpline
from casadi import SX, vertcat, Function, nlpsol, diag, atan, tan, cos, sin

# Loads discrete waypoints (x,y) from a CSV
# Generates a smooth, continuous reference trajectory using arc-length-based cubic spline interpolation
# Computes curvature to create a curvature-based velocity profile for speed adaptation in turns
# Builds full reference: position (x, y), heading (phi), and velocity at each prediction step
# Uses a Nonlinear Model Predictive Controller (NMPC) with a kinematic bicycle model for tracking
# Solves the NMPC using multiple shooting and CasADi for real-time trajectory tracking

class MPCTrajNode(Node):

    def __init__(self):
        super().__init__('mpc_path_tracker_sim')

        # ─── ROS PARAMETERS ─────────────────────────────────
        self.declare_parameter('N', 15)           # Horizon                        
        self.declare_parameter('wheelbase', 0.34)
        self.declare_parameter('dt', 0.05)        # Time step
        self.declare_parameter('Lr', 0.11)        # Distance from vehicle COG to rear axle

        self.declare_parameter('State_Weight', [10.0, 10.0, 5.0, 1.0])
        self.declare_parameter('Control_Weight', [0.1, 0.1])
        self.declare_parameter('Terminal_Weight', [1.0, 1.0, 0.5, 1.0])

        self.declare_parameter('v_min', 0.5)
        self.declare_parameter('v_max', 3.0)
        self.declare_parameter('delta_min', -0.5)
        self.declare_parameter('delta_max', 0.5)

        # ─── Load Parameters ─────────────────────────────────
        self.N = self.get_parameter('N').get_parameter_value().integer_value
        self.dt = self.get_parameter('dt').get_parameter_value().double_value
        self.L = self.get_parameter('wheelbase').get_parameter_value().double_value
        self.Lr = self.get_parameter('Lr').get_parameter_value().double_value

        self.Q = diag(SX(self.get_parameter('State_Weight').get_parameter_value().double_array_value))
        self.R = diag(SX(self.get_parameter('Control_Weight').get_parameter_value().double_array_value))
        self.Q_terminal = diag(SX(self.get_parameter('Terminal_Weight').get_parameter_value().double_array_value))

        self.v_min = self.get_parameter('v_min').get_parameter_value().double_value
        self.v_max = self.get_parameter('v_max').get_parameter_value().double_value
        self.delta_min = self.get_parameter('delta_min').get_parameter_value().double_value
        self.delta_max = self.get_parameter('delta_max').get_parameter_value().double_value

        # Subscriptions
        self.create_subscription(Odometry, '/ego_racecar/odom', self.odom_callback, 10)
        # Publishers
        self.drive_pub = self.create_publisher(AckermannDriveStamped, '/drive', 10)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.create_timer(self.dt, self.timer_callback)

        self.base_frame = 'base_link'
        self.map_frame = 'map'
        self.current_state = None
        self.current_vel = None

        self.get_logger().info('Path tracking MPC started')

    def odom_callback(self, msg : Odometry):
        self.current_vel = msg.twist.twist.linear.x

    def update_state_from_tf(self):
        try:
            t = self.tf_buffer.lookup_transform(self.map_frame, self.base_frame, Time())
            trans = t.transform.translation
            rot = t.transform.rotation
            _, _, yaw = euler_from_quaternion([rot.x, rot.y, rot.z, rot.w])
            self.current_state = np.array([trans.x, trans.y, yaw, self.current_vel])
        except (LookupException, ConnectivityException, ExtrapolationException) as e:
            self.get_logger().warn(f"TF Lookup failed: {e}")

    def get_latest_csv(self):
        path = self.get_parameter('csv_path').value
        csv_files = sorted([f for f in os.listdir(path) if f.endswith('.csv')])
        if not csv_files:
            self.get_logger().error("No CSV files found in directory.")
            return
        
        latest_csv = os.path.join(path,csv_files[-1])
        # latest_csv = '/home/deepak/Data/f1tenth/mercedes_ws/src/mercedes/storage/waypoints.csv'
        self.get_logger().info(f"Loading CSV: {latest_csv}")
        self.csv = pd.read_csv(latest_csv)

    def get_ref_traj(self):
        if not hasattr(self, 'csv'):
            self.get_logger().error("CSV file not loaded.")
            return

        x = self.csv['x'].to_numpy()
        y = self.csv['y'].to_numpy()

        # Compute arc length
        ds = np.hypot(np.diff(x), np.diff(y))
        s = np.concatenate(([0], np.cumsum(ds)))

        # Compute heading (yaw)
        dy, dx = np.diff(y), np.diff(x)
        yaw = np.arctan2(dy, dx)
        yaw = np.append(yaw, yaw[-1])

        # Build cubic spline interpolators
        self.x_of_s = CubicSpline(s, x)
        self.y_of_s = CubicSpline(s, y)
        self.yaw_of_s = CubicSpline(s, np.unwrap(yaw))

        # Compute curvature κ(s)
        dx_ds = self.x_of_s.derivative()(s)
        dy_ds = self.y_of_s.derivative()(s)
        d2x_ds2 = self.x_of_s.derivative(nu=2)(s)
        d2y_ds2 = self.y_of_s.derivative(nu=2)(s)

        curvature = np.abs(dx_ds * d2y_ds2 - dy_ds * d2x_ds2) / ((dx_ds ** 2 + dy_ds ** 2) ** 1.5 + 1e-6)

        # Build curvature interpolator
        self.kappa_of_s = CubicSpline(s, curvature)

        # Curvature-based velocity profile
        a_lat_max = 2.0  # max lateral acceleration (tunable)
        v_profile = np.minimum(self.v_max, np.sqrt(a_lat_max / (curvature + 1e-6)))
        self.v_of_s = CubicSpline(s, v_profile)

        # Project current position onto path
        Sgrid = np.linspace(0, s[-1], 1000)
        path_pts = np.vstack((self.x_of_s(Sgrid), self.y_of_s(Sgrid))).T
        idx = int(np.argmin(np.linalg.norm(path_pts - self.current_state[:2], axis=1)))
        s0 = Sgrid[idx]

        # Build reference trajectory using recursive s_k
        ref_traj = []
        s_k = s0
        for k in range(self.N):
            v_ref_k = float(self.v_of_s(s_k))
            s_k = min(s_k + v_ref_k * self.dt, s[-1])
            ref_traj.append([
                float(self.x_of_s(s_k)),
                float(self.y_of_s(s_k)),
                float(self.yaw_of_s(s_k)),
                v_ref_k
            ])

        self.ref_traj = ref_traj

    def timer_callback(self):

        self.update_state_from_tf()
        if self.current_state is None:
            return
        
        x = SX.sym('x')
        y = SX.sym('y')
        phi = SX.sym('phi')
        v_s = SX.sym('v_s')
        x_state = SX.sym(x, y, phi, v_s)

        v_c = SX.sym('v_c')
        delta = SX.sym('delta')
        u_ctrl = vertcat(v_c, delta)

        beta = atan(self.Lr * tan(delta) / self.L)
        x_next = x + v_c * cos(phi + beta) * self.dt
        y_next = y + v_c * sin(phi + beta) * self.dt
        phi_next = phi + (v_c * cos(beta) * tan(delta) / self.L) * self.dt
        v_next = v_c

        next_state = vertcat(x_next, y_next, phi_next, v_next)
        dynamics = Function('dynamics', [x_state, u_ctrl], [next_state])

        X = [SX.sym(f'X_{k}', 4) for k in range(self.N + 1)]
        U = [SX.sym(f'U_{k}', 2) for k in range(self.N)]

        cost = 0
        constraints = [X[0] - self.current_state[:4]]

        for k in range(self.N):

            x_ref, y_ref, phi_ref, vel_ref = self.ref_traj
            ref_state = np.array([x_ref, y_ref, phi_ref, vel_ref])

            cost += (X[k] - ref_state).T @ self.Q @ (X[k] - ref_state)
            cost += U[k].T @ self.R @ U[k]

            X_next = dynamics(X[k], U[k])
            constraints.append(X[k+1] - X_next)

        x_ref, y_ref, yaw_ref, vel_ref = self.ref_traj[-1]
        terminal_ref = np.array([x_ref, y_ref, yaw_ref, vel_ref])
        cost += (X[self.N] - terminal_ref).T @ self.Q_terminal @ (X[self.N] - terminal_ref)

        z = vertcat(*X, *U)
        g = vertcat(*constraints)

        nlp = {'x':z, 'f':cost, 'g':g}
        solver = nlpsol('solver', 'ipopt', nlp, {
            'ipopt.print_level': 0,
            'print_time': 0,
            'ipopt.tol': 1e-3,
        })

        lbx = []
        ubx = []

        for _ in range(self.N+1):
            lbx += [-np.inf, -np.inf, -np.inf, self.v_min]
            ubx += [np.inf, np.inf, np.inf, self.v_max]
        for _ in range(self.N):
            lbx += [self.v_min, self.delta_min]
            ubx += [self.v_max, self.delta_max]

        lbg = [0.0] * ((self.N + 1) * 4)
        ubg = [0.0] * ((self.N + 1) * 4)

        x0 = [*self.current_state[:4]] * (self.N + 1) + [0.5, 0.0] * self.N

        try:
            sol = solver(x0=x0, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)
            z_opt = sol['x'].full().flatten()
            steer = float(z_opt[4 * (self.N + 1) + 1])
            speed = float(z_opt[4 * (self.N + 1)])

            msg = AckermannDriveStamped()
            msg.drive.steering_angle = steer
            msg.drive.speed = speed

            self.drive_pub.publish(msg)
            self.get_logger().info(f"Published control: speed={speed}, steer={steer}")

        except Exception as e:
            self.get_logger().warn(f"MPC Solver failed: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = MPCTrajNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
