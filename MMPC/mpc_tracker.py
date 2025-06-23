#!/usr/bin/env python3
import numpy as np
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped
from ackermann_msgs.msg import AckermannDriveStamped
from tf_transformations import euler_from_quaternion

from casadi import SX, vertcat, Function, nlpsol, diag, atan, tan, cos, sin

from tf2_ros import Buffer, TransformListener, LookupException, ConnectivityException, ExtrapolationException
from rclpy.time import Time


class MPCTrajectoryTracker(Node):
    def __init__(self):
        super().__init__('mpc_trajectory_tracker')

        self.declare_parameter('N', 15)
        self.declare_parameter('wheelbase', 0.34)
        self.declare_parameter('State_Weight', [10.0, 10.0, 5.0, 1.0])
        self.declare_parameter('Control_Weight', [0.1, 0.1])
        self.declare_parameter('Terminal_Weight', [1.0, 1.0, 0.5, 1.0])
        self.declare_parameter('v_min', 0.5)
        self.declare_parameter('v_max', 3.0)
        self.declare_parameter('delta_min', -0.5)
        self.declare_parameter('delta_max', 0.5)

        self.N = self.get_parameter('N').get_parameter_value().integer_value
        self.dt = 0.05
        self.L = self.get_parameter('wheelbase').get_parameter_value().double_value
        self.Lr = 0.11

        self.Q = diag(SX(self.get_parameter('State_Weight').get_parameter_value().double_array_value))
        self.R = diag(SX(self.get_parameter('Control_Weight').get_parameter_value().double_array_value))
        self.Q_terminal = diag(SX(self.get_parameter('Terminal_Weight').get_parameter_value().double_array_value))
        self.v_min = self.get_parameter('v_min').get_parameter_value().double_value
        self.v_max = self.get_parameter('v_max').get_parameter_value().double_value
        self.delta_min = self.get_parameter('delta_min').get_parameter_value().double_value
        self.delta_max = self.get_parameter('delta_max').get_parameter_value().double_value

        self.current_state = None
        self.ref_traj = []

        # TF listener for position/orientation
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.map_frame = "map"
        self.base_frame = "ego_racecar/base_link"

        # Velocity from odometry
        self.velocity = 0.0
        self.create_subscription(Odometry, "/ego_racecar/odom", self.odom_callback, 10)

        self.create_subscription(Path, "/dynamic_trajectory", self.traj_callback, 10)
        self.drive_pub = self.create_publisher(AckermannDriveStamped, "/drive", 10)

        self.timer = self.create_timer(self.dt, self.timer_callback)
        self.get_logger().info("CasADi MPC Trajectory Tracker Started")

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

    def traj_callback(self, msg):
        self.ref_traj = []
        for pose in msg.poses:
            pos = pose.pose.position
            ori = pose.pose.orientation
            _, _, yaw = euler_from_quaternion([ori.x, ori.y, ori.z, ori.w])
            self.ref_traj.append(np.array([pos.x, pos.y, yaw]))
        self.ref_traj = self.ref_traj[:self.N + 1]

    def timer_callback(self):
        self.update_state_from_tf()
        if self.current_state is None or len(self.ref_traj) < self.N + 1:
            return

        # Define symbolic variables
        x = SX.sym('x')
        y = SX.sym('y')
        psi = SX.sym('psi')
        v_s = SX.sym('v_s')
        x_state = vertcat(x, y, psi, v_s)

        v_c = SX.sym('v_c')
        delta = SX.sym('delta')
        u_ctrl = vertcat(v_c, delta)

        beta = atan(self.Lr * tan(delta) / self.L)
        x_next = x + v_c * cos(psi + beta) * self.dt
        y_next = y + v_c * sin(psi + beta) * self.dt
        psi_next = psi + (v_c * cos(beta) * tan(delta) / self.L) * self.dt
        v_next = v_c

        next_state = vertcat(x_next, y_next, psi_next, v_next)
        dynamics = Function('dynamics', [x_state, u_ctrl], [next_state])

        X = [SX.sym(f'X_{k}', 4) for k in range(self.N + 1)]
        U = [SX.sym(f'U_{k}', 2) for k in range(self.N)]

        cost = 0
        constraints = [X[0] - self.current_state[:4]]

        for k in range(self.N):
            x_ref, y_ref, yaw_ref = self.ref_traj[k + 1]
            ref_state = np.array([x_ref, y_ref, yaw_ref, 0.5])
            cost += (X[k] - ref_state).T @ self.Q @ (X[k] - ref_state)
            cost += U[k].T @ self.R @ U[k]
            X_next = dynamics(X[k], U[k])
            constraints.append(X[k + 1] - X_next)

        x_ref, y_ref, yaw_ref = self.ref_traj[-1]
        terminal_ref = np.array([x_ref, y_ref, yaw_ref, 0.5])
        cost += (X[self.N] - terminal_ref).T @ self.Q_terminal @ (X[self.N] - terminal_ref)

        z = vertcat(*X, *U)
        g = vertcat(*constraints)

        nlp = {'x': z, 'f': cost, 'g': g}
        solver = nlpsol('solver', 'ipopt', nlp, {
            'ipopt.print_level': 0,
            'print_time': 0,
            'ipopt.tol': 1e-3,
        })

        lbx = []
        ubx = []
        for _ in range(self.N + 1):
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

    def visualize_trajectory(self):
        if not self.ref_traj:
            return

        path_msg = Path()
        path_msg.header.frame_id = "map"
        path_msg.header.stamp = self.get_clock().now().to_msg()

        for point in self.ref_traj:
            pose = PoseStamped()
            pose.header = path_msg.header
            pose.pose.position.x = point[0]
            pose.pose.position.y = point[1]
            pose.pose.orientation.w = 1.0


def main(args=None):
    rclpy.init(args=args)
    node = MPCTrajectoryTracker()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
