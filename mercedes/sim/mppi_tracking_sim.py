#!/usr/bin/env python3

import rclpy
import os
import numpy as np
import tf_transformations
from rclpy.node import Node
from std_msgs.msg import String  
from nav_msgs.msg import OccupancyGrid, Path, Odometry
from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import PoseStamped
from scipy.ndimage import distance_transform_edt as distance_transform
from tf2_ros import TransformException, Buffer, TransformListener
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy
from tf2_ros import Buffer, TransformListener, LookupException, ConnectivityException, ExtrapolationException
from tf_transformations import euler_from_quaternion
from casadi import atan, tan
from rclpy.time import Time
from mercedes.sim.curve import load_trajectory, create_smooth_spline, create_casadi_spline_from_lambda, nearest_s_xy

class MPPI(Node):
    def __init__(self):
        super().__init__('mppi_tracking_sim')  
        
        # Time and horizon parameters
        self.declare_parameter('dt', 0.1)
        self.declare_parameter('N', 30)  # Prediction horizon

        # Vehicle parameters
        self.declare_parameter('L', 0.36)  # Vehicle wheelbase
        self.declare_parameter('max_speed', 1.0)   # Maximum speed (m/s)
        self.declare_parameter('min_speed', 0.3)   # Minimum speed (m/s)
        self.declare_parameter('max_steer', 0.4)   # Maximum steering angle (rad)
        self.declare_parameter('min_steer', -0.4)  # Minimum steering angle (rad)

        # MPPI parameters
        self.declare_parameter('lambda_s', 0.36)       # Temperature parameter
        self.declare_parameter('K', 50)                # Number of sample trajectories
        self.declare_parameter('m', 2)                 # Dimension of control input [v, delta]
        self.declare_parameter('_sigma', [0.1, 0.01])   # Noise covariance for [v, delta]
        self.declare_parameter('safe_distance', 0.3)
        
        # Cost function weights
        self.declare_parameter('w_collision', 20.0)           # Obstacle avoidance weight
        self.declare_parameter('w_curvature', 0.5)           # Control smoothness weight
        self.declare_parameter('w_progress', 0.0)            # Forward progress weight
        self.declare_parameter('w_steering_rate', 10.0)
        self.declare_parameter('out_of_bounds_cost', 100.0) 
        self.declare_parameter('collision_cost', 10.0) 

        # Get parameter values
        self.dt = self.get_parameter('dt').get_parameter_value().double_value
        self.N = self.get_parameter('N').get_parameter_value().integer_value
        self.L = self.get_parameter('L').get_parameter_value().double_value
        self.max_speed = self.get_parameter('max_speed').get_parameter_value().double_value
        self.min_speed = self.get_parameter('min_speed').get_parameter_value().double_value
        self.max_steer = self.get_parameter('max_steer').get_parameter_value().double_value
        self.min_steer = self.get_parameter('min_steer').get_parameter_value().double_value
        self.lambda_s = self.get_parameter('lambda_s').get_parameter_value().double_value
        self.K = self.get_parameter('K').get_parameter_value().integer_value
        self.m = self.get_parameter('m').get_parameter_value().integer_value
        self._sigma = list(self.get_parameter('_sigma').get_parameter_value().double_array_value)
        self.safe_distance = self.get_parameter('safe_distance').get_parameter_value().double_value
        
        # Cost weights
        self.w_collision = self.get_parameter('w_collision').get_parameter_value().double_value
        self.w_progress = self.get_parameter('w_progress').get_parameter_value().double_value
        self.w_curvature = self.get_parameter('w_curvature').get_parameter_value().double_value
        self.w_steering_rate = self.get_parameter('w_steering_rate').get_parameter_value().double_value
        self.out_of_bounds_cost = self.get_parameter('out_of_bounds_cost').get_parameter_value().double_value
        self.collision_cost = self.get_parameter('collision_cost').get_parameter_value().double_value
        
        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            depth=1
        )
        path_qos = QoSProfile(
            depth=10,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE
        )

        # Publishers
        self.drive_pub = self.create_publisher(AckermannDriveStamped, '/drive', 10)
        self.trajectory_pub = self.create_publisher(Path, "/best_rollout", path_qos)
        # Subscribers
        self.map_sub = self.create_subscription(OccupancyGrid, "/map", self.map_callback, qos)
        self.create_subscription(Odometry, '/ego_racecar/odom', self.odom_callback, 10)
        
        self.timer = self.create_timer(self.dt, self.timer_callback)
        
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Initialize class variables
        # self.U = np.zeros((self.N, self.m))
        self.U = np.tile(np.array([0.3, 0.0]), (self.N, 1))
        self.u_min = np.array([self.min_speed, self.min_steer])
        self.u_max = np.array([self.max_speed, self.max_steer])
        self.Sigma = np.diag(self._sigma)  # Noise covariance for [v, delta]
        self.current_vel = 0.0
        self.map_info = None
        self.occupancy_grid = None
        self.x0 = np.zeros(3)
        self.map_frame = 'map'
        self.base_frame = 'ego_racecar/base_link'

        dir = '/home/deepak/Data/f1tenth/mercedes_ws/src/mercedes'
        self.centerline_path = os.path.join(dir, 'storage', 'racelines', 'Spielberg_centerline.csv')

        self.create_timer(self.dt, self.timer_callback)

        x, y = load_trajectory(self.centerline_path)
        self.get_logger().info(f"CSV loaded from {self.centerline_path}")

        self.x_func, self.y_func, self.kappa_func, self.s_eval, self.is_closed, self.closure_info = create_smooth_spline(x, y)
        self.get_logger().info("Lambda functions ready")
        if self.is_closed:
            self.get_logger().info('The curve is closed')

        self.half_width = 0.1
        self.safety_margin = 0.2
        self.ca_splines = create_casadi_spline_from_lambda(self.x_func, self.y_func, self.kappa_func, self.s_eval, self.half_width, self.safety_margin, self.is_closed)
        if self.ca_splines is not None:
            self.get_logger().info("Casadi interpolants ready")

        self.s0 = self.s_eval[0]
        self.L = self.s_eval[-1]

    def odom_callback(self, msg : Odometry):
        x_vel = msg.twist.twist.linear.x
        y_vel = msg.twist.twist.linear.y
        self.current_vel = np.hypot(x_vel, y_vel)

    def update_state_from_tf(self):
        try:
            t = self.tf_buffer.lookup_transform(self.map_frame, self.base_frame, Time())
            trans = t.transform.translation
            rot = t.transform.rotation
            _,_,yaw = euler_from_quaternion([rot.x, rot.y, rot.z, rot.w])
            self.x0 = np.array([trans.x, trans.y, yaw])
        except (LookupException, ConnectivityException, ExtrapolationException) as e:
            self.get_logger().warn(f"TF Lookup failed: {e}")

    def map_callback(self, msg):
        self.map_info = msg.info
        grid = np.array(msg.data).reshape((msg.info.height, msg.info.width))
        binary_map = (grid == 0).astype(np.uint8)
        self.occupancy_grid = distance_transform(binary_map) * msg.info.resolution
        self.get_logger().info(f"Map received with shape {grid.shape}, resolution {msg.info.resolution:.3f}")

    def kinematic_model(self, state:np.ndarray, u:np.ndarray):
        x, y, theta = state
        v, delta = u
        dx = v * np.cos(theta)
        dy = v * np.sin(theta)
        dtheta = (v / self.L) * np.tan(delta)
        return np.array([dx, dy, dtheta])

    def rk4_step(self, state:np.ndarray, u:np.ndarray):
        k1 = self.kinematic_model(state,                    u)
        k2 = self.kinematic_model(state + 0.5*self.dt*k1,   u)
        k3 = self.kinematic_model(state + 0.5*self.dt*k2,   u)
        k4 = self.kinematic_model(state + self.dt*k3,       u)

        next_state = state + (self.dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)
        next_state[2] = np.arctan2(np.sin(next_state[2]), np.cos(next_state[2]))  # wrap yaw
        return next_state
    
    def simulate_trajectory(self, x0, U_seq):
        traj = []
        s = np.array(x0, dtype=float)
        for t in range(self.N):
            s = self.rk4_step(s, U_seq[t])
            traj.append((s[0], s[1], s[2]))
        return traj
    
    def score_traj(self, traj:list, U_seq: np.ndarray):
        score = self.w_collision * self._score_collision(traj)
        score -= self.w_curvature * self._score_curvature(traj)
        score += self.w_progress  * self._score_progress(traj)
        score -= self.w_steering_rate * self._score_steering_rate(U_seq)
        return score

    def _score_collision(self, traj):
        score = 0.0
        min_dist = np.inf

        H, W = self.occupancy_grid.shape
        res = self.map_info.resolution
        ox  = self.map_info.origin.position.x
        oy  = self.map_info.origin.position.y
        
        for x, y, theta in traj:
            mx = int((x - ox) / res)
            my = int((y - oy) / res)
            
            if 10 <= mx < W-10 and 10 <= my < H-10:
                dist = self.occupancy_grid[my, mx]
                min_dist = min(min_dist, dist)
                if min_dist < self.safe_distance:
                    score -= self.collision_cost 
                score += dist
            else: 
                score -= self.out_of_bounds_cost  
        return score

    def _score_curvature(self, traj):
        yaws = [theta for _, _, theta in traj]
        diffs = np.unwrap(np.diff(yaws))
        cost = float(np.sum(np.abs(diffs)))
        return cost                             

    def _score_progress(self, traj):
        
        x0, y0, _ = traj[0]
        s0 = nearest_s_xy(x0, y0, self.s_eval, self.x_func, self.y_func, self.is_closed)
        xN, yN, _ = traj[-1]
        sN = nearest_s_xy(xN, yN, self.s_eval, self.x_func, self.y_func, self.is_closed)
        score = self._wrap_delta_s(sN-s0)
        return score

    # def _score_progress(self, traj):
        
    #     x0, y0, _ = traj[0]
    #     s = nearest_s_xy(x0, y0, self.s_eval, self.x_func, self.y_func, self.is_closed)
    #     progress = 0.0

    #     for (xk, yk, _), (xk1, yk1, _) in zip(traj[:-1], traj[1:]):
    #     # Unit tangent at current s from CasADi
    #         tx = float(self.ca_splines['tx'](s))
    #         ty = float(self.ca_splines['ty'](s))

    #         # Step vector in XY
    #         dx = xk1 - xk
    #         dy = yk1 - yk

    #         # Projection onto tangent ~ along-track increment
    #         ds_proj = dx * tx + dy * ty
    #         progress += ds_proj

    #         # Advance s and wrap/clamp
    #         s += ds_proj
    #         if self.is_closed:
    #             s = self._wrap_s(s)
    #         else:
    #             s = float(np.clip(s, self.s0, self.L))

    #     return float(progress)

    def _wrap_s(self, sq):
        return (sq - self.s0) % self.L + self.s0

    def _score_steering_rate(self, U_seq: np.ndarray):
        deltas = U_seq[:,1]
        diffs = np.diff(deltas)
        cost = np.sum(abs(diffs))
        return cost
    
    def _wrap_delta_s(self, ds):
        if not self.is_closed:
            return ds
        L = float(self.L - self.s0) if self.L > self.s0 else float(self.L)  # total length
        # map ds into (-L/2, L/2]
        while ds >  0.5*L: ds -= L
        while ds <= -0.5*L: ds += L
        return ds

    def publish_drive_command(self, v, delta):
        msg = AckermannDriveStamped()
        v = float(np.clip(v, self.min_speed, self.max_speed))
        delta = float(np.clip(delta, self.min_steer, self.max_steer))
        msg.drive.steering_angle = delta
        msg.drive.speed = v
        self.drive_pub.publish(msg)
        self.get_logger().info(f"Published control: speed={v}, steer={delta}")
        
    def publish_traj(self, traj):
        path = Path()
        path.header.frame_id = self.map_frame
        path.header.stamp = self.get_clock().now().to_msg()

        for x, y, theta in traj:
            pose = PoseStamped()
            pose.header = path.header
            pose.pose.position.x = x
            pose.pose.position.y = y
            q = tf_transformations.quaternion_from_euler(0, 0, theta)
            pose.pose.orientation.x = q[0]
            pose.pose.orientation.y = q[1]
            pose.pose.orientation.z = q[2]
            pose.pose.orientation.w = q[3]
            path.poses.append(pose)

        self.trajectory_pub.publish(path)

    def timer_callback(self):
        self.update_state_from_tf()

        if self.occupancy_grid is None or self.map_info is None:
            self.get_logger().warn("Map not recieved")
            return
        
        if self.x0 is None:
            self.get_logger().warn("TF map->base_link not recieved")

        eps = np.random.multivariate_normal(
            mean = np.zeros(self.m),
            cov = self.Sigma,
            size = (self.K, self.N)
        )

        U_candidates = self.U[None, :, :] + eps
        U_candidates = np.clip(U_candidates, self.u_min, self.u_max)

        scores = np.empty(self.K, dtype=float)
        for k in range(self.K):
            traj = self.simulate_trajectory(self.x0, U_candidates[k])
            score = self.score_traj(traj, U_candidates[k])

            scores[k] = score if np.isfinite(score) else 1e-9
        
        Jmax = np.max(scores)
        w = np.exp((scores - Jmax)/self.lambda_s)
        w_sum = np.sum(w) + 1e-12

        dU = np.tensordot(w, eps, axes=(0,0)) / w_sum
        self.U = np.clip(self.U + dU, self.u_min, self.u_max)

        traj = self.simulate_trajectory(self.x0, self.U)
        self.publish_traj(traj)

        self.U[:-1] = self.U[1:]
        self.U[-1] = 0.0

        u0 = self.U[0]
        v, delta = u0

        self.publish_drive_command(v, delta)

def main(args=None):
    rclpy.init(args=args) 
    node = MPPI()  
    try:
        rclpy.spin(node)  
    except KeyboardInterrupt:
        node.destroy_node()  
        rclpy.shutdown()  


if __name__ == '__main__':
    main()