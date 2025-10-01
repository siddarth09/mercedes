#!/usr/bin/env python3
import os
import csv
from math import sin, cos
import rclpy
import casadi as ca
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
import csv
from io import StringIO
import pandas as pd
from mercedes.sim.curve import load_trajectory, create_smooth_spline, create_casadi_spline_from_lambda, nearest_s_xy

class RacelineVisualizer(Node):
    def __init__(self):
        super().__init__('raceline_visualizer')

        # ---- Parameters ----
        self.declare_parameter('csv_dir', '/home/deepak/Data/f1tenth/mercedes_ws/src/mercedes/storage/racelines')
        self.declare_parameter('raceline_csv', 'Spielberg_raceline.csv')
        self.declare_parameter('centerline_csv', 'Spielberg_centerline.csv')
        self.declare_parameter('frame_id', 'map')
        self.declare_parameter('half_width', 0.5)
        self.declare_parameter('safety_margin', 0.0)

        csv_dir     = self.get_parameter('csv_dir').get_parameter_value().string_value
        raceline_csv   = self.get_parameter('raceline_csv').get_parameter_value().string_value
        centerline_csv    = self.get_parameter('centerline_csv').get_parameter_value().string_value
        self.frame  = self.get_parameter('frame_id').get_parameter_value().string_value
        self.half_width = self.get_parameter('half_width').get_parameter_value().double_value
        self.safety_margin = self.get_parameter('safety_margin').get_parameter_value().double_value

        # ---- QoS: latched so RViz can latch the Path ----
        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.pub_raceline = self.create_publisher(Path, '/raceline', qos)
        self.pub_centerline = self.create_publisher(Path, '/centerline', qos)
        self.s_curve_pub = self.create_publisher(Path, '/s_curve', qos)
        self.casadi_curve_pub = self.create_publisher(Path, '/casadi_curve', qos)

        # Build and publish
        raceline_path = os.path.join(csv_dir, raceline_csv)
        path_msg_raceline = self._load_raceline_csv(raceline_path, self.frame)
        self.pub_raceline.publish(path_msg_raceline)
        self.get_logger().info(f"Published raceline with {len(path_msg_raceline.poses)} poses on '/raceline' (frame_id='{self.frame}').")

        # centerline_path = os.path.join(csv_dir, centerline_csv)
        centerline_path = '/home/deepak/Data/f1tenth/mercedes_ws/src/mercedes/storage/wp-2025-09-24-15-48-56.csv'
        path_msg_centerline = self._load_centerline_csv(centerline_path, self.frame)
        self.pub_centerline.publish(path_msg_centerline)
        self.get_logger().info(f"Published raceline with {len(path_msg_centerline.poses)} poses on '/centerline' (frame_id='{self.frame}').")

        reference_path = '/home/deepak/Data/f1tenth/mercedes_ws/src/mercedes/storage/wp-2025-09-24-15-48-56.csv'
        # reference_path = centerline_path
        x, y = load_trajectory(reference_path)
        self.get_logger().info(f"CSV loaded form {reference_path}")

        self.x_func, self.y_func, self.kappa_func, self.s_eval, self.is_closed, self.closure_info = create_smooth_spline(x, y)
        self.get_logger().info("Lambda functions ready")
        if self.is_closed:
            self.get_logger().info('The curve is closed')

        self.s0 = self.s_eval[0]
        self.L = self.s_eval[-1]

        self.ca_splines = create_casadi_spline_from_lambda(self.x_func, self.y_func, self.kappa_func, self.s_eval, self.half_width, self.safety_margin, self.is_closed)
        if self.ca_splines is not None:
            self.get_logger().info("Casadi interpolants ready")

        self.publish_smooth_curve(self.s_eval)
        self.publish_casadi_curve(self.s_eval)
    
        # Give DDS a moment to deliver, then shut down
        self.create_timer(1.0, self._shutdown_after_publish)


    def publish_smooth_curve(self, s_eval: list):
        path = Path()
        path.header.frame_id = 'map'

        for s in s_eval:
            sw = self._wrap_s(s)
            ps = PoseStamped()
            ps.header.frame_id = 'map'
            ps.pose.position.x = float(self.x_func(sw))
            ps.pose.position.y = float(self.y_func(sw))
            ps.pose.position.z = 0.0
            ps.pose.orientation.w = 1.0 
            path.poses.append(ps)

        self.s_curve_pub.publish(path)
        self.get_logger().info(f"s_curve published with {len(self.s_eval)} points")


    def publish_casadi_curve(self, s_eval: list):
        path = Path()
        path.header.frame_id = 'map'

        x_ca = self.ca_splines['x']
        y_ca = self.ca_splines['y']
        psi_ca = self.ca_splines['psi']

        for s in s_eval:
            sw = self._wrap_s_sym(s)
            ps = PoseStamped()
            ps.header.frame_id = 'map'
            ps.pose.position.x = float(x_ca(sw))
            ps.pose.position.y = float(y_ca(sw))
            ps.pose.position.z = 0.0

            yaw = float(psi_ca(sw))
            half_yaw = 0.5 * yaw
            ps.pose.orientation.x = 0.0
            ps.pose.orientation.y = 0.0
            ps.pose.orientation.z = sin(half_yaw)
            ps.pose.orientation.w = cos(half_yaw)
            path.poses.append(ps)

        self.casadi_curve_pub.publish(path)
        self.get_logger().info(f"casadi_curve published with {len(self.s_eval)} points")

    def _wrap_s(self, sq):
        return (sq - self.s0) % self.L + self.s0
    
    def _wrap_s_sym(self, s_sym):
        return (s_sym - self.s0) - ca.floor((s_sym - self.s0) / self.L) * self.L + self.s0

    def _shutdown_after_publish(self):
        self.get_logger().info("Done. Shutting down.")
        rclpy.shutdown()

    @staticmethod
    def _load_raceline_csv(csv_path: str, frame_id: str) -> Path:
        path = Path()
        path.header.frame_id = frame_id

        if not csv_path or not os.path.exists(csv_path):
            raise FileNotFoundError(f"Raceline CSV file not found: {csv_path}")
        
        df = pd.read_csv(csv_path, 
                delimiter=';', 
                comment='#',   # This tells pandas to skip lines starting with #
                header=0,      # Use first non-comment line as header
                names=['s_m', 'x_m', 'y_m', 'psi_rad', 'kappa_radpm', 'vx_mps', 'ax_mps2'])

        x = df['x_m'].values
        y = df['y_m'].values
        yaw = df['psi_rad'].values
    
        for i in range(len(x)):
            ps = PoseStamped()
            ps.header.frame_id = frame_id
            ps.pose.position.x = x[i]
            ps.pose.position.y = y[i]
            ps.pose.position.z = 0.0

            # Convert yaw to quaternion
            half_yaw = 0.5 * yaw[i]
            ps.pose.orientation.x = 0.0
            ps.pose.orientation.y = 0.0
            ps.pose.orientation.z = sin(half_yaw)
            ps.pose.orientation.w = cos(half_yaw)

            path.poses.append(ps)

        return path
    
    @staticmethod
    def _load_centerline_csv(csv_path: str, frame_id: str) -> Path:
        path = Path()
        path.header.frame_id = frame_id

        if not csv_path or not os.path.exists(csv_path):
            raise FileNotFoundError(f"Centerline CSV file not found: {csv_path}")
        
        with open(csv_path) as f:
            for line in f:
                if not line.startswith("#"):
                    n_cols = len(line.strip().split(","))
                    break

        if n_cols == 4:  # old format
            df = pd.read_csv(csv_path,
                            delimiter=',',
                            comment='#',
                            header=0,
                            names=['x_m', 'y_m', 'w_tr_right_m', 'w_tr_left_m'])
            x = df['x_m'].values
            y = df['y_m'].values
        elif n_cols == 3:  # new format
            df = pd.read_csv(csv_path,
                            delimiter=',',
                            comment='#',
                            header=0,
                            names=['x', 'y', 'yaw'])
            x = df['x'].values
            y = df['y'].values
        else:
            raise ValueError(f"Unsupported CSV format with {n_cols} columns")


        for i in range(len(x)):
            ps = PoseStamped()
            ps.header.frame_id = frame_id
            ps.pose.position.x = x[i]
            ps.pose.position.y = y[i]
            ps.pose.position.z = 0.0

            # Default orientation (no yaw provided in centerline CSV)
            ps.pose.orientation.x = 0.0
            ps.pose.orientation.y = 0.0
            ps.pose.orientation.z = 0.0
            ps.pose.orientation.w = 1.0

            path.poses.append(ps)

        return path
    
def main():
    rclpy.init()
    node = RacelineVisualizer()
    rclpy.spin(node)  

if __name__ == '__main__':
    main()
