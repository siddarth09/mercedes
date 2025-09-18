#!/usr/bin/env python3
import os
import csv
from math import sin, cos
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
import csv
from io import StringIO
import pandas as pd

class RacelineVisualizer(Node):
    def __init__(self):
        super().__init__('raceline_visualizer')

        # ---- Parameters ----
        self.declare_parameter('csv_dir', '/home/deepak/Data/f1tenth/mercedes_ws/src/mercedes/storage/racelines')
        self.declare_parameter('raceline_csv', 'Spielberg_raceline.csv')
        self.declare_parameter('centerline_csv', 'Spielberg_centerline.csv')
        self.declare_parameter('frame_id', 'map')

        csv_dir     = self.get_parameter('csv_dir').get_parameter_value().string_value
        raceline_csv   = self.get_parameter('raceline_csv').get_parameter_value().string_value
        centerline_csv    = self.get_parameter('centerline_csv').get_parameter_value().string_value
        self.frame  = self.get_parameter('frame_id').get_parameter_value().string_value

        # ---- QoS: latched so RViz can latch the Path ----
        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.pub_raceline = self.create_publisher(Path, '/raceline', qos)
        self.pub_centerline = self.create_publisher(Path, '/centerline', qos)

        # Build and publish
        raceline_path = os.path.join(csv_dir, raceline_csv)
        path_msg_raceline = self._load_raceline_csv(raceline_path, self.frame)
        self.pub_raceline.publish(path_msg_raceline)
        self.get_logger().info(f"Published raceline with {len(path_msg_raceline.poses)} poses on '/raceline' (frame_id='{self.frame}').")

        centerline_path = os.path.join(csv_dir, centerline_csv)
        path_msg_centerline = self._load_centerline_csv(centerline_path, self.frame)
        self.pub_centerline.publish(path_msg_centerline)
        self.get_logger().info(f"Published raceline with {len(path_msg_centerline.poses)} poses on '/centerline' (frame_id='{self.frame}').")


        # Give DDS a moment to deliver, then shut down
        self.create_timer(1.0, self._shutdown_after_publish)
  

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
        
        df = pd.read_csv(csv_path, 
                delimiter=',', 
                comment='#',   # This tells pandas to skip lines starting with #
                header=0,      # Use first non-comment line as header
                names=['x_m', 'y_m', 'w_tr_right_m', 'w_tr_left_m'])

        x = df['x_m'].values
        y = df['y_m'].values

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
