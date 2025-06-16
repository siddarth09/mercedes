#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped
import numpy as np
import math

class AEBNode(Node):
    def __init__(self):
        super().__init__('aeb_node')

        # ---- Parameters ----
        self.ttc_threshold = 0.5        # seconds
        self.min_angle = math.radians(-30)
        self.max_angle = math.radians( 30)
        self.brake_hold_duration = 0.2  # seconds

        # ---- State ----
        self.current_speed = 0.0
        self.laser_angles  = None
        self.laser_ranges  = None
        self.brake_end_time = None

        # ---- Subscriptions ----
        self.create_subscription(LaserScan, '/scan', self.lidar_callback, 10)
        self.create_subscription(Odometry, '/odom', self.odom_callback, 10)

        # ---- Publishers ----
        self.cmd_pub = self.create_publisher(AckermannDriveStamped, '/drive', 10)
        self.forward_scan_pub = self.create_publisher(LaserScan, '/forward_scan', 10) # to visualize forward scans

        # ---- Timer ----
        self.create_timer(0.1, self.compute_ttc)

        self.get_logger().info('AEB node started')

    def odom_callback(self, msg: Odometry):
        vx = msg.twist.twist.linear.x
        vy = msg.twist.twist.linear.y
        
        self.current_speed = math.hypot(vx, vy)

    def lidar_callback(self, msg: LaserScan):
        
        angles = np.linspace(msg.angle_min, msg.angle_max, len(msg.ranges))
        ranges = np.array(msg.ranges)

        # keep only the forward-facing beams
        mask = (angles >= self.min_angle) & (angles <= self.max_angle)
        idx = np.where(mask)[0]
        if idx.size == 0:
            self.laser_ranges = None
            return

        self.laser_angles = angles[idx]
        self.laser_ranges = ranges[idx]
        self.publish_forward_scan(msg)

    def publish_forward_scan(self, original: LaserScan):
        fs = LaserScan()
        fs.header          = original.header
        fs.angle_min       = self.laser_angles[0]
        fs.angle_max       = self.laser_angles[-1]
        fs.angle_increment = (
            self.laser_angles[1] - self.laser_angles[0]
            if len(self.laser_angles) > 1 else 0.0)
        fs.time_increment  = original.time_increment
        fs.scan_time       = original.scan_time
        fs.range_min       = original.range_min
        fs.range_max       = original.range_max
        fs.ranges          = self.laser_ranges.tolist()
        self.forward_scan_pub.publish(fs)

    def compute_ttc(self):
        now = self.get_clock().now()

        # Need valid scan to compute TTC
        if self.laser_ranges is None:
            return

        # compute TTC per beam
        fr = np.clip(self.laser_ranges, 1e-3, np.inf)
        fa = self.laser_angles
        rel_speeds = np.clip(self.current_speed * np.cos(fa), 1e-2, np.inf)
        ttc_vals = fr / rel_speeds
        min_ttc = float(np.min(ttc_vals))

        self.get_logger().info(f"TTC = {min_ttc:.2f}s")

        # trigger brake if threshold exceeded
        if min_ttc < self.ttc_threshold:
            self.get_logger().warn(f"*** EMERGENCY BRAKE for {self.brake_hold_duration}s (TTC = {min_ttc:.2f}) ***")
            self.brake_end_time = now + Duration(seconds=self.brake_hold_duration)
            brake = AckermannDriveStamped()
            brake.drive.speed = 0.0
            brake.drive.steering_angle = 0.0
            self.cmd_pub.publish(brake)
            
        # else: do nothing (upstream /drive continues unchanged)

def main(args=None):
    rclpy.init(args=args)
    node = AEBNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
