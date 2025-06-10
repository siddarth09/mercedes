#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from tf_transformations import euler_from_quaternion
from numpy.linalg import norm
from time import gmtime, strftime
from os.path import expanduser
import numpy as np
import atexit
import os

class WaypointsLogger(Node):
    def __init__(self):
        super().__init__('waypoints_logger')

        # Create output directory and CSV file
        home = expanduser('~')
        log_dir = "/home/siddarth/f1ws/src/mercedes/storage"
        os.makedirs(log_dir, exist_ok=True)
        file_name = strftime('wp-%Y-%m-%d-%H-%M-%S.csv', gmtime())
        self.file_path = os.path.join(log_dir, file_name)
        self.file = open(self.file_path, 'w')
        self.get_logger().info(f'Saving waypoints to {self.file_path}')

        # Subscriber to odometry
        self.subscriber = self.create_subscription(
            Odometry,
            '/odom',
            self.save_waypoint,
            10
        )

        # Ensure file is closed on shutdown
        atexit.register(self.shutdown_hook)

    def save_waypoint(self, msg):
        # Extract orientation (quaternion to yaw)
        q = msg.pose.pose.orientation
        quat = [q.x, q.y, q.z, q.w]
        _, _, yaw = euler_from_quaternion(quat)

        # Compute speed
        v = msg.twist.twist.linear
        speed = norm([v.x, v.y, v.z])

        # Only log when moving forward
        if v.x > 0.0:
            self.get_logger().info(f"x={msg.pose.pose.position.x:.2f}, y={msg.pose.pose.position.y:.2f}, yaw={yaw:.2f}, speed={speed:.2f}")
            self.file.write(f"{msg.pose.pose.position.x}, {msg.pose.pose.position.y}, {yaw}, {speed}\n")
            self.file.flush()

    def shutdown_hook(self):
        if not self.file.closed:
            self.file.close()
            self.get_logger().info("Waypoint log file closed.")

def main(args=None):
    rclpy.init(args=args)
    node = WaypointsLogger()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.shutdown_hook()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
