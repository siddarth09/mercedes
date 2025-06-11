#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseWithCovarianceStamped
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
        log_dir = "/home/deepak/Data/f1tenth/mercedes_ws/src/mercedes/storage"
        os.makedirs(log_dir, exist_ok=True)
        file_name = strftime('wp-%Y-%m-%d-%H-%M-%S.csv', gmtime())
        self.file_path = os.path.join(log_dir, file_name)
        self.file = open(self.file_path, 'w')
        self.get_logger().info(f'Saving waypoints to {self.file_path}')

        # Subscriber to odometry
        self.amcl_pose_sub = self.create_subscription(PoseWithCovarianceStamped, '/amcl_pose', self.save_waypoint, 10)

        # Ensure file is closed on shutdown
        atexit.register(self.shutdown_hook)

    def save_waypoint(self, msg):
        # Extract orientation (quaternion to yaw)
        q = msg.pose.pose.orientation
        quat = [q.x, q.y, q.z, q.w]
        _, _, yaw = euler_from_quaternion(quat)

        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y

        # Log at every callback (no velocity info in amcl_pose)
        self.get_logger().info(f"x={x:.2f}, y={y:.2f}, yaw={yaw:.2f}")
        self.file.write(f"{x}, {y}, {yaw}\n")
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
