#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
import numpy as np
from numpy.linalg import norm
import tf_transformations
from os.path import expanduser
from time import gmtime, strftime
import atexit
import os

class WaypointLogger(Node):
    def __init__(self):
        super().__init__('waypoint_logger')

        # Setup log file path
        log_dir = '/home/siddarth/f1ws/src/mercedes/storage'
        os.makedirs(log_dir, exist_ok=True)
        log_file = strftime('wp-%Y-%m-%d-%H-%M-%S.csv', gmtime())
        self.file = open(os.path.join(log_dir, log_file), 'w')
        self.get_logger().info(f'Logging to {self.file.name}')

        # Subscribe to odometry
        self.subscription = self.create_subscription(
            Odometry,
            '/odometry/filtered',
            self.save_waypoint,
            10
        )

        # Close file on shutdown
        atexit.register(self.shutdown_hook)

    def save_waypoint(self, msg):
        # Orientation to Euler
        q = msg.pose.pose.orientation
        quat = [q.x, q.y, q.z, q.w]
        roll, pitch, yaw = tf_transformations.euler_from_quaternion(quat)

        # Speed from linear velocity
        v = msg.twist.twist.linear
        speed = norm([v.x, v.y, v.z])

        if v.x > 0.0:
            self.get_logger().info(f'Speed X: {v.x:.2f}')

        # Write to file: x, y, yaw, speed
        p = msg.pose.pose.position
        self.file.write(f'{p.x}, {p.y}, {yaw}, {speed}\n')

    def shutdown_hook(self):
        self.file.close()
        print(' Waypoint log file closed.')

def main(args=None):
    rclpy.init(args=args)
    node = WaypointLogger()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.shutdown_hook()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
