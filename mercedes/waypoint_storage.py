#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
from scipy.spatial.transform import Rotation as R
from numpy.linalg import norm
from tf2_ros import Buffer, TransformListener
import tf2_geometry_msgs
import numpy as np
import os
from time import gmtime, strftime
import atexit

class WaypointLogger(Node):
    def __init__(self):
        super().__init__('waypoint_logger')

        # Setup log file path
        log_dir = '/home/siddarth/f1ws/src/mercedes/storage'
        os.makedirs(log_dir, exist_ok=True)
        log_file = strftime('mapframe-%Y-%m-%d-%H-%M-%S.csv', gmtime())
        self.file_path = os.path.join(log_dir, log_file)
        self.file = open(self.file_path, 'w')
        self.get_logger().info(f'Logging to {self.file_path} (in map frame)')

        # TF2 setup
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Subscribe to odometry
        self.subscription = self.create_subscription(
            Odometry,
            '/odom',
            self.save_waypoint,
            10
        )

        # Close file on shutdown
        atexit.register(self.shutdown_hook)

    def save_waypoint(self, msg):
        # Wrap odometry pose as PoseStamped in odom frame
        input_pose = PoseStamped()
        input_pose.header = msg.header
        input_pose.pose = msg.pose.pose

        try:
            # Transform to map frame
            transformed_pose = self.tf_buffer.transform(
                input_pose, 
                target_frame='map',
                timeout=rclpy.duration.Duration(seconds=0.2)
            )

            # Extract orientation and convert to yaw
            q = transformed_pose.pose.orientation
            quat = [q.x, q.y, q.z, q.w]
            yaw = R.from_quat(quat).as_euler('xyz')[2]

            # Extract position
            p = transformed_pose.pose.position

            # Use original velocity
            v = msg.twist.twist.linear
            speed = norm([v.x, v.y, v.z])

            # Optional: log conditionally
            if speed > 0.1:
                self.get_logger().info(f'MAP frame log: x={p.x:.2f}, y={p.y:.2f}, yaw={yaw:.2f}, speed={speed:.2f}')
                self.file.write(f'{p.x}, {p.y}, 0.0, 0.0, 0.0, {yaw}\n')
                self.file.flush()

        except Exception as e:
            self.get_logger().warn(f"TF transform to 'map' failed: {e}")

    def shutdown_hook(self):
        if not self.file.closed:
            self.file.close()
            self.get_logger().info('Waypoint log file closed.')

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
