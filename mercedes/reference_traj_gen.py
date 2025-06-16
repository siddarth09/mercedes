#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
import pandas as pd
import os
import glob

class WaypointPathPublisher(Node):
    def __init__(self):
        super().__init__('reference_traj_gen')

        # Parameters
        self.declare_parameter('waypoint_dir', '/home/deepak/Data/f1tenth/mercedes_ws/src/mercedes/storage')
        self.frame_id = 'map'

        # Get latest CSV file
        directory = self.get_parameter('waypoint_dir').value
        latest_csv = self.get_latest_csv(directory)

        if latest_csv is None:
            self.get_logger().error(f"No CSV files found in {directory}")
            return

        self.get_logger().info(f"Using latest waypoint file: {latest_csv}")
        self.waypoints = pd.read_csv(latest_csv)
        self.path_pub = self.create_publisher(Path, '/reference_path', 10)
        self.timer = self.create_timer(1.0, self.publish_path)

    def get_latest_csv(self, directory):
        csv_files = glob.glob(os.path.join(directory, 'wp-*.csv'))
        if not csv_files:
            return None
        return max(csv_files, key=os.path.getmtime)

    def publish_path(self):
        path_msg = Path()
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.header.frame_id = self.frame_id

        for _, row in self.waypoints.iterrows():
            pose = PoseStamped()
            pose.header = path_msg.header
            pose.pose.position.x = row['x']
            pose.pose.position.y = row['y']
            pose.pose.position.z = 0.0
            pose.pose.orientation.w = 1.0  # Assuming no rotation
            path_msg.poses.append(pose)

        self.path_pub.publish(path_msg)
        self.get_logger().info("Published reference path.")

def main(args=None):
    rclpy.init(args=args)
    node = WaypointPathPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
