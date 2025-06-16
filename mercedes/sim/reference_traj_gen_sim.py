#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import pandas as pd
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Header
import tf_transformations

class WaypointPathPublisher(Node):
    def __init__(self):
        super().__init__('reference_traj_gen')

        self.declare_parameter('waypoint_csv', '/home/deepak/Data/f1tenth/mercedes_ws/src/mercedes/storage/wp-2025-06-10-16-56-58.csv')
        self.frame_id = 'map'  # Or use 'odom' if map is not available

        csv_path = self.get_parameter('waypoint_csv').value
        self.path_pub = self.create_publisher(Path, '/reference_path', 10)

        self.timer = self.create_timer(1.0, self.publish_path)

        self.waypoints = pd.read_csv(csv_path)
        self.get_logger().info(f"Loaded {len(self.waypoints)} waypoints.")

    def publish_path(self):
        path_msg = Path()
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.header.frame_id = self.frame_id

        for index, row in self.waypoints.iterrows():
            pose = PoseStamped()
            pose.header = path_msg.header
            pose.pose.position.x = row['x']
            pose.pose.position.y = row['y']
            pose.pose.position.z = 0.0
            pose.pose.orientation.w = 1.0  # No rotation
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
