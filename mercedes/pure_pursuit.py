#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np
import pandas as pd
from geometry_msgs.msg import PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped
import tf_transformations
import os
from rclpy.qos import QoSProfile, ReliabilityPolicy
from tf2_ros import Buffer, TransformListener, TransformException
from builtin_interfaces.msg import Time



class PurePursuit(Node):
    def __init__(self):
        super().__init__('pure_pursuit_csv')

        # Parameters
        self.declare_parameter('Kdd', 1.0)
        self.declare_parameter('min_ld', 0.5)
        self.declare_parameter('max_ld', 2.0)
        self.declare_parameter('max_steering_angle', 0.6)
        self.declare_parameter('wheel_base', 0.325)
        self.declare_parameter('csv_path', '/home/deepak/Data/f1tenth/mercedes_ws/src/mercedes/storage')

        self.Kdd = self.get_parameter('Kdd').value
        self.min_ld = self.get_parameter('min_ld').value
        self.max_ld = self.get_parameter('max_ld').value
        self.max_steering_angle = self.get_parameter('max_steering_angle').value
        self.wheel_base = self.get_parameter('wheel_base').value
        self.speed = 0.5
        self.goal_threshold = 0.3
        self.last_goal_index = -1
        self.goal_reached = False

        # Load CSV
        self.load_path_from_csv()

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Instead of amcl_pose_sub, create a timer to drive Pure Pursuit at fixed rate:
        self.timer = self.create_timer(0.1, self.timer_callback)

        # Publisher
        self.ackermann_pub = self.create_publisher(AckermannDriveStamped, '/drive', 10)

        self.get_logger().info('Pure Pursuit CSV node initialized')

    def load_path_from_csv(self):
        path = self.get_parameter('csv_path').value
        csv_files = sorted([f for f in os.listdir(path) if f.endswith('.csv')])
        if not csv_files:
            self.get_logger().error("No CSV files found in directory.")
            return

        latest_csv = os.path.join(path, csv_files[-1])
        self.get_logger().info(f"Loading CSV: {latest_csv}")
        df = pd.read_csv(latest_csv, header=None, names=["x", "y", "z", "roll", "pitch", "yaw"])
        self.path = list(zip(df["x"], df["y"]))
        self.get_logger().info(f"Loaded path length: {len(self.path)}")


    def find_target_point(self, x, y, lookahead_distance):
        start_index = max(0, self.last_goal_index)
        for i in range(start_index, len(self.path)):
            px, py = self.path[i]
            distance = np.hypot(px - x, py - y)
            if distance >= lookahead_distance:
                self.last_goal_index = i
                self.get_logger().info(f"Distance to point {i}: {distance:.2f} m")
                return px, py
        if self.path:
            return self.path[-1]
        return None, None

    def compute_steering_angle(self, x, y, yaw, speed):
        ld = 0.3
        goal_x, goal_y = self.find_target_point(x, y, ld)

        if goal_x is None or goal_y is None:
            self.get_logger().warn("No valid target point found.")
            return 0.0

        dx = goal_x - x
        dy = goal_y - y

        local_x = np.cos(-yaw) * dx - np.sin(-yaw) * dy
        local_y = np.sin(-yaw) * dx + np.cos(-yaw) * dy

        if local_x <= 0:
            return 0.0

        alpha = np.arctan2(local_y, local_x)
        delta = np.arctan(2 * self.wheel_base * np.sin(alpha) / ld)
        delta = np.clip(delta, -self.max_steering_angle, self.max_steering_angle)

        self.get_logger().info(f"[DEBUG] α: {alpha:.2f} rad, Steering: {delta:.2f} rad, Lookahead: {ld:.2f}")
        return delta
    
    def timer_callback(self):
        
        try: 
            # Lookup transform map → base_link
            transform = self.tf_buffer.lookup_transform(
            "map", "base_link", rclpy.time.Time(), rclpy.duration.Duration(seconds=0.5)
            )


            # Extract translation
            x = transform.transform.translation.x
            y = transform.transform.translation.y

            # Extract yaw from quaternion
            q = transform.transform.rotation
            quat = [q.x, q.y, q.z, q.w]
            _, _, yaw = tf_transformations.euler_from_quaternion(quat)

            self.get_logger().info("Got TF map → base_link")

            # Now run Pure Pursuit
            if not self.path:
                self.get_logger().warn("Path is empty.")
                return

            goal_x, goal_y = self.path[-1]
            distance_to_goal = np.hypot(goal_x - x, goal_y - y)
            if distance_to_goal < self.goal_threshold:
                if not self.goal_reached:
                    self.get_logger().info(f"FINAL GOAL REACHED at ({goal_x:.2f}, {goal_y:.2f})")
                    self.goal_reached = True

                    stop_cmd = AckermannDriveStamped()
                    stop_cmd.drive.speed = 0.0
                    stop_cmd.drive.steering_angle = 0.0
                    self.ackermann_pub.publish(stop_cmd)
                return

            steering_angle = self.compute_steering_angle(x, y, yaw, self.speed)

            cmd = AckermannDriveStamped()
            cmd.drive.speed = self.speed
            cmd.drive.steering_angle = steering_angle
            self.get_logger().info(f"Sending command → Speed: {cmd.drive.speed}, Steering: {cmd.drive.steering_angle}")

            self.ackermann_pub.publish(cmd)

        except TransformException as ex:
            self.get_logger().warn(f'Could not transform map → base_link: {ex}')
            return


def main():
    rclpy.init()
    node = PurePursuit()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
