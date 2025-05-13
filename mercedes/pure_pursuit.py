#!/usr/bin/env python3
import rclpy 
from rclpy.node import Node
import numpy as np 
from geometry_msgs.msg import PoseStamped 
from nav_msgs.msg import Path, Odometry
from ackermann_msgs.msg import AckermannDriveStamped
import tf_transformations

class PurePursuit(Node):
    def __init__(self):
        super().__init__('pure_pursuit')

        # Parameters
        self.declare_parameter('Kdd', 1.0)
        self.declare_parameter('min_ld', 0.5)
        self.declare_parameter('max_ld', 2.0)
        self.declare_parameter('max_steering_angle', 0.6)
        self.declare_parameter('wheel_base', 0.325)

        self.Kdd = self.get_parameter('Kdd').value
        self.min_ld = self.get_parameter('min_ld').value
        self.max_ld = self.get_parameter('max_ld').value
        self.max_steering_angle = self.get_parameter('max_steering_angle').value
        self.wheel_base = self.get_parameter('wheel_base').value
        self.speed = 1.0
        self.goal_threshold = 0.3  

        self.path = []
        self.last_goal_index = -1
        self.goal_reached = False

        self.path_sub = self.create_subscription(Path, '/reference_trajectory', self.path_callback, 10)
        self.odom_sub = self.create_subscription(Odometry, '/odometry/filtered', self.pose_callback, 10)
        self.ackermann_pub = self.create_publisher(AckermannDriveStamped, '/drive', 10)

        self.get_logger().info('Pure Pursuit initialized (with dynamic lookahead + α steering)')

    def path_callback(self, msg):
        self.path = [(pose.pose.position.x, pose.pose.position.y) for pose in msg.poses]
        self.goal_reached = False
        self.get_logger().info(f"Path received with {len(self.path)} points")

    def find_target_point(self, x, y, lookahead_distance):
        start_index = max(0, self.last_goal_index)
        for i in range(start_index, len(self.path)):
            px, py = self.path[i]
            distance = np.hypot(px - x, py - y)
            self.get_logger().info(f"[DEBUG] Distance to pt {i}: {distance:.2f}")
            if distance >= lookahead_distance:
                self.last_goal_index = i
                return px, py
        # fallback to last point
        if self.path:
            return self.path[-1]
        return None, None



    def compute_steering_angle(self, x, y, yaw, speed):
        ld = np.clip(self.Kdd * speed, self.min_ld, self.max_ld)
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

    def pose_callback(self, msg):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        quat = [q.x, q.y, q.z, q.w]
        _, _, yaw = tf_transformations.euler_from_quaternion(quat)

        if not self.path:
            self.get_logger().warn("Waiting for path...")
            return

        # Check if final goal reached
        if not self.goal_reached:
            goal_x, goal_y = self.path[-1]
            distance_to_goal = np.hypot(goal_x - x, goal_y - y)
            if distance_to_goal < self.goal_threshold:
                self.get_logger().info(f" FINAL GOAL REACHED at ({goal_x:.2f}, {goal_y:.2f})")
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
        self.ackermann_pub.publish(cmd)

def main():
    rclpy.init()
    node = PurePursuit()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
