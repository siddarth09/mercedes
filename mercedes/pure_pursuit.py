#!/usr/bin/env python3
import rclpy
import math
import numpy as np
import pandas as pd
import tf_transformations

from rclpy.node import Node
from geometry_msgs.msg import PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped
from rclpy.qos import QoSProfile, ReliabilityPolicy
from tf_transformations import euler_from_quaternion
from tf2_ros import Buffer, TransformListener, LookupException, ConnectivityException, ExtrapolationException
from rclpy.time import Time 


class PurePursuit(Node):
    def __init__(self):
        super().__init__('pure_pursuit_csv')

        # Parameters
        self.declare_parameter('dt', 0.1)
        self.declare_parameter('Kdd', 1.0)
        self.declare_parameter('max_steering_angle', 0.6)
        self.declare_parameter('wheel_base', 0.325)
        self.declare_parameter('lookahead', 2.0)
        self.declare_parameter('beta_kappa', 0.4)  # vel smoothing
        self.declare_parameter('v_min', 0.3)
        self.declare_parameter('v_max', 2.0)
        self.declare_parameter('a_lat_max', 2.0)   # lateral acceleration limit
        self.declare_parameter('goal_threshold', 0.3)
        self.declare_parameter('csv_path', '/home/deepak/Data/f1tenth/mercedes_ws/src/mercedes/storage/csc433_clean.csv')

        self.dt = self.get_parameter('dt').value   
        self.Kdd = self.get_parameter('Kdd').value
        self.max_steering_angle = self.get_parameter('max_steering_angle').value
        self.wheel_base = self.get_parameter('wheel_base').value
        self.lookahead = self.get_parameter('lookahead').value
        self.beta_kappa = self.get_parameter('beta_kappa').value
        self.v_mix = self.get_parameter('v_min').value
        self.v_max = self.get_parameter('v_max').value
        self.a_lat_max = self.get_parameter('a_lat_max').value
        self.goal_threshold = self.get_parameter('goal_threshold').value


        self.cmd_vel = None
        self.last_goal_index = -1
        self.goal_reached = False
        self.map_frame = 'map'
        self.base_frame = 'base_link'

        # Load CSV
        self.load_path_from_csv()

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Instead of amcl_pose_sub, create a timer to drive Pure Pursuit at fixed rate:
        self.timer = self.create_timer(self.dt, self.timer_callback)

        # Publisher
        self.ackermann_pub = self.create_publisher(AckermannDriveStamped, '/drive', 10)

        # Subscriber
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)

        self.get_logger().info('Pure Pursuit CSV node initialized')

    def load_path_from_csv(self):
        path = self.get_parameter('csv_path').value
        if not path:
            self.get_logger().error("No CSV files found in directory.")
            return

        df = pd.read_csv(path)  # assumes your CSV has headers: x, y, z, roll, pitch, yaw
        self.path = list(zip(df["x"], df["y"]))
        self.get_logger().info(f"Loaded path length: {len(self.path)}")
    
    def odom_callback(self, msg : Odometry):
        x_vel = msg.twist.twist.linear.x
        y_vel = msg.twist.twist.linear.y
        self.current_vel = math.hypot(x_vel, y_vel)
    
    def update_state_from_tf(self):
        try:
            t = self.tf_buffer.lookup_transform(self.map_frame, self.base_frame, Time())
            trans = t.transform.translation
            rot = t.transform.rotation
            _,_,yaw = euler_from_quaternion([rot.x, rot.y, rot.z, rot.w])
            self.current_state = np.array([trans.x, trans.y, yaw, self.current_vel])
        except (LookupException, ConnectivityException, ExtrapolationException) as e:
            self.get_logger().warn(f"TF Lookup failed: {e}")

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

    def compute_steering_angle(self, x, y, yaw):
        goal_x, goal_y = self.find_target_point(x, y, self.lookahead)

        if goal_x is None or goal_y is None:
            self.get_logger().warn("No valid target point found.")
            return 0.0

        dx = goal_x - x
        dy = goal_y - y

        local_x = np.cos(-yaw) * dx - np.sin(-yaw) * dy
        local_y = np.sin(-yaw) * dx + np.cos(-yaw) * dy

        if local_x <= 0:
            return 0.0

        L_d = self.lookahead
        alpha = np.arctan2(local_y, local_x)         # yaw
        kappa = 2.0 * np.sin(alpha) / max(L_d, 1e-6) # curvature
        delta = np.arctan(self.wheel_base * kappa)   # steering angle
        delta = np.clip(delta, -self.max_steering_angle, self.max_steering_angle)

        # self.get_logger().info(f"[DEBUG] α: {alpha:.2f} rad, Steering: {delta:.2f} rad, Lookahead: {self.lookahead:.2f}")
        return delta, kappa
    
    def compute_vel(self, kappa):
        eps = 1e-6
        self.kappa_s = (1.0 - self.beta_kappa) * self.kappa_s + self.beta_kappa * kappa
        v_curv = np.sqrt(self.a_lat_max / max(abs(self.kappa_s), eps))
        v_cmd  = float(np.clip(v_curv, self.v_min, self.v_max))
        return v_cmd

    def timer_callback(self):

        self.update_state_from_tf()
        if self.current_state is None:
            self.get_logger().warn("Waiting for TF/odom to update the state ...")
            return
        x, y, yaw, vel = self.current_state

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

        steering_angle, kappa = self.compute_steering_angle(x, y, yaw)
        self.cmd_vel = self.compute_vel(kappa)

        cmd = AckermannDriveStamped()
        cmd.drive.speed = self.cmd_vel
        cmd.drive.steering_angle = steering_angle
        self.get_logger().info(f"Sending command → Speed: {cmd.drive.speed}, Steering: {cmd.drive.steering_angle}")

        self.ackermann_pub.publish(cmd)


def main():
    rclpy.init()
    node = PurePursuit()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
