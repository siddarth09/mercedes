#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from ackermann_msgs.msg import AckermannDriveStamped
from sensor_msgs.msg import LaserScan
import numpy as np
import matplotlib.pyplot as plt
from collections import deque



class PidController(Node):
    def __init__(self):
        super().__init__('pid_controller')

        self.declare_parameter('desired_distance', 1.0)
        self.declare_parameter('lookahead', 1.0)
        self.declare_parameter('theta_deg', 50.0)
        self.declare_parameter('Kp', 1.2)
        self.declare_parameter('Ki', 0.0)
        self.declare_parameter('Kd', 0.2)

        self.drive_pub = self.create_publisher(AckermannDriveStamped, '/drive', 10)
        self.create_subscription(LaserScan, '/scan', self.lidar_callback, 10)
        self.create_timer(0.1, self.pid_control)

        self.error = 0.0
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_time = self.get_clock().now().nanoseconds / 1e9

        self.latest_scan = None
        # Real-time error tracking
        self.error_window = deque(maxlen=100)
        self.time_window = deque(maxlen=100)
        self.plot_start_time = self.get_clock().now().nanoseconds / 1e9

        # Matplotlib setup
        plt.ion()
        self.fig, self.ax = plt.subplots()
        self.line, = self.ax.plot([], [], label="Wall Following Error")
        self.ax.set_ylim(-2, 2)
        self.ax.set_xlim(0, 10)
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Error (m)")
        self.ax.set_title("Live PID Wall Following Error")
        self.ax.grid(True)
        self.ax.legend()


    def get_range(self, scan, angle_deg):
        angle_rad = np.deg2rad(angle_deg)
        index = int((angle_rad - scan.angle_min) / scan.angle_increment)
        index = np.clip(index, 0, len(scan.ranges) - 1)
        r = scan.ranges[index]
        return np.nan if np.isnan(r) or r == 0.0 or r >= scan.range_max else r

    def lidar_callback(self, scan):
        self.latest_scan = scan

    def pid_control(self):
        if self.latest_scan is None:
            return

        scan = self.latest_scan
        theta_deg = self.get_parameter('theta_deg').get_parameter_value().double_value
        desired = self.get_parameter('desired_distance').get_parameter_value().double_value
        lookahead = self.get_parameter('lookahead').get_parameter_value().double_value

        a = self.get_range(scan, 90 - theta_deg)
        b = self.get_range(scan, 90)

        if a is None or b is None or np.isnan(a) or np.isnan(b):
            self.get_logger().warn("Invalid LIDAR readings.")
            return

        theta_rad = np.deg2rad(theta_deg)
        alpha = np.arctan2(a * np.cos(theta_rad) - b, a * np.sin(theta_rad))
        D_t = b * np.cos(alpha)
        D_t1 = D_t + lookahead * np.sin(alpha)

        self.error = D_t1 - desired
        current_time = self.get_clock().now().nanoseconds / 1e9
        dt = current_time - self.prev_time
        self.prev_time = current_time

        self.integral += self.error * dt
        derivative = (self.error - self.prev_error) / dt if dt > 0 else 0.0
        self.prev_error = self.error

        Kp = self.get_parameter('Kp').get_parameter_value().double_value
        Ki = self.get_parameter('Ki').get_parameter_value().double_value
        Kd = self.get_parameter('Kd').get_parameter_value().double_value

        steering_angle = Kp * self.error + Ki * self.integral + Kd * derivative
        self.get_logger().info(f"Steering Angle: {np.rad2deg(steering_angle):.2f}°")
       

        # Speed policy based on steering angle
        abs_deg = np.rad2deg(abs(steering_angle))
        if abs_deg < 10:
            speed = 4.5
        elif abs_deg < 20:
            speed = 3.0
        else:
            speed = 0.5

        msg = AckermannDriveStamped()
        msg.drive.steering_angle = steering_angle
        msg.drive.speed = 5.0
        self.drive_pub.publish(msg)

        self.get_logger().info(f" e={self.error:.2f}, steer={steering_angle:.2f}, speed={speed:.2f}")
        
        
        # Realtime error plotting
        plot_now = self.get_clock().now().nanoseconds / 1e9
        t = plot_now - self.plot_start_time
        self.time_window.append(t)
        self.error_window.append(self.error)

        self.line.set_xdata(self.time_window)
        self.line.set_ydata(self.error_window)

        # Adjust X-axis dynamically
        if t > self.ax.get_xlim()[1]:
            self.ax.set_xlim(t - 10, t)

        self.ax.figure.canvas.draw()
        self.ax.figure.canvas.flush_events()

def main():
    rclpy.init()
    node = PidController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
