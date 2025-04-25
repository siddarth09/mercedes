import rclpy 
from rclpy.node import Node 
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
import numpy as np

class TTC(Node):
    def __init__(self):
        super().__init__('aeb_node')

        # Parameters
        self.ttc_threshold = 2.0  # seconds
        self.min_angle = np.deg2rad(-30)
        self.max_angle = np.deg2rad(30)

        # Runtime variables
        self.current_speed = 0.0
        self.speed_cmd = Twist()  # Default to zero cmd
        self.laser_ranges = None
        self.laser_angles = None

        # Subscriptions
        self.create_subscription(LaserScan, '/scan', self.lidar_callback, 10)
        self.create_subscription(Twist, '/cmd_vel', self.speed_callback, 10)

        # Publisher
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # Timer to compute TTC periodically
        self.timer = self.create_timer(0.1, self.compute_ttc)

        self.get_logger().info('AEB node started')

    def speed_callback(self, msg):
        self.speed_cmd = msg
        self.current_speed = msg.linear.x
        # Optional debug log
        # self.get_logger().info(f'Current speed: {self.current_speed:.2f} m/s')

    def lidar_callback(self, msg):
        # Cache laser scan angles and ranges
        angles = np.linspace(msg.angle_min, msg.angle_max, len(msg.ranges))
        ranges = np.array(msg.ranges)

        # Filter forward sector
        forward_indices = np.where((angles >= self.min_angle) & (angles <= self.max_angle))
        if len(forward_indices[0]) == 0:
            self.laser_ranges = None
            return

        self.laser_angles = angles[forward_indices]
        self.laser_ranges = ranges[forward_indices]

    def compute_ttc(self):
        if self.laser_ranges is None or self.speed_cmd is None:
            return

        forward_ranges = np.clip(self.laser_ranges, 0.001, np.inf)
        forward_angles = self.laser_angles

        # Compute relative speed
        relative_speeds = self.current_speed * np.cos(forward_angles)
        relative_speeds = np.clip(relative_speeds, 0.001, np.inf)

        # Compute TTC
        ttc = forward_ranges / relative_speeds
        min_ttc = np.min(ttc)

        # Output decision
        cmd = Twist()
        if min_ttc < self.ttc_threshold:
            self.get_logger().warn(f' EMERGENCY BRAKING! TTC = {min_ttc:.2f} sec')
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
        else:
            cmd = self.speed_cmd  # Let teleop drive normally

        self.cmd_pub.publish(cmd)

def main(args=None):
    rclpy.init(args=args)
    ttc_node = TTC()
    rclpy.spin(ttc_node)
    ttc_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
