import rclpy
from rclpy.node import Node
import numpy as np  

from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry


class AEB(Node):

    def __init__(self):
        super().__init__('aeb_sim_node')
        
        # Define parameters
        self.ttc_threshold = 2.0
        self.min_angle = np.deg2rad(-30)
        self.max_angle = np.deg2rad(30)
        

        self.current_speed = 0.0
        self.laser_angles = None
        self.laser_ranges = None

        # Subscribers
        self.create_subscription(LaserScan, "/scan", self.laser_callback, 10)
        self.create_subscription(Odometry, "/ego_racecar/odom", self.odom_callback, 10)

        # Publishers
        self.ackerman_pub = self.create_publisher(AckermannDriveStamped, "/drive", 10)

        self.create_timer(0.1, self.compute_ttc)
        self.get_logger().info("AEB Node started")


    def laser_callback(self, msg: LaserScan):
        
        angles = np.linspace(msg.angle_min, msg.angle_max, len(msg.ranges))
        ranges = np.array(msg.ranges)

        mask = (angles <= self.max_angle) & (angles >= self.min_angle)
        forward_indices = np.where(mask)[0]

        if len(forward_indices)==0:
            self.laser_ranges = None

        self.laser_angles = angles[forward_indices]
        self.laser_ranges = ranges[forward_indices]

    
    def odom_callback(self, msg: Odometry):
        
        vx = msg.twist.twist.linear.x
        vy = msg.twist.twist.linear.y

        self.current_speed = np.linalg.norm([vx,vy])

    def compute_ttc(self):

        if self.laser_ranges is None:
            return
        
        forward_ranges = np.clip(self.laser_ranges, 0.01, np.inf)
        forward_angles = self.laser_angles

        relative_speed = self.current_speed * np.cos(forward_angles)
        relative_speed = np.clip(relative_speed, 0.01, np.inf)

        ttc = forward_ranges / relative_speed
        min_ttc = np.min(ttc)

        if min_ttc <= self.ttc_threshold:
            self.get_logger().warn(f' EMERGENCY BRAKING! TTC = {min_ttc:.2f} sec')
            cmd = AckermannDriveStamped()
            cmd.drive.speed = 0.0
            cmd.drive.steering_angle = 0.0
            self.ackerman_pub.publish(cmd)


def main(args=None):

    rclpy.init(args=args)
    node = AEB()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()


                 