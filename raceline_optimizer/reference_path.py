import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path

class ReferencePublisher(Node):
    def __init__(self):
        super().__init__('reference_publisher')
        self.path_pub = self.create_publisher(Path, '/reference_path', 10)
        timer_period = 2.0
        self.timer = self.create_timer(timer_period, self.publish_path)
        self.declare_parameter('path_file', '/home/siddarth/f1ws/src/mercedes/raceline_optimizer/out/racing_line_world.csv')
        path_file = self.get_parameter('path_file').get_parameter_value().string_value
        self.data = np.loadtxt(path_file, delimiter=",", skiprows=1)
        
    def publish_path(self):

        s, x, y, kappa = self.data.T
        path_msg = Path()
        path_msg.header.frame_id = "map"
        for xi, yi in zip(x, y):
            pose = PoseStamped()
            pose.header.frame_id = "map"
            pose.pose.position.x = float(xi)
            pose.pose.position.y = float(yi)
            path_msg.poses.append(pose)
        self.path_pub.publish(path_msg)
        self.get_logger().info("Published optimized racing line.")

def main(args=None):
    rclpy.init(args=args)
    node = ReferencePublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
