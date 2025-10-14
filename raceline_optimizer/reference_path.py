import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped

from mercedes_msgs.msg import Trajectory, TrajectoryPoint
import numpy as np
import math

class TrajectoryPublisher(Node):
    def __init__(self):
        super().__init__('trajectory_publisher')
        self.pub = self.create_publisher(Trajectory, '/racing_trajectory', 10)
        self.timer = self.create_timer(1.0, self.publish_trajectory)
        self.traj = self.load_csv('/home/siddarth/f1ws/src/mercedes/raceline_optimizer/out/racing_line_with_speed.csv')
        self.path_pub = self.create_publisher(Path, '/racing_trajectory_path', 10)

    def load_csv(self, path):
        data = np.loadtxt(path, delimiter=',', skiprows=1)
        s, x, y, kappa, v = data[:,0], data[:,1], data[:,2], data[:,3], data[:,4]
        traj = Trajectory()
        traj.header.frame_id = "map"

        # estimate yaw and acceleration
        dx, dy = np.gradient(x, s), np.gradient(y, s)
        yaw = np.arctan2(dy, dx)
        dv = np.gradient(v, s)
        a = v * dv  # approx longitudinal accel a = v dv/ds

        for i in range(len(s)):
            p = TrajectoryPoint()
            p.s = float(s[i])
            p.x = float(x[i])
            p.y = float(y[i])
            p.yaw = float(yaw[i])
            p.kappa = float(kappa[i])
            p.v_mps = float(v[i])
            p.a_mps2 = float(a[i])
            traj.points.append(p)
        return traj

    
    def publish_trajectory(self):
        # Publish custom trajectory
        self.traj.header.stamp = self.get_clock().now().to_msg()
        self.pub.publish(self.traj)

        # Also publish Path for RViz
        path_msg = Path()
        path_msg.header.frame_id = "map"
        path_msg.header.stamp = self.get_clock().now().to_msg()

        for p in self.traj.points:
            pose = PoseStamped()
            pose.pose.position.x = p.x
            pose.pose.position.y = p.y
            pose.pose.orientation.z = math.sin(p.yaw / 2.0)
            pose.pose.orientation.w = math.cos(p.yaw / 2.0)
            path_msg.poses.append(pose)

        self.path_pub.publish(path_msg)
        self.get_logger().info("Published full racing trajectory + Path for RViz")


def main(args=None):
    rclpy.init(args=args)
    node = TrajectoryPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
