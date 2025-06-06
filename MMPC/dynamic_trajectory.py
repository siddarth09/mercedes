#!/usr/bin/env python3
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from geometry_msgs.msg import PoseStamped
from scipy.spatial.transform import Rotation as R
from scipy.ndimage import distance_transform_edt as distance_transform
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy


qos_profile = QoSProfile(
    reliability=QoSReliabilityPolicy.RELIABLE,
    durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
    depth=1
)


class DynamicTrajectoryPublisher(Node):
    def __init__(self):
        super().__init__("dynamic_trajectory_publisher")

        qos_profile = QoSProfile(depth=10)
        qos_profile.durability = QoSDurabilityPolicy.TRANSIENT_LOCAL

        self.map_sub = self.create_subscription(OccupancyGrid, "/map", self.map_callback, qos_profile)
        self.odom_sub = self.create_subscription(Odometry, "/ego_racecar/odom", self.odom_callback, 10)
        self.trajectory_pub = self.create_publisher(Path, "/dynamic_trajectory", qos_profile)

        # Parameters
        self.declare_parameter('wheelbase', 0.4)
        self.declare_parameter('horizon_length', 10)
        self.declare_parameter('dt', 0.1)
        self.declare_parameter('max_speed', 1.0)
        self.declare_parameter('max_steering_angle', np.pi / 4)

        self.L = self.get_parameter('wheelbase').get_parameter_value().double_value
        self.N = self.get_parameter('horizon_length').get_parameter_value().integer_value
        self.dt = self.get_parameter('dt').get_parameter_value().double_value
        self.v = self.get_parameter('max_speed').get_parameter_value().double_value
        self.theta_max = self.get_parameter('max_steering_angle').get_parameter_value().double_value

        self.current_odom = None
        self.occupancy_grid = None
        self.map_info = None
        self.safe_distance = 0.3  # meters to wall (CBF threshold)

    def map_callback(self, msg):
        self.map_info = msg.info
        self.get_logger().info(f"Received map with size: {msg.info.width}x{msg.info.height}")
        grid = np.array(msg.data).reshape((msg.info.height, msg.info.width))
        binary_map = (grid == 0).astype(np.uint8)
        self.get_logger().info(f"Unique map values: {np.unique(grid)}")

        self.occupancy_grid = distance_transform(binary_map) * msg.info.resolution
   

    def odom_callback(self, msg):
        if self.occupancy_grid is None or self.map_info is None:
            self.get_logger().warn("Map not received yet.")
            return

        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        theta = R.from_quat([
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w
        ]).as_euler('xyz')[2]

        best_score = -np.inf
        best_trajectory = None

        for delta in np.linspace(-self.theta_max, self.theta_max, 21):
            traj = self.simulate_rk4_trajectory(x, y, theta, delta)
            score = self.score_trajectory(traj)
            if score > best_score:
                best_score = score
                best_trajectory = traj

        if best_trajectory:
            self.publish_path(best_trajectory)

    def kinematic_model(self, state, delta):
        x, y, theta = state
        dx = self.v * np.cos(theta)
        dy = self.v * np.sin(theta)
        dtheta = (self.v / self.L) * np.tan(delta)
        return np.array([dx, dy, dtheta])

    def simulate_rk4_trajectory(self, x, y, theta, delta):
        state = np.array([x, y, theta])
        traj = []
        for _ in range(self.N):
            k1 = self.kinematic_model(state, delta)
            k2 = self.kinematic_model(state + 0.5 * self.dt * k1, delta)
            k3 = self.kinematic_model(state + 0.5 * self.dt * k2, delta)
            k4 = self.kinematic_model(state + self.dt * k3, delta)
            state += (self.dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
            traj.append(tuple(state))
        return traj

    def score_trajectory(self, traj):
        score = 0.0
        min_dist = np.inf

        for x, y, _ in traj:
            mx = int((x - self.map_info.origin.position.x) / self.map_info.resolution)
            my = int((y - self.map_info.origin.position.y) / self.map_info.resolution)

            if 0 <= mx < self.occupancy_grid.shape[1] and 0 <= my < self.occupancy_grid.shape[0]:
                dist = self.occupancy_grid[my, mx]
                min_dist = min(min_dist, dist)
                score += dist
            else:
                return -np.inf 

        
        if min_dist < self.safe_distance:
            return -np.inf
        score -= 0.1 * self.curvature_cost(traj)

        return score

    def curvature_cost(self, traj):
        yaws = [theta for _, _, theta in traj]
        diffs = np.diff(yaws)
        return np.sum(np.abs(diffs))

    def publish_path(self, traj):
        path = Path()
        path.header.frame_id = "map"
        path.header.stamp = self.get_clock().now().to_msg()

        for x, y, _ in traj:
            pose = PoseStamped()
            pose.header = path.header
            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.orientation.w = 1.0
            path.poses.append(pose)

        self.trajectory_pub.publish(path)


def main(args=None):
    rclpy.init(args=args)
    node = DynamicTrajectoryPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
