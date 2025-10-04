#!/usr/bin/env python3
import numpy as np
import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid, Path
from geometry_msgs.msg import PoseStamped
from scipy.ndimage import distance_transform_edt as distance_transform
from tf2_ros import TransformException, Buffer, TransformListener
import tf_transformations
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy
from casadi import atan, tan

class DynamicTrajectoryPublisher(Node):
    def __init__(self):
        super().__init__("dynamic_trajectory_publisher")

        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            depth=1
        )

        self.map_sub = self.create_subscription(OccupancyGrid, "/map", self.map_callback, qos)
        self.trajectory_pub = self.create_publisher(Path, "/dynamic_trajectory", qos)

        self.declare_parameter('wheelbase', 0.34)
        self.declare_parameter('horizon_length', 30)
        self.declare_parameter('dt', 0.1)
        self.declare_parameter('max_speed', 1.0)
        self.declare_parameter('max_steering_angle', np.pi / 4)

        self.L = self.get_parameter('wheelbase').get_parameter_value().double_value
        self.N = self.get_parameter('horizon_length').get_parameter_value().integer_value
        self.dt = self.get_parameter('dt').get_parameter_value().double_value
        self.v = self.get_parameter('max_speed').get_parameter_value().double_value
        self.theta_max = self.get_parameter('max_steering_angle').get_parameter_value().double_value

        self.map_info = None
        self.occupancy_grid = None
        self.safe_distance = 0.5  # meters

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.timer = self.create_timer(0.1, self.timer_callback)

    def map_callback(self, msg):
        self.map_info = msg.info
        grid = np.array(msg.data).reshape((msg.info.height, msg.info.width))
        binary_map = (grid == 0).astype(np.uint8)
        self.occupancy_grid = distance_transform(binary_map) * msg.info.resolution
        self.get_logger().info(f"Map received with shape {grid.shape}, resolution {msg.info.resolution:.3f}")

    def timer_callback(self):
        if self.occupancy_grid is None or self.map_info is None:
            self.get_logger().warn("Map not ready yet")
            return

        try:
            now = rclpy.time.Time()
            transform = self.tf_buffer.lookup_transform("map", "base_link", now)

            x = transform.transform.translation.x
            y = transform.transform.translation.y
            q = transform.transform.rotation
            theta = tf_transformations.euler_from_quaternion([q.x, q.y, q.z, q.w])[2]

            best_score = -np.inf
            best_traj = None
            best_delta = 0.0

            for delta in np.linspace(-self.theta_max, self.theta_max, 50):
                traj = self.simulate_rk4_trajectory(x, y, theta, delta)
                score = self.score_trajectory(traj)
                if score > best_score:
                    best_score = score
                    best_traj = traj
                    best_delta = delta

            if best_traj and len(best_traj) > 0:
                self.get_logger().info(f"Best delta: {np.round(best_delta, 3)} rad")
                self.publish_path(best_traj)

        except TransformException as ex:
            self.get_logger().warn(f"TF transform failed: {ex}")

    def wrap_angle(self, angle):
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def kinematic_model(self, state, delta):
        x, y, theta = state
        dx = self.v * np.cos(theta)
        dy = self.v * np.sin(theta)
        dtheta = (self.v / self.L) * np.tan(delta)
        return np.array([dx, dy, dtheta])

    def simulate_rk4_trajectory(self, x, y, theta, delta):
        state = np.array([x, y, theta])
        trajectory = []
        for _ in range(self.N):
            k1 = self.kinematic_model(state, delta)
            k2 = self.kinematic_model(state + 0.5 * self.dt * k1, delta)
            k3 = self.kinematic_model(state + 0.5 * self.dt * k2, delta)
            k4 = self.kinematic_model(state + self.dt * k3, delta)
            state += (self.dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            state[2] = self.wrap_angle(state[2])  # prevent yaw drift
            trajectory.append(tuple(state))
        return trajectory

    def score_trajectory(self, traj):
        score = 0.0
        min_dist = np.inf

        for x, y, _ in traj:
            mx = int((x - self.map_info.origin.position.x) / self.map_info.resolution)
            my = int((y - self.map_info.origin.position.y) / self.map_info.resolution)

            if 10 <= mx < self.occupancy_grid.shape[1] - 10 and 10 <= my < self.occupancy_grid.shape[0] - 10:
                dist = self.occupancy_grid[my, mx]
                min_dist = min(min_dist, dist)
                score += dist
            else:
                return -np.inf  # out of bounds

        if min_dist < self.safe_distance:
            return -np.inf

        goal_reward = self.goal_attractor_score(traj)
        score += goal_reward

        score -= 0.5 * self.curvature_cost(traj)
        return score

    def goal_attractor_score(self, traj):
        goal = np.array([3.5, -4.5])  # change this to your actual goal
        end = np.array([traj[-1][0], traj[-1][1]])
        return -np.linalg.norm(end - goal)

    def curvature_cost(self, traj):
        yaws = [theta for _, _, theta in traj]
        diffs = np.unwrap(np.diff(yaws))
        return np.sum(np.abs(diffs))

    def publish_path(self, traj):
        path = Path()
        path.header.frame_id = "map"
        path.header.stamp = self.get_clock().now().to_msg()

        for x, y, theta in traj:
            pose = PoseStamped()
            pose.header = path.header
            pose.pose.position.x = x
            pose.pose.position.y = y
            q = tf_transformations.quaternion_from_euler(0, 0, theta)
            pose.pose.orientation.x = q[0]
            pose.pose.orientation.y = q[1]
            pose.pose.orientation.z = q[2]
            pose.pose.orientation.w = q[3]
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
