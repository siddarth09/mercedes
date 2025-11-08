#!/usr/bin/env python3
# sac_f1tenth_inference_node.py
import os
import rclpy
import numpy as np
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import OccupancyGrid, Path, Odometry
from ackermann_msgs.msg import AckermannDriveStamped
from std_msgs.msg import Header

# SB3
from stable_baselines3 import SAC

from mercedes.sim.waypoints_handler import nearest_point_on_trajectory, calculate_curvatures


class SACF1TenthNode(Node):
    """
    Reconstructs the SAME 29-dim observation your F1TenthWrapper used in training:
      0: vx
      1: vy
      2: ang_v (yaw rate)
      3: dvx
      4: dvy
      5: dang_v
      6: yaw_dev (car yaw - path tangent yaw at nearest waypoint)  [-pi, pi]
      7:25  18 sparse LiDAR ranges
      25:  prev_steer_angle
      26:  path_curvature (from next N points)
      27:  collisions flag (0/1)
      28:  delta_yaw
    Then feeds it to SAC.predict(deterministic=True) and publishes /drive.
    """

    def __init__(self):
        super().__init__("sac_inference")

        # ---- Parameters (align these with your setup) --------------------------
        self.declare_parameter("model_path", "/home/deepak/Data/f1tenth/mercedes_ws/src/mercedes/storage/models/working/model.zip")
        self.declare_parameter("waypoints_path", "/home/deepak/Data/f1tenth/mercedes_ws/src/mercedes/storage/csc433_clean.csv")

        self.declare_parameter("scan_topic", "/scan")
        self.declare_parameter("odom_topic", "/ego_racecar/odom")
        self.declare_parameter("drive_topic", "/drive")
        self.declare_parameter("frame_id", "ego_racecar/base_link")

        self.declare_parameter("control_hz", 20.0)

        # lidar handling: clip + even downsample to 18 beams
        self.declare_parameter("scan_clip", 10.0)
        self.declare_parameter("collision_range_thresh", 0.25)  # if min range < thresh → collision flag=1
        self.declare_parameter("use_collision_heuristic", True)

        # vehicle safety limits (clip actions)
        self.declare_parameter("max_speed", 2.0)            # m/s
        self.declare_parameter("max_steering_angle", 0.4)   # rad

        # curvature lookahead points (training used N=10)
        self.declare_parameter("curvature_points", 10)

        # ---- Resolve params ----------------------------------------------------
        self.model_path  = self.get_parameter("model_path").get_parameter_value().string_value
        self.waypoints_path = self.get_parameter("waypoints_path").get_parameter_value().string_value

        self.scan_topic  = self.get_parameter("scan_topic").get_parameter_value().string_value
        self.odom_topic  = self.get_parameter("odom_topic").get_parameter_value().string_value
        self.drive_topic = self.get_parameter("drive_topic").get_parameter_value().string_value
        self.frame_id    = self.get_parameter("frame_id").get_parameter_value().string_value

        self.scan_clip   = float(self.get_parameter("scan_clip").value)
        self.col_thresh  = float(self.get_parameter("collision_range_thresh").value)
        self.use_col_heur= bool(self.get_parameter("use_collision_heuristic").value)

        self.max_speed   = float(self.get_parameter("max_speed").value)
        self.max_steer   = float(self.get_parameter("max_steering_angle").value)
        self.N_curv      = int(self.get_parameter("curvature_points").value)

        hz = float(self.get_parameter("control_hz").value)
        self.dt = 1.0 / max(1.0, hz)

        # ---- Load model --------------------------------------------------------
        assert os.path.exists(self.model_path), f"model.zip not found: {self.model_path}"
        self.get_logger().info(f"Loading SAC model: {self.model_path}")
        self.model = SAC.load(
            self.model_path,
            device="cpu",
            custom_objects={
                "lr_schedule": (lambda _: 3e-4),   # constant LR fallback
                "learning_rate": 3e-4,             # some SB3 versions read this directly
            },
        )

        # ---- Load waypoints (same parsing as training) ------------------------
        wp = self._load_waypoints(self.waypoints_path)
        self._waypoints = wp  # shape (M,2)

        # ---- Runtime state (to reproduce prev_* terms) ------------------------
        self.prev_vels = np.zeros(3, dtype=np.float32)   # [vx, vy, ang_v]
        self.prev_yaw  = 0.0
        self.prev_steer = 0.0
        self.prev_waypoint = self._waypoints[0].copy()
        self._current_index = 0

        # latest sensor values
        self._scan = None
        self._vx = 0.0
        self._vy = 0.0
        self._yaw = 0.0
        self._ang_v = 0.0
        self._occ = None
        self._map_info = None
        self._map_ready = False

        # ---- ROS I/O ----------------------------------------------------------
        qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
        )

        map_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,  # latched
            history=HistoryPolicy.KEEP_LAST,
        )

        self.map_sub = self.create_subscription(OccupancyGrid, "/map", self._cb_map, map_qos)
        self.create_subscription(LaserScan, self.scan_topic, self._cb_scan, qos)
        self.create_subscription(Odometry,  self.odom_topic, self._cb_odom, qos)

        self.pub_drive = self.create_publisher(AckermannDriveStamped, self.drive_topic, 10)

        self.timer = self.create_timer(self.dt, self._control_step)
        self.get_logger().info("SAC F1TENTH inference node started.")

    # ---------------- Waypoints (match training) -------------------------------
    def _load_waypoints(self, path):
        path = os.path.expanduser(path)
        path = os.path.abspath(path)
        try:
            data = np.genfromtxt(path, delimiter=",", comments="#", dtype=float, skip_header=1)
            if data.ndim == 1:
                data = np.atleast_2d(data)
            xs = data[:, 0]
            ys = data[:, 1]
            waypoints = np.stack([xs, ys], axis=1).astype(np.float32)
            if waypoints.shape[0] < 3:
                raise ValueError("Too few waypoints after load.")
            self.get_logger().info(f"Loaded {waypoints.shape[0]} waypoints from {path}")
            return waypoints
        except Exception as e:
            raise RuntimeError(f"Failed to load waypoints '{path}': {e}")

    # ---------------- Subscribers ---------------------------------------------
    def _cb_map(self, msg):
    # Store map as (H, W) uint8 with values in {0..100, 255}
        w = msg.info.width
        h = msg.info.height
        data = np.asarray(msg.data, dtype=np.int16).reshape(h, w)
        # Normalize unknowns (-1) -> 255, occupied >=50 -> 100, free <50 -> 0
        occ = np.where(data < 0, 255, data).astype(np.uint8)
        self._occ = occ
        self._map_info = msg.info
        self._map_ready = True

    def _cb_scan(self, _msg_ignored):
        """
        Ignore /scan and synthesize 18 evenly spaced beams via ray tracing on the map.
        Requires: self._map_ready, pose (_pos_x, _pos_y, _yaw). Produces self._scan (18,).
        """
        if not self._map_ready:
            # No map yet; fall back to 'no obstacles' at clip range
            self._scan = np.full(18, self.scan_clip, dtype=np.float32)
            return

        # Robot pose in world (updated in _cb_odom):
        x = getattr(self, "_pos_x", None)
        y = getattr(self, "_pos_y", None)
        yaw = getattr(self, "_yaw", None)
        if x is None or y is None or yaw is None:
            return

        # LiDAR model: 270° FOV centered on forward (base_link +x), 18 beams
        fov_rad = np.deg2rad(270.0)
        num_beams = 18
        # Angles from -fov/2 to +fov/2 relative to robot forward
        angles = np.linspace(-0.5 * fov_rad, 0.5 * fov_rad, num=num_beams, dtype=np.float32)

        max_r = float(self.scan_clip)
        ranges = np.empty(num_beams, dtype=np.float32)

        # Cast each beam in world frame (robot yaw + relative angle)
        for i, rel in enumerate(angles):
            th = yaw + float(rel)
            d = self._raycast_distance(x, y, th, max_r)
            # Clip and sanitize like your original code:
            if not np.isfinite(d):
                d = max_r
            d = np.clip(d, 0.0, max_r)
            ranges[i] = d

        # Exactly the 18 beams your obs expects:
        self._scan = ranges

    def _cb_odom(self, msg: Odometry):
        self._pos_x = float(msg.pose.pose.position.x)
        self._pos_y = float(msg.pose.pose.position.y)

        # Body-frame velocities (assumes odom twist is in base_link)
        self._vx = float(msg.twist.twist.linear.x)
        self._vy = float(msg.twist.twist.linear.y)
        self._ang_v = float(msg.twist.twist.angular.z)

        # Extract yaw from quaternion
        q = msg.pose.pose.orientation
        # yaw = atan2(2(wz + xy), 1 - 2(y^2 + z^2))
        # Use robust conversion:
        ysqr = q.y * q.y
        t3 = +2.0 * (q.w * q.z + q.x * q.y)
        t4 = +1.0 - 2.0 * (ysqr + q.z * q.z)
        self._yaw = np.arctan2(t3, t4)

    # --- Utility: world <-> map index ---
    def _world_to_map(self, x, y):
        """Return integer (i,j) indices into self._occ for world (x,y)."""
        info = self._map_info
        # world = origin + R * (i*res, j*res); map is axis-aligned, assume yaw=0 for map
        mx = int((x - info.origin.position.x) / info.resolution)
        my = int((y - info.origin.position.y) / info.resolution)
        # Note: OccupancyGrid data is row-major with (0,0) at map origin (bottom-left).
        # We'll index as occ[my, mx] with bounds checks.
        return mx, my

    def _cell_is_occupied(self, mx, my, occ_thresh=50):
        if self._occ is None:
            return False
        H, W = self._occ.shape
        if mx < 0 or my < 0 or mx >= W or my >= H:
            return True  # treat out-of-bounds as hit
        v = int(self._occ[my, mx])
        if v == 255:      # unknown -> treat as obstacle for safety
            return True
        return v >= occ_thresh
    
    # --- Core: raycast a single beam ---
    def _raycast_distance(self, x0, y0, theta, max_range):
        """DDA ray-march in world coords until hit or max_range (meters)."""
        if not self._map_ready:
            return max_range
        res = self._map_info.resolution
        step = 0.5 * res  # conservative step
        dist = 0.0
        cos_t = float(np.cos(theta))
        sin_t = float(np.sin(theta))
        while dist < max_range:
            px = x0 + dist * cos_t
            py = y0 + dist * sin_t
            mx, my = self._world_to_map(px, py)
            if self._cell_is_occupied(mx, my):
                return max(0.0, dist)
            dist += step
        return max_range

    # ---------------- Helpers to match training obs ----------------------------
    def _calc_current_waypoint_and_index(self, pos_xy):
        nearest_pt, _, _, idx = nearest_point_on_trajectory(pos_xy, self._waypoints)
        return nearest_pt.astype(np.float32), int(idx)

    def _yaw_dev_to_path(self, idx, yaw):
        next_idx = (idx + 1) % self._waypoints.shape[0]
        dx = self._waypoints[next_idx, 0] - self._waypoints[idx, 0]
        dy = self._waypoints[next_idx, 1] - self._waypoints[idx, 1]
        yaw_ref = np.arctan2(dy, dx)
        yaw_dev = yaw - yaw_ref
        return np.arctan2(np.sin(yaw_dev), np.cos(yaw_dev)).astype(np.float32)

    def _path_curvature(self, idx):
        N = self.N_curv
        ahead = np.zeros((N, 2), dtype=np.float32)
        M = self._waypoints.shape[0]
        for i in range(N):
            ahead[i, :] = self._waypoints[(idx + i) % M, :]
        curv = calculate_curvatures(ahead)  # same function as training
        return float(curv[0])

    def _flatten_action(self, action) -> np.ndarray:
        """
        Return action as a flat 1D float32 vector, regardless of whether it's
        a np.ndarray, (1,2), (2,), list/tuple of arrays, etc.
        """
        # already a numpy array
        if isinstance(action, np.ndarray):
            return action.astype(np.float32).reshape(-1)

        # list/tuple → concatenate each leaf as 1D
        if isinstance(action, (list, tuple)):
            parts = []
            for a in action:
                a = np.asarray(a, dtype=np.float32).reshape(-1)
                parts.append(a)
            if len(parts) == 0:
                return np.zeros(2, dtype=np.float32)
            return np.concatenate(parts, axis=0)

        # fallback: try to coerce
        try:
            return np.asarray(action, dtype=np.float32).reshape(-1)
        except Exception:
            # last resort: zero command
            return np.zeros(2, dtype=np.float32)

    def _extract_speed_steer(self, action) -> tuple[float, float]:
        a = self._flatten_action(action)
        if a.size < 2:
            # pad if somehow fewer than 2 scalars
            a = np.pad(a, (0, 2 - a.size))
        return float(a[0]), float(a[1])

    
    # ---------------- Control loop --------------------------------------------
    def _control_step(self):

        if self._scan is None:
            return
        
        pos_x = getattr(self, "_pos_x", 0.0)
        pos_y = getattr(self, "_pos_y", 0.0)

        # Compute nearest waypoint + index
        current_wp, idx = self._calc_current_waypoint_and_index(np.array([pos_x, pos_y], dtype=np.float32))
        self._current_index = idx

        # Yaw deviation
        yaw_dev = self._yaw_dev_to_path(idx, self._yaw)

        # downsampled scans
        depths = self._scan.astype(np.float32)  # shape (18,)

        # Curvature ahead
        path_curv = np.float32(self._path_curvature(idx))

        # Collision flag: if you don't have a contact sensor, emulate with LiDAR min range
        if self.use_col_heur:
            collisions_flag = np.float32(1.0 if np.min(depths) < self.col_thresh else 0.0)
        else:
            collisions_flag = np.float32(0.0)

        # Delta terms (prev_* from last step)
        dvx    = np.float32(self._vx - self.prev_vels[0])
        dvy    = np.float32(self._vy - self.prev_vels[1])
        dang_v = np.float32(self._ang_v - self.prev_vels[2])
        delta_yaw = np.float32(np.arctan2(np.sin(self._yaw - self.prev_yaw),
                                          np.cos(self._yaw - self.prev_yaw)))

        # Assemble obs (29,)
        obs = np.empty(29, dtype=np.float32)
        obs[0] = np.float32(self._vx)
        obs[1] = np.float32(self._vy)
        obs[2] = np.float32(self._ang_v)
        obs[3] = dvx
        obs[4] = dvy
        obs[5] = dang_v
        obs[6] = yaw_dev
        obs[7:25] = depths  # exactly 18 beams
        obs[25] = np.float32(self.prev_steer)
        obs[26] = path_curv
        obs[27] = collisions_flag
        obs[28] = delta_yaw

        # Sanity (mirror training guard)
        if not np.all(np.isfinite(obs)):
            self.get_logger().warn(f"Non-finite obs, skipping step. obs={obs}")
            return

        # Predict deterministic action
        try:
            action = self.model.predict(obs, deterministic=True)
        except Exception as e:
            self.get_logger().warn(f"Predict failed: {e}")
            return
        
        if not hasattr(self, "_logged_action_shape"):
            try:
                self.get_logger().info(f"predict() action repr sample: {repr(action)}")
            except Exception:
                pass
            self._logged_action_shape = True


        # Clip for safety:
        steer, speed = self._extract_speed_steer(action)
        speed_cmd = float(np.clip(speed,   0.3, self.max_speed))
        steer_cmd = float(np.clip(steer, -self.max_steer, self.max_steer))

        # Publish Ackermann
        msg = AckermannDriveStamped()
        msg.header = Header()
        msg.header.frame_id = self.frame_id
        msg.drive.speed = speed_cmd
        msg.drive.steering_angle = steer_cmd
        self.pub_drive.publish(msg)
        self.get_logger().info(f"Publishing drive - Speed:{speed_cmd} Steer:{steer_cmd}")

        # ---- Update "prev_*" to match training wrapper's _take_over() ----------
        self.prev_steer = steer_cmd
        self.prev_yaw   = float(self._yaw)
        self.prev_vels  = np.array([self._vx, self._vy, self._ang_v], dtype=np.float32)
        self.prev_waypoint = current_wp


def main():
    # reduce thread thrash on embedded CPUs
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"

    rclpy.init()
    node = SACF1TenthNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
