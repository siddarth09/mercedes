#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from tf2_ros import Buffer, TransformListener, TransformException
from tf_transformations import euler_from_quaternion
from rclpy.duration import Duration
from time import gmtime, strftime
from os.path import expanduser
import atexit
import os

class WaypointsLogger(Node):
    def __init__(self):
        super().__init__('waypoints_logger')

        # Create output directory and CSV file
        home = expanduser('~')
        log_dir = "/home/deepak/Data/f1tenth/mercedes_ws/src/mercedes/storage"
        os.makedirs(log_dir, exist_ok=True)
        file_name = strftime('wp-%Y-%m-%d-%H-%M-%S.csv', gmtime())
        self.file_path = os.path.join(log_dir, file_name)
        self.file = open(self.file_path, 'w')

        # ✅ Write CSV header
        self.file.write("x,y,yaw\n")

        self.get_logger().info(f'Saving waypoints to {self.file_path}')

        # TF2 setup
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Timer to log at fixed rate
        self.timer = self.create_timer(0.2, self.log_waypoint_from_tf)

        # Ensure file is closed on shutdown
        atexit.register(self.shutdown_hook)

    def log_waypoint_from_tf(self):
        try:
            # Get latest transform from map → base_link
            transform = self.tf_buffer.lookup_transform(
                'map', 'base_link', rclpy.time.Time(), Duration(seconds=0.5)
            )

            x = transform.transform.translation.x
            y = transform.transform.translation.y
            q = transform.transform.rotation
            quat = [q.x, q.y, q.z, q.w]
            _, _, yaw = euler_from_quaternion(quat)

            self.get_logger().info(f"x={x:.2f}, y={y:.2f}, yaw={yaw:.2f}")
            self.file.write(f"{x},{y},{yaw}\n")
            self.file.flush()

        except TransformException as ex:
            self.get_logger().warn(f"TF lookup failed: {ex}")
            return

    def shutdown_hook(self):
        if not self.file.closed:
            self.file.close()
            self.get_logger().info("Waypoint log file closed.")

def main(args=None):
    rclpy.init(args=args)
    node = WaypointsLogger()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.shutdown_hook()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
