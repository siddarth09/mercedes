#!/usr/bin/env python3
import rclpy 
from rclpy.node import Node
import pandas as pd 
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
import os
from tf_transformations import quaternion_from_euler



class TrajGene(Node):
    def __init__(self):
        super().__init__("trajectory_generator")

        # Load latest CSV from storage directory
        storage_path = "/home/siddarth/f1ws/src/mercedes/storage"
        csv_files = sorted([f for f in os.listdir(storage_path) if f.endswith('.csv')])
        if not csv_files:
            self.get_logger().error("No CSV files found in storage directory.")
            return
        
        latest_csv = os.path.join(storage_path, csv_files[-1])
        self.get_logger().info(f"Using CSV file: {latest_csv}")

        # Expecting x, y, z, roll, pitch, yaw format
        self.df = pd.read_csv(latest_csv, header=None, names=["x", "y", "z", "roll", "pitch", "yaw"])

        self.path_pub = self.create_publisher(Path, "/reference_trajectory", 10)
        self.timer = self.create_timer(0.1, self.ref_callback)

    def ref_callback(self):
        path_msg = Path()
        path_msg.header.frame_id = "map"

        for i in range(len(self.df)):
            pose = PoseStamped()
            pose.header.frame_id = "map"
            pose.header.stamp = self.get_clock().now().to_msg()
            pose.pose.position.x = float(self.df.loc[i, "x"])
            pose.pose.position.y = float(self.df.loc[i, "y"])
            pose.pose.position.z = float(self.df.loc[i, "z"])
            # Convert roll, pitch, yaw to quaternion
            roll = float(self.df.loc[i, "roll"])
            pitch = float(self.df.loc[i, "pitch"])
            yaw = float(self.df.loc[i, "yaw"])
            q = quaternion_from_euler(roll, pitch, yaw)
            pose.pose.orientation.x = q[0]
            pose.pose.orientation.y = q[1]
            pose.pose.orientation.z = q[2]
            pose.pose.orientation.w = q[3]

            
            # Add the pose to the path message
            

            # Orientation is not required for pure pursuit, but included here for completeness
            pose.pose.orientation.w = 1.0  # neutral quaternion
            path_msg.poses.append(pose)

        self.path_pub.publish(path_msg)


def main():
    rclpy.init()
    traj_gene = TrajGene()
    rclpy.spin(traj_gene)
    traj_gene.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
