#!/usr/bin/env python3
import rclpy 
from rclpy.node import Node
import pandas as pd 
import numpy as np
from scipy.interpolate import BSpline
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
import os 


class TrajGene(Node):
    def __init__(self):
        super().__init__("trajectory_generator")
        storage_path="/home/siddarth/f1ws/src/mercedes/storage"
        csv_files=sorted([f for f in os.listdir(storage_path) if f.endswith('.csv')])
        
        latest_csv=os.path.join(storage_path,csv_files[-1])
        self.get_logger().info(f"Latest CSV file: {latest_csv}")
        
        #READING
        self.df=pd.read_csv(latest_csv,header=None,names=["x","y","z","roll","pitch","yaw"])
        
        
        
        self.path=self.create_publisher(Path,"/reference_trajectory",10)
        self.timer=self.create_timer(0.1,self.ref_callback)
        
            
    def ref_callback(self):
        #approximation for the loop
        x,y=self.df["x"].values,self.df["y"].values
    
        k = 3
        n = len(x)
        if n < k + 1:
            self.get_logger().error("Not enough waypoints for B-spline of degree 3")
            return

        # Clamp knot vector for full spline
        t = np.concatenate(([0]*k, np.linspace(0, 1, n - k + 1), [1]*k))

        self.spline_x = BSpline(t, x, k)
        self.spline_y = BSpline(t, y, k)

        
        u=np.linspace(0,1,100)
        x=self.spline_x(u)
        y=self.spline_y(u)
        
      
        
        self.path_msg=Path()
        self.path_msg.header.frame_id="map"
        for xi,yi in zip(x,y):
            pose=PoseStamped()
            pose.header.frame_id="map"
            pose.pose.position.x=float(xi)
            pose.pose.position.y=float(yi)
            pose.pose.position.z=0.0
            self.path_msg.poses.append(pose)
            
        self.path.publish(self.path_msg)
      
        
def main():
    rclpy.init()
    traj_gene=TrajGene()
    rclpy.spin(traj_gene)
    traj_gene.destroy_node()
    rclpy.shutdown()
    
if __name__=="__main__":
    main()