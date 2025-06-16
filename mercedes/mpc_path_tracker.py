#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import casadi as ca
import numpy as np
import pandas as pd
from ackermann_msgs.msg import AckermannDriveStamped
from builtin_interfaces.msg import Time
import tf_transformations
from tf2_ros import Buffer, TransformListener, TransformException
import math
import os
from glob import glob

class NMPCPathTracker(Node):
    def __init__(self):
        super().__init__('nmpc_path_tracker')

        # ─── ROS PARAMETERS ─────────────────────────────────
        self.csv_path      = self.declare_parameter('csv_path',    '/home/deepak/Data/f1tenth/mercedes_ws/src/mercedes/storage/wp-2025-06-16-02-38-38.csv').value
        self.N             = self.declare_parameter('horizon',       30).value    # prediction horizon
        self.dt            = self.declare_parameter('dt',           0.1).value    # timestep
        self.L             = self.declare_parameter('wheelbase',  0.330).value    # vehicle wheelbase
        self.v_max         = self.declare_parameter('v_max',        1.0).value    # max speed
        self.delta_max     = self.declare_parameter('steer_max',   0.34).value    # max steering angle (rad)
        self.Q_xy          = self.declare_parameter('Q_xy',         9.0).value    # position tracking cost
        self.R_v           = self.declare_parameter('R_v',         0.05).value    # speed cost
        self.R_delta       = self.declare_parameter('R_delta',     0.01).value    # steering cost
        self.R_acc         = self.declare_parameter('R_acc',        0.1).value    # acceleration rate cost
        self.R_steer_rate  = self.declare_parameter('R_steer_rate', 0.1).value    # steering rate cost
        # ────────────────────────────────────────────────────

        # Load & subsample waypoints
        df = pd.read_csv(self.csv_path)
        self.wp = df[['x','y']].to_numpy()
        # self.wp = self.wp[::5]

        # Setup TF2 listener for map → base_link
        self.tf_buffer   = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Ackermann publisher
        self.pub = self.create_publisher(AckermannDriveStamped, '/drive', 10)

        # Build the CasADi NLP solver once
        self._build_solver()

        # Timer driving the control loop at dt
        self.create_timer(self.dt, self.control_loop)

    def _lookup_pose(self):
        """Lookup base_link in map frame via TF; return (x, y, yaw) or None."""
        try:
            t = self.tf_buffer.lookup_transform(
                'map', 'base_link',
                Time(sec=0, nanosec=0),
                timeout=rclpy.duration.Duration(seconds=0.5)
            )
        except TransformException as ex:
            self.get_logger().warn(f"TF lookup failed: {ex}")
            return None

        x = t.transform.translation.x
        y = t.transform.translation.y
        q = t.transform.rotation
        yaw = tf_transformations.euler_from_quaternion((
            q.x, q.y, q.z, q.w
        ))[2]
        return np.array([x, y, yaw])

    def _build_solver(self):
        N, dt, L = self.N, self.dt, self.L
        Q, Rv, Rδ = self.Q_xy, self.R_v, self.R_delta
        Racc, Rsr = self.R_acc, self.R_steer_rate

        X = ca.SX.sym('X', 3, N+1)
        U = ca.SX.sym('U', 2, N)
        P = ca.SX.sym('P', 3 + 2*N)

        obj = 0
        g   = [X[:,0] - P[0:3]]

        for k in range(N):
            st      = X[:,k]
            v_k     = U[0,k]
            delta_k = U[1,k]
            x_ref   = P[3+2*k]
            y_ref   = P[3+2*k+1]

            obj += Q*((st[0]-x_ref)**2 + (st[1]-y_ref)**2)
            obj += Rv*v_k**2 + Rδ*delta_k**2
            if k>0:
                obj += Racc*(v_k-U[0,k-1])**2
                obj += Rsr *(delta_k-U[1,k-1])**2

            st_next = X[:,k+1]
            f = ca.vertcat(
                st[0] + v_k*ca.cos(st[2])*dt,
                st[1] + v_k*ca.sin(st[2])*dt,
                st[2] + v_k/L*ca.tan(delta_k)*dt
            )
            g.append(st_next - f)

        OPT = ca.vertcat(ca.reshape(X, -1,1), ca.reshape(U, -1,1))
        G   = ca.vertcat(*g)

        nlp = {'x':OPT, 'f':obj, 'g':G, 'p':P}
        opts = {'ipopt.print_level':0,'print_time':0}
        self.solver = ca.nlpsol('solver','ipopt',nlp,opts)

        n_con = G.size()[0]
        self.lbg = np.zeros(n_con)
        self.ubg = np.zeros(n_con)

        lbx, ubx = [], []
        for _ in range(N+1):
            lbx += [-ca.inf, -ca.inf, -ca.inf]
            ubx += [ ca.inf,  ca.inf,  ca.inf]
        for _ in range(N):
            lbx += [0.0,            -self.delta_max]
            ubx += [self.v_max,     self.delta_max]
        self.lbx = np.array(lbx)
        self.ubx = np.array(ubx)

        self.init_p    = np.zeros(3 + 2*N)
        self.init_xopt = np.zeros(((N+1)*3 + N*2))

        self.get_logger().info(f"NMPC solver built (horizon={N})")

    def control_loop(self):
        pose = self._lookup_pose()
        if pose is None:
            return

        # nearest waypoint
        d = np.linalg.norm(self.wp - pose[:2], axis=1)
        idx = int(np.argmin(d))

        # pack parameters
        p_val = np.zeros_like(self.init_p)
        p_val[0:3] = pose
        for k in range(self.N):
            i = min(idx+k, len(self.wp)-1)
            p_val[3+2*k]   = self.wp[i,0]
            p_val[3+2*k+1] = self.wp[i,1]

        # solve
        sol = self.solver(
            x0=self.init_xopt, p=p_val,
            lbg=self.lbg, ubg=self.ubg,
            lbx=self.lbx, ubx=self.ubx,
        )
        xopt = sol['x'].full().flatten()
        self.init_xopt = xopt

        offset = (self.N+1)*3
        v0     = float(xopt[offset])
        delta0 = float(xopt[offset+1])

        msg = AckermannDriveStamped()
        msg.drive.speed          = v0
        msg.drive.steering_angle = delta0
        self.pub.publish(msg)

    def destroy_node(self):
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = NMPCPathTracker()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
