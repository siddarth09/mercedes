#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import casadi as ca
import numpy as np
import pandas as pd
from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry
import math

class NMPCPathTracker(Node):
    def __init__(self):
        super().__init__('nmpc_path_tracker')

        # ─── ROS PARAMETERS ─────────────────────────────────
        self.csv_path      = self.declare_parameter('csv_path',    '/home/deepak/Data/f1tenth/mercedes_ws/src/mercedes/storage/waypoints.csv').value
        self.N             = self.declare_parameter('horizon',       30).value     # prediction horizon
        self.dt            = self.declare_parameter('dt',           0.1).value     # timestep
        self.L             = self.declare_parameter('wheelbase',  0.335).value    # vehicle wheelbase
        self.v_max         = self.declare_parameter('v_max',        3.0).value    # max speed
        self.delta_max     = self.declare_parameter('steer_max',   0.34).value    # max steering angle (rad)
        self.Q_xy          = self.declare_parameter('Q_xy',         1.0).value    # position tracking cost
        self.R_v           = self.declare_parameter('R_v',         0.01).value  # speed cost
        self.R_delta       = self.declare_parameter('R_delta',     0.01).value    # steering cost
        self.R_acc         = self.declare_parameter('R_acc',        0.1).value    # acceleration rate cost
        self.R_steer_rate  = self.declare_parameter('R_steer_rate', 0.1).value    # steering rate cost
        # ────────────────────────────────────────────────────

        # Load and optionally subsample waypoints (map‐frame x,y)
        df = pd.read_csv(self.csv_path)
        self.wp = df[['x','y']].to_numpy()
        self.wp = self.wp[::5] # subsample

        # Robot pose holder
        self.x0 = None
        self.create_subscription(Odometry, '/ego_racecar/odom', self.odom_cb, 10)

        # Ackermann publisher
        self.pub = self.create_publisher(AckermannDriveStamped, '/drive', 10)

        # Build the CasADi NLP solver once
        self._build_solver()

        # Timer driving the control loop at dt
        self.create_timer(self.dt, self.control_loop)

    def odom_cb(self, msg: Odometry):
        """Cache current robot pose in map frame."""
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        yaw = math.atan2(
            2*(q.w*q.z + q.x*q.y),
            1 - 2*(q.y*q.y + q.z*q.z)
        )
        self.x0 = np.array([p.x, p.y, yaw])

    def _build_solver(self):
        """Constructs the CasADi NLP for the full nonlinear MPC with rate costs."""
        N, dt, L = self.N, self.dt, self.L
        Q, Rv, Rδ = self.Q_xy, self.R_v, self.R_delta
        Racc, Rsr = self.R_acc, self.R_steer_rate

        # Decision variables
        X = ca.SX.sym('X', 3, N+1)   # states: x, y, theta
        U = ca.SX.sym('U', 2, N)     # controls: v, delta
        P = ca.SX.sym('P', 3 + 2*N)  # parameters: init state + refs

        obj = 0
        g   = []

        # initial condition constraint
        g.append(X[:,0] - P[0:3])

        for k in range(N):
            # states and controls at step k
            st      = X[:,k]
            v_k     = U[0,k]
            delta_k = U[1,k]
            # reference waypoint at step k
            x_ref = P[3 + 2*k]
            y_ref = P[3 + 2*k + 1]

            # tracking cost
            obj += Q * ((st[0]-x_ref)**2 + (st[1]-y_ref)**2)
            # control effort cost
            obj += Rv * v_k**2 + Rδ * delta_k**2

            # rate-of-change costs
            if k > 0:
                v_prev     = U[0,k-1]
                delta_prev = U[1,k-1]
                obj += Racc * (v_k - v_prev)**2
                obj += Rsr  * (delta_k - delta_prev)**2

            # model dynamics: kinematic bicycle
            st_next = X[:,k+1]
            f = ca.vertcat(
                st[0] + v_k * ca.cos(st[2]) * dt,
                st[1] + v_k * ca.sin(st[2]) * dt,
                st[2] + v_k/L * ca.tan(delta_k) * dt
            )
            g.append(st_next - f)

        # flatten decision vars and constraints
        OPT = ca.vertcat(ca.reshape(X, -1,1), ca.reshape(U, -1,1))
        G   = ca.vertcat(*g)

        # NLP definition
        nlp = {'x': OPT, 'f': obj, 'g': G, 'p': P}
        opts = {'ipopt.print_level': 0, 'print_time': 0}
        self.solver = ca.nlpsol('solver', 'ipopt', nlp, opts)

        # bounds for equality constraints
        n_con = G.size()[0]
        self.lbg = np.zeros(n_con)
        self.ubg = np.zeros(n_con)

        # variable bounds
        lbx, ubx = [], []
        # states: no bounds
        for _ in range(N+1):
            lbx += [-ca.inf, -ca.inf, -ca.inf]
            ubx += [ ca.inf,  ca.inf,  ca.inf]
        # controls: 0 ≤ v ≤ v_max, -delta_max ≤ delta ≤ delta_max
        for _ in range(N):
            lbx += [0.0,            -self.delta_max]
            ubx += [self.v_max,     self.delta_max]

        self.lbx = np.array(lbx)
        self.ubx = np.array(ubx)

        # placeholders for parameters & warm-start
        self.init_p    = np.zeros(3 + 2*N)
        self.init_xopt = np.zeros(((N+1)*3 + N*2))

        self.get_logger().info(f"Nonlinear MPC solver built (horizon={N})")

    def control_loop(self):
        """Every dt: update parameters, solve MPC, publish first control."""
        if self.x0 is None:
            return

        # nearest waypoint index
        dists = np.linalg.norm(self.wp - self.x0[:2], axis=1)
        idx   = int(np.argmin(dists))

        # pack parameters: [x0,y0,theta0, x_ref0,y_ref0, ...]
        p_val = np.zeros_like(self.init_p)
        p_val[0:3] = self.x0
        for k in range(self.N):
            i = min(idx + k, len(self.wp)-1)
            p_val[3 + 2*k]   = self.wp[i,0]
            p_val[3 + 2*k+1] = self.wp[i,1]

        # solve NLP
        sol = self.solver(
            x0=self.init_xopt,
            p = p_val,
            lbg=self.lbg, ubg=self.ubg,
            lbx=self.lbx, ubx=self.ubx
        )

        xopt = sol['x'].full().flatten()
        self.init_xopt = xopt  # warm start

        # extract first control
        offset = (self.N+1)*3
        v0     = float(xopt[offset])
        delta0 = float(xopt[offset+1])

        # self.get_logger().info(f"v0={v0:.2f}, delta0={math.degrees(delta0):.1f}°")

        # publish
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
