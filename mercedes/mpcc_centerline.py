#!/usr/bin/env python3
import os, math, glob
import numpy as np
import pandas as pd
import casadi as ca
import rclpy
import tf_transformations
from rclpy.node import Node
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped
from tf2_ros import Buffer, TransformListener, TransformException
from rclpy.time import Time
from tf_transformations import euler_from_quaternion
from tf2_ros import Buffer, TransformListener, LookupException, ConnectivityException, ExtrapolationException
from scipy.interpolate import CubicSpline

from casadi import SX, vertcat, Function, nlpsol, diag, atan, tan, cos, sin
from ament_index_python.packages import get_package_share_directory

from mercedes.sim.curve import load_trajectory, create_smooth_spline, create_casadi_spline_from_lambda, nearest_s_xy

class MPCCNode(Node):

    def __init__(self):
        super().__init__('mpcc_centerline')

        # Parameters
        self.declare_parameter('dt', 0.01)
        self.declare_parameter('N', 30)
        self.declare_parameter('wheelbase', 0.36)
        self.declare_parameter('half_width', 0.5)
        self.declare_parameter('safety_margin', 0.0)

        # # Weights for the cost function
        # self.declare_parameter('Q_norm', 8.0)
        # self.declare_parameter('Q_tang', 6.0)
        # self.declare_parameter('Q_psi', 6.0)
        # self.declare_parameter('R_vel', 0.8)
        # self.declare_parameter('R_delta', 10.0)
        # self.declare_parameter('Q_prog', 2.0)
        # self.declare_parameter('Q_norm_K', 6.0)
        # self.declare_parameter('Q_tang_K', 6.0)
        # self.declare_parameter('Q_psi_K', 6.0)

        # # Control input constraints
        # self.declare_parameter('v_min', 0.5)
        # self.declare_parameter('v_max', 2.0)
        # self.declare_parameter('delta_min', -0.4)
        # self.declare_parameter('delta_max', 0.4)
        # self.declare_parameter('R_ddelta', 100.0) # steering rate (big)

        # Weights for the cost function
        self.declare_parameter('Q_norm', 3.0)
        self.declare_parameter('Q_tang', 2.0)
        self.declare_parameter('Q_psi', 6.0)
        self.declare_parameter('R_vel', 0.5)
        self.declare_parameter('R_delta', 0.5)
        self.declare_parameter('Q_prog', 3.0)
        self.declare_parameter('Q_norm_K', 2.0)
        self.declare_parameter('Q_tang_K', 2.0)
        self.declare_parameter('Q_psi_K', 6.0)

        # Control input constraints
        self.declare_parameter('v_min', 0.3)
        self.declare_parameter('v_max', 0.8)
        self.declare_parameter('delta_min', -0.4)
        self.declare_parameter('delta_max', 0.4)
        self.declare_parameter('R_ddelta', 0.5) # steering rate (big)

        # Retrieve parameters
        self.dt = self.get_parameter('dt').get_parameter_value().double_value
        self.N = self.get_parameter('N').get_parameter_value().integer_value
        self.wheelbase = self.get_parameter('wheelbase').get_parameter_value().double_value
        self.half_width = self.get_parameter('half_width').get_parameter_value().double_value
        self.safety_margin = self.get_parameter('safety_margin').get_parameter_value().double_value

        self.Q_norm = self.get_parameter('Q_norm').get_parameter_value().double_value
        self.Q_tang = self.get_parameter('Q_tang').get_parameter_value().double_value
        self.Q_psi = self.get_parameter('Q_psi').get_parameter_value().double_value
        self.R_vel = self.get_parameter('R_vel').get_parameter_value().double_value
        self.R_delta = self.get_parameter('R_delta').get_parameter_value().double_value
        self.Q_prog = self.get_parameter('Q_prog').get_parameter_value().double_value
        self.Q_norm_K = self.get_parameter('Q_norm_K').get_parameter_value().double_value
        self.Q_tang_K = self.get_parameter('Q_tang_K').get_parameter_value().double_value
        self.Q_psi_K = self.get_parameter('Q_psi_K').get_parameter_value().double_value

        self.v_min = self.get_parameter('v_min').get_parameter_value().double_value
        self.v_max = self.get_parameter('v_max').get_parameter_value().double_value
        self.delta_min = self.get_parameter('delta_min').get_parameter_value().double_value
        self.delta_max = self.get_parameter('delta_max').get_parameter_value().double_value
        self.R_ddelta = self.get_parameter('R_ddelta').get_parameter_value().double_value

        # Subscriptions
        self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        # Publishers
        self.drive_pub = self.create_publisher(AckermannDriveStamped, '/drive', 10)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.base_frame = 'base_link'
        self.map_frame = 'map'
        self.current_state = None
        self.current_vel = 0.0
        self.last_delta = 0.0

        dir = '/home/deepak/Data/f1tenth/mercedes_ws/src/mercedes'
        # self.centerline_path = os.path.join(dir, 'storage', 'racelines', 'Spielberg_centerline.csv')
        self.centerline_path = '/home/deepak/Data/f1tenth/mercedes_ws/src/mercedes/storage/wp-2025-09-24-15-48-56.csv'

        self.create_timer(self.dt, self.timer_callback)

        x, y = load_trajectory(self.centerline_path)
        self.get_logger().info(f"CSV loaded form {self.centerline_path}")

        self.x_func, self.y_func, self.kappa_func, self.s_eval, self.is_closed, self.closure_info = create_smooth_spline(x, y)
        self.get_logger().info("Lambda functions ready")
        if self.is_closed:
            self.get_logger().info('The curve is closed')

        self.ca_splines = create_casadi_spline_from_lambda(self.x_func, self.y_func, self.kappa_func, self.s_eval, self.half_width, self.safety_margin, self.is_closed)
        if self.ca_splines is not None:
            self.get_logger().info("Casadi interpolants ready")

        self.s0 = self.s_eval[0]
        self.L = self.s_eval[-1]

    def odom_callback(self, msg : Odometry):
        x_vel = msg.twist.twist.linear.x
        y_vel = msg.twist.twist.linear.y
        self.current_vel = math.hypot(x_vel, y_vel)

    def update_state_from_tf(self):
        try:
            t = self.tf_buffer.lookup_transform(self.map_frame, self.base_frame, Time())
            trans = t.transform.translation
            rot = t.transform.rotation
            _,_,yaw = euler_from_quaternion([rot.x, rot.y, rot.z, rot.w])
            self.current_state = np.array([trans.x, trans.y, yaw, self.current_vel])
        except (LookupException, ConnectivityException, ExtrapolationException) as e:
            self.get_logger().warn(f"TF Lookup failed: {e}")
        
    def _wrap_s(self, sq):
        return (sq - self.s0) % self.L + self.s0

    @staticmethod
    def _wrap_pi(a):
        return np.arctan2(np.sin(a), np.cos(a))
    
    # symbolic modulo wrapper 
    def _wrap_s_sym(self, s_sym):
        return (s_sym - self.s0) - ca.floor((s_sym - self.s0) / self.L) * self.L + self.s0

    def get_errors(self, x, y, psi, s):
        # wrap/clip s once, then use the same 'sw' everywhere
        try:
            sw = self._wrap_s_sym(s)
        except Exception:
            sw = s  # if you're calling this in numpy context, but here it's CasADi

        # centerline point
        x_ref = self.ca_splines['x'](sw)
        y_ref = self.ca_splines['y'](sw)

        # unit tangent and normal
        tx = self.ca_splines['tx'](sw)
        ty = self.ca_splines['ty'](sw)
        nx = -ty
        ny =  tx

        # position error vector
        ex = x - x_ref
        ey = y - y_ref

        # Frenet errors
        e_norm = nx*ex + ny*ey         # lateral error (left positive)
        e_tang = tx*ex + ty*ey         # along-track error (optional)

        # heading error via dot/cross with tangent (no psi_ref needed)
        hx = ca.cos(psi)
        hy = ca.sin(psi)
        cos_epsi = tx*hx + ty*hy
        sin_epsi = tx*hy - ty*hx       # 2D cross(t, h)
        e_psi = ca.atan2(sin_epsi, cos_epsi)

        # curvature at s (use kappa from geometry or your kappa spline)
        kap = self.ca_splines['kappa_xy'](sw)  # or self.ca_splines['kappa'](sw)

        return e_norm, e_tang, e_psi, kap


    def timer_callback(self):
        # Get (x,y,yaw,v) 
        self.update_state_from_tf()
        if self.current_state is None:
            self.get_logger().warn("Waiting for TF/odom to update the state ...")
            return
    
        x, y, psi, v = self.current_state
        
        s_est = nearest_s_xy(x, y, self.s_eval, self.x_func, self.y_func, is_closed=True)

        if s_est is None:
            self.get_logger().warn("Could not find nearest s! Skippins this tick")
            return

        self.measured_state = np.array([x, y, psi, s_est, v])

        x   = ca.SX.sym('x')
        y   = ca.SX.sym('y')
        psi = ca.SX.sym('psi')
        s   = ca.SX.sym('s')
        x_state = ca.vertcat(x, y, psi, s)

        u_ctrl  = ca.SX.sym('u_ctrl', 2)  # single 2x1 control vector symbol
        v_u, d_u = u_ctrl[0], u_ctrl[1]   # extract speed and steering    

        x_next = x + v_u*ca.cos(psi) * self.dt
        y_next = y + v_u*ca.sin(psi) * self.dt
        psi_next = psi + ((v_u*ca.tan(d_u))/self.wheelbase) * self.dt
        
        e_norm, e_tang, e_psi, kap = self.get_errors(x, y, psi, s)
        
        eps = 1e-6
        # s_dot = v_u * cos_epsi / (1.0 - kap*e_norm + eps)
        s_dot   = v_u * ca.cos(e_psi) / (1.0 - kap * e_norm + eps)
        s_next  = s + s_dot * self.dt

        next_state = ca.vertcat(x_next, y_next, psi_next, s_next)
        dynamics = ca.Function('dynamics', [x_state, u_ctrl], [next_state])

        X = [ca.SX.sym(f'X_{k}', 4) for k in range(self.N+1)]
        U = [ca.SX.sym(f'U_{k}', 2) for k in range(self.N)]

        cost = 0
        constraints = []
        lbg = []
        ubg = []

        constraints.append(X[0] - ca.DM(self.measured_state[:4]))
        lbg += [0.0, 0.0, 0.0, 0.0]
        ubg += [0.0, 0.0, 0.0, 0.0]
        
        for k in range(self.N):

            Xk = X[k]
            Uk = U[k]

            xk  = Xk[0]
            yk  = Xk[1]
            psik= Xk[2]
            sk  = Xk[3]

            v_k     = Uk[0]
            delta_k = Uk[1]
            if k == 0:
                delta_prev = ca.DM(self.last_delta)
            else:
                delta_prev = U[k-1][1]
            ddel = delta_k - delta_prev

            e_norm_k, e_tang_k, e_psi_k, kap_k = self.get_errors(xk, yk, psik, sk)

            eps = 1e-6
            tx_k = self.ca_splines['tx'](sk)
            ty_k = self.ca_splines['ty'](sk)
            den = 1.0 - kap_k * e_norm_k
            den = ca.fmax(0.2, den)   # floor

            s_dot_k = v_k * ca.cos(e_psi_k) / den
            # s_dot_k = v_k * ca.cos(e_psi_k) / (1.0 - kap_k * e_norm_k + eps)
            cos_epsi = tx_k*ca.cos(psik) + ty_k*ca.sin(psik)

            # Stage cost
            cost += self.Q_norm*e_norm_k**2 + self.Q_tang*e_tang_k**2 # Contour errors
            # cost += + self.Q_psi*e_psi_k**2
            cost += self.R_vel*v_k**2 + self.R_delta*delta_k**2       # Regularization
            cost += -self.Q_prog*s_dot_k                              # Reward progress
            cost += self.R_ddelta * ddel**2                           # Steering rate penalty
            cost += self.Q_psi * (1.0 - cos_epsi)                     # bounded, smooth
            

            X_next = dynamics(Xk, Uk)
            constraints.append(X[k+1] - X_next)
            lbg += [0.0, 0.0, 0.0, 0.0]
            ubg += [0.0, 0.0, 0.0, 0.0]

            constraints.append(e_norm_k - self.half_width)
            lbg += [-ca.inf]
            ubg += [0.0]

            constraints.append(-e_norm_k - self.half_width)
            lbg += [-ca.inf]
            ubg += [0.0]

        # Terminal cost
        XN = X[self.N]
        xN  = XN[0]
        yN  = XN[1]
        psiN= XN[2]
        sN  = XN[3]

        e_norm_N, e_tang_N, e_psi_N, kap_N = self.get_errors(xN, yN, psiN, sN)
        cost += self.Q_norm_K*e_norm_N**2 + self.Q_tang_K*e_tang_N**2 + self.Q_psi_K*e_psi_N**2

        z = vertcat(*X, *U)
        g = vertcat(*constraints)

        lbx = []
        ubx = []

        for _ in range(self.N+1):
            lbx += [-np.inf, -np.inf, -np.inf, -np.inf]
            ubx += [np.inf, np.inf, np.inf, np.inf]
        for _ in range(self.N):
            lbx += [self.v_min, self.delta_min]
            ubx += [self.v_max, self.delta_max]

        z0 = [*self.measured_state[:4]] * (self.N+1) + [0.5, 0.0] * (self.N) # initial guess for all decision variables

        nlp = {'x':z, 'f':cost, 'g':g}
        solver = nlpsol('solver', 'ipopt', nlp, {
            'ipopt.print_level': 0,
            'print_time': 0,
            'ipopt.tol': 1e-3,
        })

        try:
            sol = solver(x0=z0, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)
            z_opt = sol['x'].full().flatten()
            steer = float(z_opt[4 * (self.N + 1) + 1])
            speed = float(z_opt[4 * (self.N + 1)])
            self.last_delta = steer

            msg = AckermannDriveStamped()
            msg.drive.steering_angle = steer
            msg.drive.speed = speed

            self.drive_pub.publish(msg)
            self.get_logger().info(f"Published control: speed={speed}, steer={steer}")

        except Exception as e:
            self.get_logger().warn(f"MPC Solver failed: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = MPCCNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

    


    