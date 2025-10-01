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

class MPCCNode(Node):

    def __init__(self):
        super().__init__('mpcc_sim')

        # Parameters
        self.declare_parameter('dt', 0.01)
        self.declare_parameter('N', 50)
        self.declare_parameter('wheelbase', 0.36)

        # Weights for the cost function
        self.declare_parameter('Q_norm', 5.0)
        self.declare_parameter('Q_tang', 8.0)
        self.declare_parameter('Q_psi', 2.0)
        self.declare_parameter('R_acc', 1.2)
        self.declare_parameter('R_delta', 10.0)
        self.declare_parameter('Q_prog', 5.0)
        self.declare_parameter('Q_norm_K', 10.0)
        self.declare_parameter('Q_psi_K', 8.0)

        self.declare_parameter('v_min', 0.0)
        self.declare_parameter('v_max', 5.0)

        # Control input constraints
        self.declare_parameter('delta_min', -0.4)
        self.declare_parameter('delta_max', 0.4)
        self.declare_parameter('R_ddelta', 50.0) # steering rate (big)

        # Retrieve parameters
        self.dt = self.get_parameter('dt').get_parameter_value().double_value
        self.N = self.get_parameter('N').get_parameter_value().integer_value
        self.wheelbase = self.get_parameter('wheelbase').get_parameter_value().double_value

        self.Q_norm = self.get_parameter('Q_norm').get_parameter_value().double_value
        self.Q_tang = self.get_parameter('Q_tang').get_parameter_value().double_value
        self.Q_psi = self.get_parameter('Q_psi').get_parameter_value().double_value
        self.R_acc = self.get_parameter('R_acc').get_parameter_value().double_value
        self.R_delta = self.get_parameter('R_delta').get_parameter_value().double_value
        self.Q_prog = self.get_parameter('Q_prog').get_parameter_value().double_value
        self.Q_norm_K = self.get_parameter('Q_norm_K').get_parameter_value().double_value
        self.Q_psi_K = self.get_parameter('Q_psi_K').get_parameter_value().double_value

        self.v_min = self.get_parameter('v_min').get_parameter_value().double_value
        self.v_max = self.get_parameter('v_max').get_parameter_value().double_value
        self.delta_min = self.get_parameter('delta_min').get_parameter_value().double_value
        self.delta_max = self.get_parameter('delta_max').get_parameter_value().double_value
        self.R_ddelta = self.get_parameter('R_ddelta').get_parameter_value().double_value

        # Subscriptions
        self.create_subscription(Odometry, '/ego_racecar/odom', self.odom_callback, 10)
        # Publishers
        self.drive_pub = self.create_publisher(AckermannDriveStamped, '/drive', 10)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.base_frame = 'ego_racecar/base_link'
        self.map_frame = 'map'
        self.current_state = None
        self.current_vel = 0.0
        self.last_delta = 0.0

        dir = '/home/deepak/Data/f1tenth/mercedes_ws/src/mercedes'
        self.raceline_path = os.path.join(dir, 'storage','racelines', 'Spielberg_raceline.csv')

        self.create_timer(self.dt, self.timer_callback)

        try:
            self.read_raceline_csv()
            self.build_spatial_splines()
            self.get_logger().info(f"Raceline loaded....") 
            self.build_casadi_interpolants()
            self.get_logger().info("CasADi interpolants ready....")
        except Exception as e:
            self.get_logger().error(f"Failed to load/build raceline: {e}")


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
        
    def read_raceline_csv(self):
        df = pd.read_csv(self.raceline_path, 
                        delimiter=';', 
                        comment='#',   # This tells pandas to skip lines starting with #
                        header=0,      # Use first non-comment line as header
                        names=['s_m', 'x_m', 'y_m', 'psi_rad', 'kappa_radpm', 'vx_mps', 'ax_mps2'])

        self.s = df['s_m'].values
        self.x = df['x_m'].values
        self.y = df['y_m'].values
        self.psi = df['psi_rad'].values
        self.kappa = df['kappa_radpm'].values

        self.s0 = float(self.s[0])
        self.L = float(self.s[-1] - self.s[0])

    
    def build_spatial_splines(self):
        """
        Builds C^2 cubic splines for xr(s), yr(s), psir(s), kappa(s).
        - Unwraps psi before fitting, wraps back to [-pi, pi] at query.
        - Wraps query s with modulo L into [s0, s0+L).
        Builds functions: xr(s), yr(s), psir(s), kappa(s).
        """
        psi_unw = np.unwrap(self.psi)
        bc = "not-a-knot"

        self._xs = CubicSpline(self.s, self.x, bc_type=bc)
        self._ys = CubicSpline(self.s, self.y, bc_type=bc)
        self._ps = CubicSpline(self.s, psi_unw, bc_type=bc)
        self._ks = CubicSpline(self.s, self.kappa, bc_type=bc)

        # Bind lightweight callables for use elsewhere in the node
        self.xr   = lambda sq: self._xs(self._wrap_s(sq))
        self.yr   = lambda sq: self._ys(self._wrap_s(sq))
        self.psir = lambda sq: self._wrap_pi(self._ps(self._wrap_s(sq)))
        self.kappar = lambda sq: self._ks(self._wrap_s(sq))

    def _wrap_s(self, sq):
        return (sq - self.s0) % self.L + self.s0

    @staticmethod
    def _wrap_pi(a):
        return np.arctan2(np.sin(a), np.cos(a))
    
    def build_casadi_interpolants(self):
        """
        Build CasADi symbolic interpolants vs arc-length s:
        self.xr_fun(s), self.yr_fun(s), self.psir_fun(s), self.kap_fun(s)
        Used *inside* CasADi graphs (dynamics, cost, constraints).
        """
        # unwrap heading to avoid 2π jumps for the interpolant coefficients
        psi_unw = np.unwrap(self.psi)

        # CasADi BSpline (or 'linear') interpolants; grid must be strictly increasing
        self.xr_fun   = ca.interpolant('xr',   'bspline', [self.s], self.x)
        self.yr_fun   = ca.interpolant('yr',   'bspline', [self.s], self.y)
        self.psir_fun = ca.interpolant('psir', 'bspline', [self.s], psi_unw)
        self.kap_fun  = ca.interpolant('kap',  'bspline', [self.s], self.kappa)

    # symbolic modulo wrapper 
    def _wrap_s_sym(self, s_sym):
        return (s_sym - self.s0) - ca.floor((s_sym - self.s0) / self.L) * self.L + self.s0
    
    def find_s_from_xy(self, x, y, return_closest=True, tolerance=1e-2):
        """
        Find arc-length s given position (x,y) 
        """
        # Calculate Euclidean distances to all points
        distances = np.sqrt((self.x - x)**2 + (self.y - y)**2)
        
        # Find index of minimum distance
        min_idx = np.argmin(distances)
        min_distance = distances[min_idx]
        
        # Check if within tolerance
        if min_distance <= tolerance:
            return self.s[min_idx]
        elif return_closest:
            # Return closest point's s even if outside tolerance
            # print(f"Warning: Closest point is {min_distance:.4f}m away (tolerance={tolerance})")
            return self.s[min_idx]
        else:
            # No match found within tolerance
            return None


    def get_errors(self, x, y, psi, s):

        try:
            s_w = self._wrap_s_sym(s)
        except Exception:
            s_w = self._wrap_s(s)  

        x_ref = self.xr_fun(s_w)
        y_ref = self.yr_fun(s_w)
        psi_ref = self.psir_fun(s_w)
        kap   = self.kap_fun(s_w)

        dx = x-x_ref
        dy = y-y_ref
        pos_err = ca.vertcat(dx,dy)

        tang_ref = ca.vertcat(ca.cos(psi_ref), ca.sin(psi_ref))
        norm_ref = ca.vertcat(-ca.sin(psi_ref), ca.cos(psi_ref))

        e_norm = ca.dot(norm_ref, pos_err)
        e_tang = ca.dot(tang_ref, pos_err)
        e_psi = ca.atan2(ca.sin(psi - psi_ref), ca.cos(psi - psi_ref))

        return e_norm, e_tang, e_psi, kap


    def timer_callback(self):
        # Get (x,y,yaw,v) 
        self.update_state_from_tf()
        if self.current_state is None:
            self.get_logger().warn("Waiting for TF/odom to update the state ...")
            return
    
        x, y, psi, current_vel = self.current_state
        
        s_est = self.find_s_from_xy(x,y)
        self.get_logger().info(f"Current s is {s_est}")
        if s_est is None:
            self.get_logger().warn("Could not find nearest s! Skipping this tick.")
            return

        self.measured_state = np.array([x, y, psi, s_est, current_vel])

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
        s_dot   = v_u * ca.cos(e_psi) / (1.0 - kap * e_norm + eps)
        s_next  = s + s_dot * self.dt

        next_state = ca.vertcat(x_next, y_next, psi_next, s_next)
        dynamics = ca.Function('dynamics', [x_state, u_ctrl], [next_state])

        X = [ca.SX.sym(f'X_{k}', 4) for k in range(self.N+1)]
        U = [ca.SX.sym(f'U_{k}', 2) for k in range(self.N)]

        cost = 0
        constraints = [X[0] - ca.DM(self.measured_state[:4])]
        
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
            s_dot_k = v_k * ca.cos(e_psi_k) / (1.0 - kap_k * e_norm_k + eps)

            # Stage cost
            cost += self.Q_norm*e_norm_k**2 + self.Q_tang*e_tang_k**2 + self.Q_psi*e_psi_k**2
            cost += self.R_acc*v_k**2 + self.R_delta*delta_k**2  # Regularization
            cost += -self.Q_prog*s_dot_k # Reward progress
            cost += self.R_ddelta * ddel**2 #Steering rate penalty

            X_next = dynamics(Xk, Uk)
            constraints.append(X[k+1] - X_next)
        
        # Terminal cost
        XN = X[self.N]
        xN  = XN[0]
        yN  = XN[1]
        psiN= XN[2]
        sN  = XN[3]

        e_norm_N, e_tang_N, e_psi_N, kap_N = self.get_errors(xN, yN, psiN, sN)

        cost += self.Q_norm_K*e_norm_N**2 + self.Q_psi_K*e_psi_N**2

        z = vertcat(*X, *U)
        g = vertcat(*constraints)

        nlp = {'x':z, 'f':cost, 'g':g}
        solver = nlpsol('solver', 'ipopt', nlp, {
            'ipopt.print_level': 0,
            'print_time': 0,
            'ipopt.tol': 1e-3,
        })

        lbx = []
        ubx = []

        for _ in range(self.N+1):
            lbx += [-np.inf, -np.inf, -np.inf, -np.inf]
            ubx += [np.inf, np.inf, np.inf, np.inf]
        for _ in range(self.N):
            lbx += [self.v_min, self.delta_min]
            ubx += [self.v_max, self.delta_max]

        lbg = [0.0] * ((self.N+1)*4)
        ubg = [0.0] * ((self.N+1)*4)

        x0 = [*self.measured_state[:4]] * (self.N+1) + [0.5, 0.0] * (self.N)

        try:
            sol = solver(x0=x0, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)
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

    


    