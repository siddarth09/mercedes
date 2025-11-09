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
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from casadi import SX, vertcat, Function, nlpsol, diag, atan, tan, cos, sin
from ament_index_python.packages import get_package_share_directory

from mercedes.sim.curve import load_trajectory, create_smooth_spline, create_casadi_spline_from_lambda, nearest_s_xy

class MPCCNode(Node):

    def __init__(self):
        super().__init__('mpcc_sim')

        # Parameters
        self.declare_parameter('dt', 0.01)
        self.declare_parameter('N', 30)
        self.declare_parameter('wheelbase', 0.36)
        self.declare_parameter('half_width', 0.5)
        self.declare_parameter('safety_margin', 0.0)


        # Weights for the cost function
        self.declare_parameter('Q_norm',   5.0)   # penalize lateral offset strongly
        self.declare_parameter('Q_tang',   2.0)   # smaller, keep smooth progress along track
        self.declare_parameter('Q_psi',    2.0)   # heading alignment weight

        # Control effort weights
        self.declare_parameter('R_vel',    1.0)   # moderate penalty on speed magnitude
        self.declare_parameter('R_delta',  0.1)   # strong penalty on steering magnitude
        self.declare_parameter('R_ddelta', 1.0)   # big penalty on steering *rate* (smoothness)

        # Progress reward
        self.declare_parameter('Q_prog',   0.1)   # higher = drive forward faster

        # Terminal (end-of-horizon) error weights
        self.declare_parameter('Q_norm_K', 3.0)
        self.declare_parameter('Q_tang_K', 3.0)
        self.declare_parameter('Q_psi_K',  3.0)

        # Control input constraints
        self.declare_parameter('v_min', 0.3)
        self.declare_parameter('v_max', 1.5)
        self.declare_parameter('delta_min', -0.4)
        self.declare_parameter('delta_max', 0.4)

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

        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        # Subscriptions
        self.create_subscription(Odometry, '/ego_racecar/odom', self.odom_callback, 10)
        # Publishers
        self.drive_pub = self.create_publisher(AckermannDriveStamped, '/drive', 10)
        self.s_curve_pub = self.create_publisher(Path, '/s_curve', qos)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.base_frame = 'ego_racecar/base_link'
        self.map_frame = 'map'
        self.current_state = None
        self.current_vel = 0.0
        self.last_delta = 0.0
        self.prev_X = None   # shape (N+1, 4)
        self.prev_U = None   # shape (N, 2)
        self.have_warm = False


        dir = '/home/deepak/Data/f1tenth/mercedes_ws/src/mercedes'
        self.centerline_path = os.path.join(dir, 'storage', 'csc433_clean.csv')

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

        self.publish_smooth_curve(self.s_eval)
        self.s0 = self.s_eval[0]
        self.L = self.s_eval[-1]

    def publish_smooth_curve(self, s_eval: list):
        path = Path()
        path.header.frame_id = 'map'

        for s in s_eval:
            ps = PoseStamped()
            ps.header.frame_id = 'map'
            ps.pose.position.x = float(self.x_func(s))
            ps.pose.position.y = float(self.y_func(s))
            ps.pose.position.z = 0.0
            ps.pose.orientation.w = 1.0 
            path.poses.append(ps)

        self.s_curve_pub.publish(path)
        self.get_logger().info(f"s_curve published with {len(self.s_eval)} points")

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
        # wrap s once
        sw = self._wrap_s_sym(s) if isinstance(s, ca.SX) else self._wrap_s(float(s))

        # centerline
        x_ref = self.ca_splines['x'](sw)
        y_ref = self.ca_splines['y'](sw)

        # raw tangent
        tx = self.ca_splines['tx'](sw)
        ty = self.ca_splines['ty'](sw)

        # normalize safely: t = t / max(norm, eps)
        if isinstance(tx, ca.SX):
            tnorm = ca.sqrt(tx*tx + ty*ty)
            tnorm = ca.fmax(1e-9, tnorm)      # avoid divide-by-zero
            txu = tx / tnorm
            tyu = ty / tnorm
        else:
            tnorm = max(1e-9, math.hypot(float(tx), float(ty)))
            txu = float(tx)/tnorm
            tyu = float(ty)/tnorm

        nx = -tyu
        ny =  txu

        # position error
        ex = x - x_ref
        ey = y - y_ref

        e_norm = nx*ex + ny*ey
        e_tang = txu*ex + tyu*ey

        # heading error (robust)
        hx = ca.cos(psi) if isinstance(psi, ca.SX) else math.cos(float(psi))
        hy = ca.sin(psi) if isinstance(psi, ca.SX) else math.sin(float(psi))
        cos_epsi = txu*hx + tyu*hy
        sin_epsi = txu*hy - tyu*hx

        # atan2 is safe unless both are exactly zero; make sure that can't happen
        # (with the normalization above, they won't.)
        e_psi = ca.atan2(sin_epsi, cos_epsi) if isinstance(sin_epsi, ca.SX) \
                else math.atan2(float(sin_epsi), float(cos_epsi))

        # curvature
        kap = self.ca_splines['kappa_xy'](sw)

        return e_norm, e_tang, e_psi, kap


    def _f_cont(self, x, u):
        # x = [X, Y, psi, s], u = [v, delta]
        X, Y, psi, s = map(float, x)
        v, delta = map(float, u)

        # Wrap s and read geometry
        sw  = self._wrap_s(s)
        tx  = float(self.ca_splines['tx'](sw))   # unit tangent (already normalized)
        ty  = float(self.ca_splines['ty'](sw))
        nx, ny = -ty, tx                         # unit normal (left-positive)

        # Centerline position
        x_ref = float(self.ca_splines['x'](sw))
        y_ref = float(self.ca_splines['y'](sw))

        # Frenet lateral error
        ex, ey  = X - x_ref, Y - y_ref
        e_norm  = nx*ex + ny*ey

        # Heading alignment with tangent
        cos_epsi = tx*math.cos(psi) + ty*math.sin(psi)

        # Curvature and progress rate
        kap = float(self.ca_splines['kappa_xy'](sw))
        den = max(0.2, 1.0 - kap*e_norm)         # same floor you use in the NLP
        sdot = v * cos_epsi / den

        # Kinematic bicycle
        Xdot   = v * math.cos(psi)
        Ydot   = v * math.sin(psi)
        psidot = v * math.tan(delta) / self.wheelbase

        return np.array([Xdot, Ydot, psidot, sdot], dtype=float)

    def _rk4_step(self, x, u, dt):
        k1 = self._f_cont(x, u)
        k2 = self._f_cont(x + 0.5*dt*k1, u)
        k3 = self._f_cont(x + 0.5*dt*k2, u)
        k4 = self._f_cont(x + dt*k3, u)
        return x + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)


    def timer_callback(self):
        # Get (x,y,yaw,v) 
        self.update_state_from_tf()
        if self.current_state is None:
            self.get_logger().warn("Waiting for TF/odom to update the state ...")
            return
    
        x, y, psi, v = self.current_state
        
        s_est = nearest_s_xy(x, y, self.s_eval, self.x_func, self.y_func, is_closed=True)

        if s_est is None or not np.isfinite(s_est):
            # fall back to previous s (if warm) or to start of track
            if self.have_warm and self.prev_X is not None:
                s_est = float(self.prev_X[0, 3])
            else:
                s_est = float(self.s0)

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

            den = 1.0 - kap_k * e_norm_k
            den = ca.fmax(0.2, den)   # floor

            s_dot_k = v_k * ca.cos(e_psi_k) / den
            # s_dot_k = v_k * ca.cos(e_psi_k) / (1.0 - kap_k * e_norm_k + eps)
            cos_epsi = ca.cos(e_psi_k)

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


        # ---------- Build seed X_seed, U_seed ----------
        v_seed = float(np.clip(self.current_vel if self.current_vel > 0.1 else 0.8,
                            self.v_min, self.v_max))

        X_seed = np.zeros((self.N+1, 4), dtype=float)
        U_seed = np.zeros((self.N, 2), dtype=float)

        use_warm = (self.have_warm and
                    self.prev_X is not None and self.prev_U is not None and
                    self.prev_X.shape == (self.N+1, 4) and self.prev_U.shape == (self.N, 2))

        if use_warm:
            # Shift last solution forward; propagate terminal node with RK4
            X_seed[:-1, :] = self.prev_X[1:, :]
            X_seed[-1,  :] = self._rk4_step(X_seed[-2, :], self.prev_U[-1, :], self.dt)
            U_seed[:-1, :] = self.prev_U[1:, :]
            U_seed[-1,  :] = self.prev_U[-1, :]
        else:
            # Centerline guess at constant speed using CasADi interpolants
            for k in range(self.N+1):
                s_guess = self._wrap_s(s_est + k*self.dt*v_seed)

                # centerline position
                xg = float(self.ca_splines['x'](s_guess))
                yg = float(self.ca_splines['y'](s_guess))

                # unit tangent -> heading
                tx = float(self.ca_splines['tx'](s_guess))
                ty = float(self.ca_splines['ty'](s_guess))
                psi_g = float(np.arctan2(ty, tx))

                X_seed[k, :] = [xg, yg, psi_g, s_guess]

            U_seed[:, 0] = v_seed
            U_seed[:, 1] = self.last_delta

        # Enforce measured state at node 0
        X_seed[0, 0] = float(x)
        X_seed[0, 1] = float(y)
        X_seed[0, 2] = float(psi)
        X_seed[0, 3] = float(s_est)

        # Unwrap s along the horizon to keep continuity
        for k in range(1, self.N+1):
            ds = X_seed[k, 3] - X_seed[k-1, 3]
            if ds >  0.5*self.L:  X_seed[k, 3] -= self.L
            if ds < -0.5*self.L:  X_seed[k, 3] += self.L

        # Flatten AFTER building the seed; you’ll apply s-trust on X_seed later
        x0 = list(X_seed.flatten()) + list(U_seed.flatten())

        # s_trust = max(2.0, 0.5 * v_seed * self.N * self.dt)  # heuristic
        # for k in range(self.N+1):
        #     sk = X_seed[k, 3]
        #     lbx[4*k + 3] = sk - s_trust
        #     ubx[4*k + 3] = sk + s_trust

        # Sanity check the seed
        x0 = np.array(x0, dtype=float)
        if not np.all(np.isfinite(x0)):
            # Replace any inf/nan with measured state / safe defaults
            x0 = np.nan_to_num(x0, nan=0.0, posinf=0.0, neginf=0.0)
            x0[:4] = [float(self.measured_state[0]), float(self.measured_state[1]),
                    float(self.measured_state[2]), float(self.measured_state[3])]


        nlp = {'x':z, 'f':cost, 'g':g}
        solver = nlpsol('solver', 'ipopt', nlp, {
            'ipopt.print_level': 0, 'print_time': 0,
            'ipopt.tol': 1e-3,
            'ipopt.mu_strategy': 'adaptive',
            'ipopt.max_iter': 100,
            'ipopt.hessian_approximation': 'limited-memory',
            'ipopt.acceptable_tol': 1e-2,
            'ipopt.acceptable_iter': 5,
            'ipopt.warm_start_init_point': 'yes',
        })

        try:
            sol = solver(x0=x0, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)
            z_opt = sol['x'].full().flatten()

            X_opt = z_opt[:4*(self.N+1)].reshape(self.N+1, 4)
            U_opt = z_opt[4*(self.N+1):].reshape(self.N, 2)

            self.prev_X = X_opt
            self.prev_U = U_opt

            if not np.all(np.isfinite(self.prev_X)) or not np.all(np.isfinite(self.prev_U)):
                self.have_warm = False
            else:
                self.have_warm = True

            steer = float(z_opt[4 * (self.N + 1) + 1])
            speed = float(z_opt[4 * (self.N + 1)])
            self.last_delta = steer

            msg = AckermannDriveStamped()
            msg.drive.steering_angle = steer
            msg.drive.speed = speed

            self.drive_pub.publish(msg)
            # self.get_logger().info(f"Published control: speed={speed}, steer={steer}")

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

    


    