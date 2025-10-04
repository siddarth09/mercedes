#!/usr/bin/env python3
import numpy as np
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped
from ackermann_msgs.msg import AckermannDriveStamped
from tf_transformations import euler_from_quaternion

from casadi import SX, vertcat, Function, nlpsol, diag, atan, tan, cos, sin

from tf2_ros import Buffer, TransformListener, LookupException, ConnectivityException, ExtrapolationException
from rclpy.time import Time



class MPCCTrajectoryTracker(Node):
    def __init__(self):
        super().__init__('mpcc_trajectory_tracker')

        self.declare_parameter('N', 30)
        self.declare_parameter('wheelbase', 0.34)
        self.declare_parameter('q_c', 20.0)
        self.declare_parameter('q_l', 1.0)
        self.declare_parameter('R_weights', [0.1, 0.1])
        self.declare_parameter('v_min', 0.2)
        self.declare_parameter('v_max', 0.5)
        self.declare_parameter('delta_min', -0.5)
        self.declare_parameter('delta_max', 0.5)
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('q_theta', 1.0)
        
        self.N = self.get_parameter('N').get_parameter_value().integer_value
        self.dt = 0.05
        self.L = self.get_parameter('wheelbase').get_parameter_value().double_value
        self.q_c = self.get_parameter('q_c').get_parameter_value().double_value
        self.q_l = self.get_parameter('q_l').get_parameter_value().double_value
        self.R = diag(SX(self.get_parameter('R_weights').get_parameter_value().double_array_value))
        self.v_min = self.get_parameter('v_min').get_parameter_value().double_value
        self.v_max = self.get_parameter('v_max').get_parameter_value().double_value
        self.delta_min = self.get_parameter('delta_min').get_parameter_value().double_value
        self.delta_max = self.get_parameter('delta_max').get_parameter_value().double_value
        self.base_frame = self.get_parameter('base_frame').get_parameter_value()
        
        self.q_theta = self.get_parameter('q_theta').get_parameter_value().double_value
        self.current_state = None
        self.ref_traj = []

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.map_frame = "map"
        self.base_frame = "ego_racecar/base_link"

        self.velocity = 0.0
        self.create_subscription(Odometry, "/odom", self.odom_callback, 10)
        self.create_subscription(Path, "/dynamic_trajectory", self.traj_callback, 10)
        self.drive_pub = self.create_publisher(AckermannDriveStamped, "/drive", 10)

        self.timer = self.create_timer(self.dt, self.timer_callback)
        self.get_logger().info("MPCC Tracker Started")
        
        # CasADi setup
        self.x0_param = SX.sym('x0', 4)                     # current state param
        self.ref_traj_param = SX.sym('ref', (self.N+1)*3)   # reference trajectory param
        self.build_solver()
        
        
    def odom_callback(self, msg):
        self.velocity = msg.twist.twist.linear.x

    def update_state_from_tf(self):
        try:
            t = self.tf_buffer.lookup_transform(self.map_frame, self.base_frame, Time())
            trans = t.transform.translation
            rot = t.transform.rotation
            _, _, yaw = euler_from_quaternion([rot.x, rot.y, rot.z, rot.w])
            self.current_state = np.array([trans.x, trans.y, yaw, self.velocity])
        except (LookupException, ConnectivityException, ExtrapolationException) as e:
            self.get_logger().warn(f"TF Lookup failed: {e}")

    def traj_callback(self, msg):
        self.ref_traj = []
        for pose in msg.poses:
            pos = pose.pose.position
            ori = pose.pose.orientation
            _, _, yaw = euler_from_quaternion([ori.x, ori.y, ori.z, ori.w])
            self.ref_traj.append(np.array([pos.x, pos.y, yaw]))
        self.ref_traj = self.ref_traj[:self.N + 1]

    
    def compute_errors(self, x, y, psi, ref):
        xr, yr, psir = ref
        dx = x - xr
        dy = y - yr
        ec = -sin(psir) * dx + cos(psir) * dy # lateral error
        el = cos(psir) * dx + sin(psir) * dy  # longitudinal error
        return ec, el

    
    def build_solver(self):
        x = SX.sym('x'); y = SX.sym('y'); psi = SX.sym('psi'); v = SX.sym('v')
        x_state = vertcat(x, y, psi, v)

        v_c = SX.sym('v_c'); delta = SX.sym('delta')
        u_ctrl = vertcat(v_c, delta)

        x_next = x + v_c * cos(psi) * self.dt
        y_next = y + v_c * sin(psi) * self.dt
        psi_next = psi + (v_c / self.L) * tan(delta) * self.dt
        v_next = v_c
        dynamics = Function('dynamics', [x_state, u_ctrl], [vertcat(x_next, y_next, psi_next, v_next)])

        X = [SX.sym(f'X_{k}', 4) for k in range(self.N + 1)]
        U = [SX.sym(f'U_{k}', 2) for k in range(self.N)]

        cost = 0
        constraints = [X[0] - self.x0_param]   # initial state constraint

        for k in range(self.N):
            ref_k = self.ref_traj_param[3*k:3*(k+1)]
            xr, yr, psir = ref_k[0], ref_k[1], ref_k[2]

            dx = X[k][0] - xr
            dy = X[k][1] - yr
            theta = X[k][2]

            ec = -sin(psir)*dx + cos(psir)*dy
            el =  cos(psir)*dx + sin(psir)*dy

            cost += self.q_c * ec**2 + self.q_l * el**2
            cost += -self.q_theta * theta

            if k > 0:
                du = U[k] - U[k-1]
                cost += du.T @ self.R @ du
            else:
                cost += U[k].T @ self.R @ U[k]

            X_next = dynamics(X[k], U[k])
            constraints.append(X[k+1] - X_next)

            # inequality: ec <= 0.5
            constraints.append(ec)

        z = vertcat(*X, *U)
        g = vertcat(*constraints)

        nlp = {'x': z,
            'p': vertcat(self.x0_param, self.ref_traj_param),
            'f': cost,
            'g': g}

        self.solver = nlpsol('solver', 'ipopt', nlp,
                            {'ipopt.print_level': 0, 'print_time': 0})
        self.z0 = None

        # store bounds for decision variables
        self.lbx = []
        self.ubx = []
        for _ in range(self.N + 1):
            self.lbx += [-np.inf, -np.inf, -np.inf, self.v_min]
            self.ubx += [ np.inf,  np.inf,  np.inf, self.v_max]
        for _ in range(self.N):
            self.lbx += [self.v_min, self.delta_min]
            self.ubx += [self.v_max, self.delta_max]


    def timer_callback(self):
        self.update_state_from_tf()
        if self.current_state is None or len(self.ref_traj) < self.N + 1:
            return

        # Prepare parameters (current state + reference trajectory flattened)
        x0 = self.current_state[:4]
        ref_flat = np.array(self.ref_traj[:self.N + 1]).flatten()

        # Concatenate parameters into one vector
        p = np.concatenate([x0, ref_flat])

        # Initialize warm start if first time
        if not hasattr(self, "z0") or self.z0 is None:
            self.z0 = np.zeros(self.solver.size_in("x0"))

        # Bounds
        

        try:
            n_g, _ = self.solver.size_out('g')

            # equality constraints for dynamics
            lbg_dyn = [0.0] * (4*(self.N+1))
            ubg_dyn = [0.0] * (4*(self.N+1))

            # inequality constraints ec <= 0.5
            num_ineq = n_g - len(lbg_dyn)
            lbg_ineq = [-np.inf] * num_ineq
            ubg_ineq = [0.5] * num_ineq

            lbg = lbg_dyn + lbg_ineq
            ubg = ubg_dyn + ubg_ineq

            sol = self.solver(
                x0=self.z0,
                lbx=self.lbx,
                ubx=self.ubx,
                lbg=lbg,
                ubg=ubg,
                p=p,
            )
            z_opt = sol["x"].full().flatten()
            self.z0 = z_opt  # save for warm start

            steer = float(z_opt[4 * (self.N + 1) + 1])
            speed = float(z_opt[4 * (self.N + 1)])

            msg = AckermannDriveStamped()
            msg.drive.steering_angle = steer
            msg.drive.speed = speed
            self.drive_pub.publish(msg)

            self.get_logger().info(f"Published MPCC control: speed={speed:.2f}, steer={steer:.2f}")

        except Exception as e:
            self.get_logger().warn(f"MPCC solver failed: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = MPCCTrajectoryTracker()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
