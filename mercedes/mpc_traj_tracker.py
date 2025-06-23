#!/usr/bin/env python3
import os, math, glob
import numpy as np, pandas as pd, casadi as ca
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped
from tf2_ros import Buffer, TransformListener, TransformException
from builtin_interfaces.msg import Time
import tf_transformations

class TrajMPC(Node):
    def __init__(self):
        super().__init__('traj_mpc')

        # ─── ROS PARAMETERS ─────────────────────────────────────
        self.csv_dir     = self.declare_parameter('csv_dir',    '/home/deepak/.../storage/').value
        self.N           = self.declare_parameter('horizon',     30).value
        self.dt          = self.declare_parameter('dt',         0.1).value
        self.L           = self.declare_parameter('wheelbase',  0.335).value
        self.v_max       = self.declare_parameter('v_max',       2.0).value
        self.delta_max   = self.declare_parameter('steer_max',   0.34).value
        self.a_lat_max   = self.declare_parameter('a_lat_max',   2.0).value  # max lateral accel
        self.Q_xy        = self.declare_parameter('Q_xy',        9.0).value
        self.Q_theta     = self.declare_parameter('Q_theta',     1.0).value
        self.Q_v         = self.declare_parameter('Q_v',         1.0).value
        self.R_v         = self.declare_parameter('R_v',         0.05).value
        self.R_delta     = self.declare_parameter('R_delta',     0.01).value
        self.R_acc       = self.declare_parameter('R_acc',       0.1).value
        self.R_delta_dot= self.declare_parameter('R_delta_dot',  0.1).value
        # ─────────────────────────────────────────────────────────

        # 1) load & parameterize trajectory + speed profile
        self._load_and_interpolate()

        # 2) TF listener for current pose
        self.tf_buf = Buffer()
        TransformListener(self.tf_buf, self)

        # 3) publisher & build solver
        self.pub = self.create_publisher(AckermannDriveStamped, '/drive', 10)
        self._build_solver()

        # 4) run MPC at dt
        self.create_timer(self.dt, self.control_loop)

    def _load_and_interpolate(self):
        # pick latest CSV
        files = glob.glob(os.path.join(self.csv_dir, '*.csv'))
        if not files:
            self.get_logger().error(f"No CSVs in {self.csv_dir}")
            rclpy.shutdown(); return
        path = max(files, key=os.path.getmtime)
        self.get_logger().info(f"Loading trajectory → {path}")

        df = pd.read_csv(path)
        x = df['x'].to_numpy()
        y = df['y'].to_numpy()
        # compute yaw if missing
        if 'yaw' in df:
            yaw = df['yaw'].to_numpy()
        else:
            dy, dx = np.diff(y), np.diff(x)
            yaw = np.arctan2(dy, dx)
            yaw = np.append(yaw, yaw[-1])

        # arc length
        ds = np.hypot(np.diff(x), np.diff(y))
        s  = np.concatenate(([0], np.cumsum(ds)))

        # 1st & 2nd derivatives w.r.t s
        dx_ds  = np.gradient(x, s)
        dy_ds  = np.gradient(y, s)
        d2x_ds = np.gradient(dx_ds, s)
        d2y_ds = np.gradient(dy_ds, s)

        # curvature kappa(s)
        κ = np.abs(dx_ds * d2y_ds - dy_ds * d2x_ds) / ((dx_ds**2 + dy_ds**2)**1.5 + 1e-6)

        # interpolators
        self.x_of_s       = lambda S: np.interp(S, s, x)
        self.y_of_s       = lambda S: np.interp(S, s, y)
        self.yaw_of_s     = lambda S: np.interp(S, s, np.unwrap(yaw))
        self.kappa_of_s   = lambda S: np.interp(S, s, κ)
        self.s_max        = s[-1]

        # speed profile: v_ref(s) = min(v_max, sqrt(a_lat_max/κ))
        self.vref_of_s = lambda S: np.minimum(
            self.v_max,
            np.sqrt(self.a_lat_max / (self.kappa_of_s(S) + 1e-6))
        )

    def _lookup_pose(self):
        try:
            tfm = self.tf_buf.lookup_transform(
                'map','base_link', Time(sec=0,nanosec=0),
                timeout=rclpy.duration.Duration(seconds=0.5))
        except TransformException:
            return None

        t, q = tfm.transform.translation, tfm.transform.rotation
        yaw = tf_transformations.euler_from_quaternion((q.x,q.y,q.z,q.w))[2]
        return np.array([t.x, t.y, yaw])

    def _build_solver(self):
        N, dt, L = self.N, self.dt, self.L
        Qxy, Qt, Qv = self.Q_xy, self.Q_theta, self.Q_v
        Rv, Rδ, Ra, Rδd = self.R_v, self.R_delta, self.R_acc, self.R_delta_dot

        # States = [x, y, θ, v], Controls = [v, δ]
        X = ca.SX.sym('X',  4, N+1)
        U = ca.SX.sym('U',  2, N)
        P = ca.SX.sym('P',  3 + 4*N)  # x0,y0,θ0 + (xref,yref,yawref,vref)×N

        obj = 0
        g   = [X[:,0] - P[0:3]]

        for k in range(N):
            st    = X[:,k]
            v_k   = U[0,k]
            δ_k   = U[1,k]
            # references
            xr    = P[3+4*k]
            yr    = P[3+4*k+1]
            θr    = P[3+4*k+2]
            vr    = P[3+4*k+3]

            # tracking cost
            obj += Qxy*((st[0]-xr)**2 + (st[1]-yr)**2)
            obj += Qt *(st[2]-θr)**2
            obj += Qv *(st[3]-vr)**2

            # control & rate cost
            obj += Rv  * v_k**2 + Rδ * δ_k**2
            if k>0:
                obj += Ra * (v_k - U[0,k-1])**2
                obj += Rδd* (δ_k - U[1,k-1])**2

            # kinematic bicycle + hold v
            st_next = X[:,k+1]
            f = ca.vertcat(
                st[0] + v_k * ca.cos(st[2])   * dt,
                st[1] + v_k * ca.sin(st[2])   * dt,
                st[2] + v_k / L * ca.tan(δ_k) * dt,
                v_k
            )
            g.append(st_next - f)

        # stack
        OPT = ca.vertcat(ca.reshape(X,-1,1), ca.reshape(U,-1,1))
        G   = ca.vertcat(*g)
        nlp = {'x':OPT, 'f':obj, 'g':G, 'p':P}
        self.solver = ca.nlpsol('solver','ipopt',nlp,
                                {'ipopt.print_level':0,'print_time':0})

        # bounds
        nc = G.size()[0]
        self.lbg = np.zeros(nc); self.ubg = np.zeros(nc)
        lbx, ubx = [], []
        for _ in range(N+1): lbx += [-ca.inf]*4; ubx += [ca.inf]*4
        for _ in range(N):    lbx += [0.0, -self.delta_max]; ubx += [self.v_max, self.delta_max]
        self.lbx, self.ubx = np.array(lbx), np.array(ubx)

        # placeholders
        self.p0    = np.zeros(3 + 4*N)
        self.x0opt = np.zeros(((N+1)*4 + N*2))

    def control_loop(self):
        pose = self._lookup_pose()
        if pose is None: return

        # project onto s via coarse sampling
        Sgrid = np.linspace(0, self.s_max, 500)
        pts   = np.vstack((self.x_of_s(Sgrid), self.y_of_s(Sgrid))).T
        idx   = int(np.argmin(np.linalg.norm(pts - pose[:2], axis=1)))
        s0    = Sgrid[idx]

        # pack parameter P
        p = np.zeros_like(self.p0)
        p[0:3] = pose
        for k in range(self.N):
            sk = min(s0 + (k+1)*self.dt*self.v_max, self.s_max)
            p[3+4*k:3+4*k+4] = [
                self.x_of_s(sk),
                self.y_of_s(sk),
                self.yaw_of_s(sk),
                self.vref_of_s(sk)
            ]

        sol = self.solver(x0=self.x0opt, p=p,
                          lbg=self.lbg, ubg=self.ubg,
                          lbx=self.lbx, ubx=self.ubx)
        solx = sol['x'].full().flatten()
        self.x0opt = solx  # warm start

        off = (self.N+1)*4
        v0, δ0 = float(solx[off]), float(solx[off+1])

        msg = AckermannDriveStamped()
        msg.drive.speed          = v0
        msg.drive.steering_angle = δ0
        self.pub.publish(msg)

def main():
    rclpy.init()
    node = TrajMPC()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__=='__main__':
    main()
