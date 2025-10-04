import numpy as np
import pandas as pd
import casadi as ca
from scipy.interpolate import UnivariateSpline, CubicSpline
from scipy.integrate import cumulative_trapezoid
import matplotlib.pyplot as plt

def load_trajectory(csv_path):
    with open(csv_path) as f:
        for line in f:
            if not line.startswith("#"):
                n_cols = len(line.strip().split(","))
                break

    if n_cols == 4:  # old format
        df = pd.read_csv(csv_path,
                        delimiter=',',
                        comment='#',
                        header=0,
                        names=['x_m', 'y_m', 'w_tr_right_m', 'w_tr_left_m'])
        return df['x_m'].values, df['y_m'].values
    
    elif n_cols == 3:  # new format
        df = pd.read_csv(csv_path,
                        delimiter=',',
                        comment='#',
                        header=0,
                        names=['x', 'y', 'yaw'])
        return df['x'].values, df['y'].values
    else:
        raise ValueError(f"Unsupported CSV format with {n_cols} columns")


def detect_loop_closure(x, y, tolerance=0.1, n_points_check=10):
    """
    Detect if the trajectory forms a loop by checking if any of the last n points are close to the start.
    Fully vectorized implementation.
    
    Args:
        x, y: trajectory coordinates
        tolerance: maximum distance for considering points as the same
        n_points_check: number of end points to check for loop closure (default: 10)
    
    Returns:
        is_closed: boolean indicating if loop closure is detected
        closure_distance: minimum distance between start and checked end points
        closure_index: index of the best closure point (-1 if no closure)
    """
    
    n_check = min(n_points_check, len(x) - 1)
    end_points = np.column_stack((x[-n_check:], y[-n_check:]))
    start_point = np.array([x[0], y[0]])

    distances = np.linalg.norm(end_points - start_point, axis=1)
    min_idx = np.argmin(distances)
    min_distance = distances[min_idx]
    
    closure_index = len(x) - n_check + min_idx
    
    is_closed = min_distance < tolerance
    
    # Debug information
    print(f"Loop closure check: Checking last {n_check} points")
    print(f"Minimum distance: {min_distance:.6f} at index {closure_index} (point {closure_index - len(x)})")
    if n_check > 1:
        print(f"Distance range: [{distances.min():.6f}, {distances.max():.6f}]")
    
    return is_closed, min_distance, closure_index

def compute_arc_length(x, y, closed=False, closure_index=None):
    """
    Compute arc length parameterization of the trajectory.
    
    Args:
        x, y: trajectory coordinates
        closed: whether to treat as closed curve
        closure_index: index where to close the loop (if None and closed=True, append first point)
    
    Returns:
        s: arc length parameter
        x_param: x coordinates (possibly trimmed/extended for closure)
        y_param: y coordinates (possibly trimmed/extended for closure)
    """
    if closed:
        if closure_index is not None and closure_index < len(x) - 1:
            # Trim trajectory at the closure point and add first point to close
            x = x[:closure_index + 1]
            y = y[:closure_index + 1]
        
        # Add the first point at the end to close the loop
        x = np.append(x, x[0])
        y = np.append(y, y[0])
    
    # Compute differential arc length using vectorized operations
    dx = np.diff(x)
    dy = np.diff(y)
    ds = np.sqrt(dx**2 + dy**2)
    
    # Cumulative arc length (starting from 0)
    s = np.zeros(len(x))
    s[1:] = np.cumsum(ds)
    
    return s, x, y

def create_smooth_spline(x, y, tolerance=0.1, n_points_check=10, smoothing_factor=None, num_points=1000):
    """
    Create smooth splines parameterized by arc length with loop closure detection.
    Also creates a curvature function.
    
    Args:
        x, y: input trajectory coordinates
        tolerance: maximum distance for loop closure detection
        n_points_check: number of end points to check for loop closure
        smoothing_factor: smoothing parameter for spline (None for automatic)
        num_points: number of points for evaluation
    
    Returns:
        x_func: lambda function x(s)
        y_func: lambda function y(s)
        kappa_func: lambda function κ(s) for curvature
        s_eval: arc length evaluation points
        is_closed: whether the curve is closed
        closure_info: dictionary with closure details
    """
    
    # Remove duplicate consecutive points using vectorized operations
    points = np.column_stack((x, y))
    mask = np.ones(len(points), dtype=bool)
    mask[1:] = np.any(points[1:] != points[:-1], axis=1)
    x = x[mask]
    y = y[mask]
    
    # Detect loop closure
    is_closed, closure_dist, closure_index = detect_loop_closure(x, y, tolerance, n_points_check)
    
    closure_info = {
        'is_closed': is_closed,
        'distance': closure_dist,
        'index': closure_index if is_closed else -1,
        'trimmed_points': len(x) - closure_index - 1 if is_closed and closure_index < len(x) - 1 else 0
    }
    
    print(f"Loop closure detection: {'CLOSED' if is_closed else 'OPEN'} (distance: {closure_dist:.6f})")
    if is_closed and closure_index < len(x) - 1:
        print(f"Trimming {closure_info['trimmed_points']} points after closure at index {closure_index}")
    
    # Compute arc length parameterization
    s, x_param, y_param = compute_arc_length(x, y, closed=is_closed, closure_index=closure_index if is_closed else None)
    
    if is_closed:
        # For closed curves, ensure periodicity
        period = s[-1]
        
        # Remove the duplicate last point for fitting
        s_fit = s[:-1]
        x_fit = x_param[:-1]
        y_fit = y_param[:-1]
        
        # Create periodic splines using CubicSpline with periodic boundary conditions
        x_spline = CubicSpline(s_fit, x_fit, bc_type='periodic')
        y_spline = CubicSpline(s_fit, y_fit, bc_type='periodic')
        
        # Create lambda functions for position
        x_func = lambda s_val: x_spline(np.asarray(s_val) % period)
        y_func = lambda s_val: y_spline(np.asarray(s_val) % period)
        
        # Create lambda function for curvature using spline derivatives
        def kappa_func(s_val):
            s_mod = np.asarray(s_val) % period
            dx_ds = x_spline.derivative(1)(s_mod)
            dy_ds = y_spline.derivative(1)(s_mod)
            d2x_ds2 = x_spline.derivative(2)(s_mod)
            d2y_ds2 = y_spline.derivative(2)(s_mod)
            
            numerator = np.abs(dx_ds * d2y_ds2 - dy_ds * d2x_ds2)
            denominator = (dx_ds**2 + dy_ds**2)**(3/2)
            return numerator / (denominator + 1e-10)
        
        # Evaluation points
        s_eval = np.linspace(0, period, num_points)
        
    else:
        # For open curves, use regular splines
        if smoothing_factor is None:
            # Automatic smoothing based on number of points
            smoothing_factor = len(x) * 0.01
        
        # Create splines with degree 3 or 4 for smoother derivatives
        k = min(4, len(s) - 1)  # Use degree 4 if possible for smoother curvature
        x_spline = UnivariateSpline(s, x_param, s=smoothing_factor, k=k)
        y_spline = UnivariateSpline(s, y_param, s=smoothing_factor, k=k)
        
        # Create lambda functions for position
        x_func = lambda s_val: x_spline(s_val)
        y_func = lambda s_val: y_spline(s_val)
        
        # Create lambda function for curvature using spline derivatives
        def kappa_func(s_val):
            s_val = np.asarray(s_val)
            dx_ds = x_spline.derivative(1)(s_val)
            dy_ds = y_spline.derivative(1)(s_val)
            d2x_ds2 = x_spline.derivative(2)(s_val)
            d2y_ds2 = y_spline.derivative(2)(s_val)
            
            numerator = np.abs(dx_ds * d2y_ds2 - dy_ds * d2x_ds2)
            denominator = (dx_ds**2 + dy_ds**2)**(3/2)
            return numerator / (denominator + 1e-10)
        
        # Evaluation points
        s_eval = np.linspace(0, s[-1], num_points)
    
    return x_func, y_func, kappa_func, s_eval, is_closed, closure_info

def compute_heading(x_func, y_func, s_eval):
    """
    Compute heading angle θ(s) along the spline.
    
    Args:
        x_func, y_func: lambda functions for x(s) and y(s)
        s_eval: arc length evaluation points
    
    Returns:
        theta: heading angles in radians
    """
    h = 1e-6
    dx_ds = (x_func(s_eval + h) - x_func(s_eval - h)) / (2 * h)
    dy_ds = (y_func(s_eval + h) - y_func(s_eval - h)) / (2 * h)
    theta = np.arctan2(dy_ds, dx_ds)
    return theta


def create_casadi_spline_from_lambda(x_func, y_func, kappa_func, s_eval, half_width, safety_margin, is_closed=False):
    """
    Create CasADi interpolant functions directly from lambda functions and s_eval.
    
    Args:
        x_func, y_func, kappa_func: Lambda functions from create_smooth_spline
        s_eval: Arc length evaluation points from create_smooth_spline
        is_closed: Whether the curve is closed
    
    Returns:
        Dictionary with CasADi functions and interpolants
    """
    
    # Evaluate the lambda functions at s_eval points
    x_points = x_func(s_eval)
    y_points = y_func(s_eval)
    kappa_points = kappa_func(s_eval)
    
    # Create CasADi interpolants
    
    # Bspline interpolation (smoother) - requires at least 4 points
    x_interp_bsp = ca.interpolant('x_bsp', 'bspline', [s_eval], x_points)
    y_interp_bsp = ca.interpolant('y_bsp', 'bspline', [s_eval], y_points)
    kappa_interp_bsp = ca.interpolant('kappa_bsp', 'bspline', [s_eval], kappa_points)
    
    # Create wrapper functions that handle periodicity if needed]
    s_max = float(s_eval[-1])
    s0    = float(s_eval[0])

    def wrap_open_sym(s):
        # clamp to [s0, s_max]
        return ca.fmin(ca.fmax(s, s0), s_max)

    def wrap_closed_sym(s):
        # modulo wrap into [s0, s0 + s_max - s0) == [s0, s_max)
        # use floor-based wrap to keep it CasADi-friendly
        return (s - s0) - ca.floor((s - s0)/(s_max - s0))*(s_max - s0) + s0

    wrap_sym = wrap_closed_sym if is_closed else wrap_open_sym

    # === Build scalar CasADi Functions ===
    s = ca.SX.sym('s')

    sw   = wrap_sym(s)
    x_s  = x_interp_bsp(sw)
    y_s  = y_interp_bsp(sw)
    kap_s= kappa_interp_bsp(sw)

    # First derivatives wrt s (exact for the bspline)
    dx_s  = ca.gradient(x_s, s)    # == ca.gradient(x_s, s) for scalar
    dy_s  = ca.gradient(y_s, s)

    # Second derivatives wrt s
    ddx_s = ca.gradient(dx_s, s)
    ddy_s = ca.gradient(dy_s, s)

    # Unit tangent t = [tx, ty]
    tnorm = ca.sqrt(dx_s*dx_s + dy_s*dy_s) + 1e-12
    tx    = dx_s / tnorm
    ty    = dy_s / tnorm

    # Curvature from x,y (independent of kappa_func if you want)
    # kappa = (x' y'' - y' x'') / (x'^2 + y'^2)^(3/2)
    kappa_geom = (dx_s*ddy_s - dy_s*ddx_s) / (tnorm**3)

    # ---- Yaw (phi/psi) curve from tangent ----
    psi_s     = ca.atan2(ty, tx)        # path yaw
    cos_psi_s = tx                       # cos(psi) = tx
    sin_psi_s = ty                       # sin(psi) = ty
    dpsi_ds   = kappa_geom               # since s ~ arc-length

    w = ca.DM(max(0, half_width - safety_margin))
    nx = -ty
    ny = tx

    xL_s = x_s + nx*w
    yL_s = y_s + ny*w
    xR_s = x_s - nx*w
    yR_s = y_s - ny*w

    out = {
        'x':           ca.Function('x_ca',           [s], [x_s]),
        'y':           ca.Function('y_ca',           [s], [y_s]),
        'kappa':       ca.Function('kappa_ca',       [s], [kap_s]),
        'dx':          ca.Function('dx_ca',          [s], [dx_s]),
        'dy':          ca.Function('dy_ca',          [s], [dy_s]),
        'ddx':         ca.Function('ddx_ca',         [s], [ddx_s]),
        'ddy':         ca.Function('ddy_ca',         [s], [ddy_s]),
        'tx':          ca.Function('tx_ca',          [s], [tx]),
        'ty':          ca.Function('ty_ca',          [s], [ty]),
        'kappa_xy':    ca.Function('kappa_xy_ca',    [s], [kappa_geom]),
        # yaw (phi/psi) curve and its derivative
        'psi':         ca.Function('psi_ca',      [s], [psi_s]),
        'cos_psi':     ca.Function('cospsi_ca',   [s], [cos_psi_s]),
        'sin_psi':     ca.Function('sinpsi_ca',   [s], [sin_psi_s]),
        'dpsi_ds':     ca.Function('dpsids_ca',   [s], [dpsi_ds]),
        # constant width functions (useful for constraints)
        'wL':          ca.Function('wL_ca',          [s], [w]),
        'wR':          ca.Function('wR_ca',          [s], [w]),
        # boundary curves (for plotting / RViz)
        'xL':          ca.Function('xL_ca',          [s], [xL_s]),
        'yL':          ca.Function('yL_ca',          [s], [yL_s]),
        'xR':          ca.Function('xR_ca',          [s], [xR_s]),
        'yR':          ca.Function('yR_ca',          [s], [yR_s]),
        's_wrap_closed': is_closed
    }
    return out


def nearest_s_xy(xp, yp, s_eval, x_func, y_func,
                 is_closed=True, max_it=8, tol=1e-10):
    """
    Find arc-length s on the spline (x(s), y(s)) closest to point p=(xp,yp).
    Only needs x_func,y_func; derivatives are computed via FD.
    """
    s0 = float(s_eval[0]); s1 = float(s_eval[-1]); L = s1 - s0
    mean_ds = L / max(4, len(s_eval)-1)
    h = 0.5 * mean_ds  # FD step tied to sampling

    def wrap(s):
        if is_closed:
            return (s - s0) % L + s0
        else:
            return float(np.clip(s, s0, s1))

    def d1(f, s, h):  # central 1st derivative
        return (f(s + h) - f(s - h)) / (2.0*h)

    def d2(f, s, h):  # central 2nd derivative
        return (f(s + h) - 2.0*f(s) + f(s - h)) / (h*h)

    # 1) coarse candidate
    xs = x_func(s_eval); ys = y_func(s_eval)
    i0 = int(np.argmin((xs - xp)**2 + (ys - yp)**2))
    s = float(s_eval[i0])

    # 2) Newton on g(s) = (r(s)-p)·r'(s) = 0
    for _ in range(max_it):
        xr  = float(x_func(s)); yr  = float(y_func(s))
        dx  = float(d1(x_func, s, h)); dy  = float(d1(y_func, s, h))
        ddx = float(d2(x_func, s, h)); ddy = float(d2(y_func, s, h))

        rx = xr - xp; ry = yr - yp
        g  = rx*dx + ry*dy
        gp = dx*dx + dy*dy + rx*ddx + ry*ddy

        if gp == 0.0:
            break
        s_new = wrap(s - g/gp)
        if abs(s_new - s) < tol:
            s = s_new
            break
        s = s_new

    return wrap(s)
