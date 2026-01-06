import numpy as np 
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt


data = np.loadtxt(
    "out/racing_line_world.csv",
    delimiter=",",
    skiprows=1
)

print("Shape:", data.shape)  # should print (N, 4)
s, x, y, kappa = data.T
print("First few samples:", s[:3], x[:3], y[:3], kappa[:3])

s -= s[0] # normalize s to start from 0
s_uniform = np.linspace(0, s[-1], 2000)
x_spline=CubicSpline(s,x,bc_type='natural')
y_spline=CubicSpline(s,y,bc_type='natural')

x_ref=x_spline(s)
y_ref=y_spline(s)

# Computing first derivatives
dx_ds = x_spline(s_uniform, 1)
dy_ds = y_spline(s_uniform, 1)
d2x_ds2 = x_spline(s_uniform, 2)
d2y_ds2 = y_spline(s_uniform, 2)

phi_ref = np.arctan2(dy_ds, dx_ds)
kappa_ref = (dx_ds * d2y_ds2 - dy_ds * d2x_ds2) / (dx_ds**2 + dy_ds**2)**1.5

np.savetxt(
    "out/racing_line_smooth.csv",
    np.column_stack([s_uniform, x_ref, y_ref, phi_ref, kappa_ref]),
    delimiter=",",
    header="s_m,x_m,y_m,phi,kappa",
    comments=""
)

plt.figure(figsize=(8,8))
plt.plot(x, y, 'k.', label='original')
plt.plot(x_ref, y_ref, 'r-', label='cubic spline')
plt.axis('equal'); plt.legend(); plt.show()