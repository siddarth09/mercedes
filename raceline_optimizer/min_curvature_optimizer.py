"""
JAX based minimum curvature optimizer.     
"""

import jax
import jax.numpy as jnp

import optax 
import numpy as np 
import matplotlib.pyplot as plt 


centerline = np.loadtxt("out/centerline_world.csv",delimiter=",",skiprows=1)
x_c,y_c= centerline[:,0],centerline[:,1] 
s = np.insert(np.cumsum(np.sqrt(np.diff(x_c)**2 + np.diff(y_c)**2)),0,0)
track_half_width= 1.0 


def curvature_cost(delta):
    # Uniformize arc length
    s_uniform = jnp.linspace(0.0, 1.0, len(x_c))
    x_interp = jnp.interp(s_uniform, s, x_c)
    y_interp = jnp.interp(s_uniform, s, y_c)

    # Tangent and normal
    dx = jnp.gradient(x_interp, s_uniform)
    dy = jnp.gradient(y_interp, s_uniform)
    n = jnp.stack([-dy, dx], axis=1)
    n = n / (jnp.linalg.norm(n, axis=1, keepdims=True) + 1e-8)

    # Offset line
    x = x_interp + delta * n[:, 0]
    y = y_interp + delta * n[:, 1]

    # Curvature
    dx = jnp.gradient(x, s_uniform)
    dy = jnp.gradient(y, s_uniform)
    ddx = jnp.gradient(dx, s_uniform)
    ddy = jnp.gradient(dy, s_uniform)
    kappa = (dx * ddy - dy * ddx) / ((dx ** 2 + dy ** 2) ** 1.5 + 1e-8)
    J_curv = jnp.mean(kappa ** 2)

    # Smoothness regularization on δ″(s)
    d2 = jnp.gradient(jnp.gradient(delta, s_uniform), s_uniform)
    J_smooth = jnp.mean(d2 ** 2)

    # Soft lateral constraint
    penalty = 1e3 * jnp.mean(jnp.maximum(0, jnp.abs(delta) - track_half_width) ** 2)

    return J_curv + 50.0 * J_smooth + penalty


delta0= jnp.zeros(len(x_c))
loss_grad=jax.grad(curvature_cost)
schedule = optax.cosine_decay_schedule(init_value=1e-3, decay_steps=2000, alpha=0.05)
optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adam(schedule)
)
opt_state = optimizer.init(delta0)

@jax.jit
def step(delta, opt_state):
    grads = loss_grad(delta)
    updates, opt_state = optimizer.update(grads, opt_state)
    delta = optax.apply_updates(delta, updates)
    # projection: enforce |delta| <= w each iteration
    delta = jnp.clip(delta, -track_half_width, track_half_width)
    return delta, opt_state


delta= delta0
best_J, best_delta = float("inf"), None
for i in range(5000):
    delta, opt_state = step(delta, opt_state)
    J = float(curvature_cost(delta))
    if J < best_J:
        best_J, best_delta = J, jnp.array(delta)
    if i % 100 == 0:
        print(f"Iter {i}, cost: {J:.3f}")
delta = best_delta 


dx= np.gradient(x_c,s)
dy = np.gradient(y_c,s)
n = np.stack([-dy,dx],axis=1)
n = n / np.linalg.norm(n,axis=1,keepdims=True)
x_opt = x_c + np.array(delta) * n[:,0]
y_opt = y_c + np.array(delta) * n[:,1]



dx = np.gradient(x_opt); dy = np.gradient(y_opt)
ddx = np.gradient(dx);   ddy = np.gradient(dy)
kappa = (dx*ddy - dy*ddx)/((dx**2 + dy**2 + 1e-8)**1.5)
print("mean|kappa|:", np.mean(np.abs(kappa)),
      "p95|kappa|:", np.percentile(np.abs(kappa), 95))

# visualize delta smoothness
plt.figure(); plt.plot(delta); plt.title("Lateral offset δ(s)"); plt.show()

plt.figure(figsize=(10,10))
plt.plot(x_c,y_c,'k--',label="Centerline")
plt.plot(x_opt,y_opt,'r-',label="Optimized path")
plt.axis('equal')
plt.legend()
plt.title("Minimum Curvature Path Optimization")
plt.show()

ds = np.sqrt(np.diff(x_opt)**2 + np.diff(y_opt)**2)
s = np.insert(np.cumsum(ds), 0, 0)

# Save full trajectory
racing_line = np.column_stack([s, x_opt, y_opt, kappa])
np.savetxt("out/racing_line_world.csv", racing_line,
           delimiter=",", header="s_m,x_m,y_m,kappa", comments="")
print("✓ Saved optimized racing line to out/racing_line_world.csv")