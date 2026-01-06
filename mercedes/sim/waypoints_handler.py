# Kohei Honda, 2024

from __future__ import annotations
from typing import Tuple
import numpy as np
import torch

# --- Make Numba optional (avoids runtime failures if versions mismatch) ---
try:
    from numba import njit  # type: ignore
    _NUMBA_AVAILABLE = True
except Exception:
    _NUMBA_AVAILABLE = False
    def njit(*args, **kwargs):  # no-op decorator
        def deco(f):
            return f
        return deco

from scipy.interpolate import splev, splrep


@njit(fastmath=False, cache=True)
def nearest_point_on_trajectory(point, trajectory):
    """
    Return the nearest point along the given piecewise linear trajectory.
    point: (2,) array
    trajectory: (N,2) array of unique (x,y) waypoints
    """
    diffs = trajectory[1:, :] - trajectory[:-1, :]
    l2s = diffs[:, 0] ** 2 + diffs[:, 1] ** 2

    # Guard repeated waypoints (zero-length segments)
    # If l2s == 0, set t = 0 for that segment.
    dots = np.empty((trajectory.shape[0] - 1,))
    for i in range(dots.shape[0]):
        dots[i] = np.dot((point - trajectory[i, :]), diffs[i, :])

    t = np.empty_like(dots)
    for i in range(t.shape[0]):
        if l2s[i] > 0.0:
            t[i] = dots[i] / l2s[i]
        else:
            t[i] = 0.0

    # Clamp to segment
    t[t < 0.0] = 0.0
    t[t > 1.0] = 1.0

    projections = trajectory[:-1, :] + (t * diffs.T).T

    dists = np.empty((projections.shape[0],))
    for i in range(dists.shape[0]):
        temp = point - projections[i]
        dists[i] = np.sqrt(np.sum(temp * temp))

    min_dist_segment = np.argmin(dists)
    return (
        projections[min_dist_segment],
        dists[min_dist_segment],
        t[min_dist_segment],
        min_dist_segment,
    )


def nearest_points_on_waypoints(
    points: torch.Tensor, waypoints: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Return:
      nearest_indices: (M,) indices of the nearest waypoint for each point
      nearest_dists:   (M,) distances to the nearest waypoint

    Args:
      points:    (M,2) [x, y]
      waypoints: (N,3) [x, y, v]  (only x,y used here)
    """
    diffs = waypoints[:, :2] - points.unsqueeze(1)   # (M,N,2)
    dists = torch.norm(diffs, dim=2)                 # (M,N)
    nearest_dists, nearest_indices = torch.min(dists, dim=1)
    return nearest_indices, nearest_dists


@njit(fastmath=False, cache=True)
def first_point_on_trajectory_intersecting_circle(
    point, radius, trajectory, t=0.0, wrap=False
):
    """
    Find the first point along the trajectory that intersects a circle at 'point' with 'radius'.
    """
    start_i = int(t)
    start_t = t % 1.0
    first_t = None
    first_i = None
    first_p = None
    trajectory = np.ascontiguousarray(trajectory)
    for i in range(start_i, trajectory.shape[0] - 1):
        start = trajectory[i, :]
        end = trajectory[i + 1, :] + 1e-6
        V = np.ascontiguousarray(end - start).astype(np.float32)

        a = np.dot(V, V)
        b = np.float32(2.0) * np.dot(V, start - point)
        c = (
            np.dot(start, start)
            + np.dot(point, point)
            - np.float32(2.0) * np.dot(start, point)
            - radius * radius
        )
        discriminant = b * b - 4 * a * c

        if discriminant < 0:
            continue
        discriminant = np.sqrt(discriminant)
        t1 = (-b - discriminant) / (2.0 * a)
        t2 = (-b + discriminant) / (2.0 * a)
        if i == start_i:
            if 0.0 <= t1 <= 1.0 and t1 >= start_t:
                first_t = t1; first_i = i; first_p = start + t1 * V; break
            if 0.0 <= t2 <= 1.0 and t2 >= start_t:
                first_t = t2; first_i = i; first_p = start + t2 * V; break
        elif 0.0 <= t1 <= 1.0:
            first_t = t1; first_i = i; first_p = start + t1 * V; break
        elif 0.0 <= t2 <= 1.0:
            first_t = t2; first_i = i; first_p = start + t2 * V; break

    if wrap and first_p is None:
        for i in range(-1, start_i):
            start = trajectory[i % trajectory.shape[0], :]
            end = trajectory[(i + 1) % trajectory.shape[0], :] + 1e-6
            V = (end - start).astype(np.float32)

            a = np.dot(V, V)
            b = np.float32(2.0) * np.dot(V, start - point)
            c = (
                np.dot(start, start)
                + np.dot(point, point)
                - np.float32(2.0) * np.dot(start, point)
                - radius * radius
            )
            discriminant = b * b - 4 * a * c
            if discriminant < 0:
                continue
            discriminant = np.sqrt(discriminant)
            t1 = (-b - discriminant) / (2.0 * a)
            t2 = (-b + discriminant) / (2.0 * a)
            if 0.0 <= t1 <= 1.0:
                first_t = t1; first_i = i; first_p = start + t1 * V; break
            elif 0.0 <= t2 <= 1.0:
                first_t = t2; first_i = i; first_p = start + t2 * V; break

    return first_p, first_i, first_t


@njit(fastmath=False, cache=True)
def get_actuation(pose_theta, lookahead_point, position, lookahead_distance, wheelbase):
    """
    Returns (speed, steering_angle).
    lookahead_point expected as [x,y,vs]; uses lookahead_point[2] as speed.
    """
    waypoint_y = np.dot(
        np.array([np.sin(-pose_theta), np.cos(-pose_theta)], dtype=np.float32),
        lookahead_point[0:2] - position,
    )
    speed = lookahead_point[2]
    if np.abs(waypoint_y) < 1e-6:
        return speed, 0.0
    radius = 1.0 / (2.0 * waypoint_y / (lookahead_distance ** 2))
    steering_angle = np.arctan(wheelbase / radius)
    return speed, steering_angle


def calculate_curvatures(points: np.ndarray) -> np.ndarray:
    """
    Calculate signed curvature along points (N,2) using cubic splines in parameter t.
    Returns (N,) curvature array.
    """
    x, y = points[:, 0], points[:, 1]
    t = np.arange(x.shape[0], dtype=float)

    spl_x = splrep(t, x)
    spl_y = splrep(t, y)

    dx  = splev(t, spl_x, der=1)
    ddx = splev(t, spl_x, der=2)
    dy  = splev(t, spl_y, der=1)
    ddy = splev(t, spl_y, der=2)

    small = 1e-6
    denom = np.maximum((dx**2 + dy**2) ** 1.5, small)
    curv = (dx * ddy - dy * ddx) / denom
    return curv
