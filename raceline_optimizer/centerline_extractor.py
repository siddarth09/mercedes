#!/usr/bin/env python3
import argparse
import os
import math
import yaml
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev
from scipy.spatial import cKDTree

# ------------- Small utils ------------- #
def ensure_dir(p):
    os.makedirs(p, exist_ok=True); return p

def arc_length_param(points):
    d = np.sqrt(((np.roll(points, -1, axis=0) - points)**2).sum(axis=1))
    s = np.cumsum(d)
    s = np.insert(s[:-1], 0, 0.0)
    return s

def resample_polyline(points, n_out):
    """Resample closed polyline to n_out points equally spaced by arc length."""
    s = arc_length_param(points)
    total = s[-1] + np.linalg.norm(points[0] - points[-1])
    ss = np.linspace(0, total, n_out, endpoint=False)
    # duplicate one point for wrap-around interpolation
    pts = np.vstack([points, points[0]])
    s_wrap = np.append(s, total)
    x = np.interp(ss, s_wrap, pts[:,0])
    y = np.interp(ss, s_wrap, pts[:,1])
    return np.column_stack([x, y])

def load_yaml(yaml_path):
    if yaml_path and os.path.exists(yaml_path):
        with open(yaml_path, "r") as f:
            meta = yaml.safe_load(f)
        res = float(meta.get("resolution", 0.05))
        origin = meta.get("origin", [0.0, 0.0, 0.0])
        return res, origin
    # sensible defaults if map.yaml not provided
    return 0.05, [0.0, 0.0, 0.0]

def world_from_pixels(px_xy, resolution, origin_xytheta, img_height):
    """
    Convert image pixels (j = x pixel, i = y pixel) to ROS map frame coordinates.
    - YAML 'origin' is the pose of the lower-left corner of the image in the map frame.
    - Image coords start at top-left, so we must flip Y by (H - i).
    - Apply yaw rotation from the YAML.
    """
    origin_x, origin_y, origin_yaw = origin_xytheta
    j = px_xy[:, 0].astype(np.float64)  # column index
    i = px_xy[:, 1].astype(np.float64)  # row index

    # Image -> local metric coords before rotation/translation
    x_local = j * resolution
    y_local = (img_height - i) * resolution  # flip Y

    c = np.cos(origin_yaw)
    s = np.sin(origin_yaw)

    # Rotate, then translate
    x_world = origin_x + c * x_local - s * y_local
    y_world = origin_y + s * x_local + c * y_local

    return np.column_stack([x_world, y_world])

def remove_border_touching(contours, img_w, img_h, margin=1):
    out = []
    for c in contours:
        xs = c[:,0,0]; ys = c[:,0,1]
        if (xs <= margin).any() or (ys <= margin).any() or (xs >= img_w-1-margin).any() or (ys >= img_h-1-margin).any():
            continue
        out.append(c)
    return out

# ------------- Core pipeline ------------- #
def extract_centerline(pgm_path, yaml_path=None, out_dir="store/centerline/",
                       samples_boundary=1500, samples_center=2000, smooth_s=300.0, show=False):

    ensure_dir(out_dir)
    # Load image
    img = cv2.imread(pgm_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {pgm_path}")
    H, W = img.shape[:2]

    _, obstacles = cv2.threshold(img, 25, 255, cv2.THRESH_BINARY_INV)
    border = 15
    # --- Add 10-pixel padding to avoid border-touching walls ---
    obstacles = cv2.copyMakeBorder(obstacles, border, border, border, border, cv2.BORDER_CONSTANT, value=0)

    # --- Morphological closing to seal small gaps ---
    kernel = np.ones((7, 7), np.uint8)
    obstacles = cv2.morphologyEx(obstacles, cv2.MORPH_CLOSE, kernel, iterations=3)

    # --- Optional dilation to thicken walls slightly ---
    obstacles = cv2.dilate(obstacles, kernel, iterations=1)

    cv2.imwrite(os.path.join(out_dir, "debug_padded.png"), obstacles)
    print(f"[debug] saved padded/closed obstacle map to {out_dir}/debug_padded.png")


    # --- Robust contour extraction using Canny edges ---
    edges = cv2.Canny(obstacles, 50, 150)
    
    obstacles_inverted = cv2.bitwise_not(obstacles)

    contours, hierarchy = cv2.findContours(obstacles_inverted, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    if hierarchy is None or len(contours) == 0:
        raise RuntimeError("No contours found in filled obstacle mask. Check thresholding/morphology.")

    # hierarchy shape: (1, N, 4) -> [Next, Prev, First_Child, Parent]
    hierarchy = hierarchy[0]

    # All external contours (no parent)
    outer_candidates = [i for i, h in enumerate(hierarchy) if h[3] == -1]

    # Sort them by area, largest first
    outer_candidates_sorted = sorted(outer_candidates, key=lambda i: cv2.contourArea(contours[i], oriented=False), reverse=True)

    # Skip the very largest (background), take the 2nd largest as the true track perimeter
    if len(outer_candidates_sorted) >= 2:
        outer_idx = outer_candidates_sorted[1]
    else:
        outer_idx = outer_candidates_sorted[0]

    outer = contours[outer_idx]

    # 2) Among its children (holes), choose the largest as the INNER wall
    child_idxs = []
    child = hierarchy[outer_idx][2]
    while child != -1:
        child_idxs.append(child)
        child = hierarchy[child][0]  # iterate siblings via 'Next'

    if not child_idxs:
        raise RuntimeError("Outer contour has no holes; increase closing/dilation to make inner loop solid.")

    inner_idx = max(child_idxs, key=lambda i: cv2.contourArea(contours[i], oriented=False))
    inner = contours[inner_idx]

    # 3) Resample both walls to consistent density
    A_raw = inner[:, 0, :].astype(np.float32)   # inner wall (closer to infield)
    B_raw = outer[:, 0, :].astype(np.float32)   # outer wall (track exterior)

    A = resample_polyline(A_raw, samples_boundary)
    B = resample_polyline(B_raw, samples_boundary)

    # 4) Pair points by nearest neighbor from inner→outer and midpoint
    from scipy.spatial import cKDTree
    treeB = cKDTree(B)
    _, nn_idx = treeB.query(A, k=1)
    mid = 0.5 * (A + B[nn_idx])

    # 5) Smooth centerline (periodic)
    try:
        (tck, u) = splprep([mid[:,0], mid[:,1]], s=smooth_s, per=True)
        u_f = np.linspace(0, 1, samples_center, endpoint=False)
        xs, ys = splev(u_f, tck)
        center_smooth = np.column_stack([xs, ys]).astype(np.float32)
    except Exception as e:
        print(f"[warn] spline smoothing failed ({e}); using raw midpoints.")
        center_smooth = resample_polyline(mid, samples_center)

    A[:, 0] -= border
    A[:, 1] -= border
    B[:, 0] -= border
    B[:, 1] -= border
    center_smooth[:, 0] -= border
    center_smooth[:, 1] -= border

    # --- Debug overlay ---
    dbg = cv2.cvtColor(obstacles, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(dbg, [outer], -1, (255, 0, 0), 2)   # blue outer
    cv2.drawContours(dbg, [inner], -1, (0, 165, 255), 2) # orange inner (BGR)
    for p in center_smooth[:: max(1, len(center_smooth)//200) ]:
        cv2.circle(dbg, (int(p[0]), int(p[1])), 1, (0,255,0), -1)  # green dots
    cv2.imwrite(os.path.join(out_dir, "debug_hierarchy_contours.png"), dbg)
    print(f"[debug] saved hierarchy contours to {os.path.join(out_dir,'debug_hierarchy_contours.png')}")


    # Save pixel-space CSV
    center_px_csv = os.path.join(out_dir, "centerline_pixels.csv")
    np.savetxt(center_px_csv, center_smooth, delimiter=",", header="x_px,y_px", comments="")
    print(f"✓ Saved pixel centerline: {center_px_csv}")

    # Try to convert to world coords if YAML available
    
    res, origin = load_yaml(yaml_path)  # origin = [x, y, yaw]
    H, W = img.shape[:2]

    center_world = world_from_pixels(center_smooth, res, origin, H)
   
    center_w_csv = os.path.join(out_dir, "centerline_world.csv")
    np.savetxt(center_w_csv, center_world, delimiter=",", header="x_m,y_m", comments="")
    print(f"✓ Saved world centerline: {center_w_csv} (resolution={res}, origin={origin[:2]})")

  
    fig = plt.figure(figsize=(8,8))
    plt.imshow(img, cmap="gray")
    plt.plot(A[:,0], A[:,1], '-', lw=1, label='Boundary A')
    plt.plot(B[:,0], B[:,1], '-', lw=1, label='Boundary B')
    plt.plot(center_smooth[:,0], center_smooth[:,1], '-', lw=2, label='Centerline')
    plt.gca().invert_yaxis()  
    plt.legend()
    plt.title("Centerline Extraction")
    png_path = os.path.join(out_dir, "centerline_preview.png")
    plt.savefig(png_path, dpi=200, bbox_inches='tight')
    if show:
        plt.show()
    plt.close(fig)
    print(f"✓ Preview saved: {png_path}")

    return center_smooth, center_world

# ------------- CLI ------------- #
def main():
    ap = argparse.ArgumentParser(description="Extract a smoothed, evenly-spaced centerline from an occupancy grid (.pgm).")
    ap.add_argument("--map", required=True, help="Path to .pgm occupancy grid")
    ap.add_argument("--yaml", default=None, help="Optional path to map .yaml (for resolution/origin)")
    ap.add_argument("--out", required=True, help="Output directory")
    ap.add_argument("--samples_boundary", type=int, default=1500, help="Samples per boundary after resampling")
    ap.add_argument("--samples_center", type=int, default=2000, help="Samples for final centerline")
    ap.add_argument("--smooth_s", type=float, default=300.0, help="Spline smoothing parameter (higher = smoother)")
    ap.add_argument("--show", action="store_true", help="Show matplotlib windows")
    args = ap.parse_args()

    extract_centerline(
        pgm_path=args.map,
        yaml_path=args.yaml,
        out_dir=ensure_dir(args.out),
        samples_boundary=args.samples_boundary,
        samples_center=args.samples_center,
        smooth_s=args.smooth_s,
        show=args.show
    )

if __name__ == "__main__":
    main()
