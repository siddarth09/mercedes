#!/usr/bin/env python3

import cv2
import yaml
import numpy as np
import csv
from skimage.morphology import skeletonize
from scipy.spatial import cKDTree

def extract_waypoints(image_path, yaml_path,
                      blue_lower=(100,0,0), blue_upper=(255,100,100)):
    # 1) Load YAML
    with open(yaml_path, 'r') as f:
        cfg = yaml.safe_load(f)
    res    = float(cfg['resolution'])
    origin = np.array(cfg['origin'][:2], dtype=float)

    # 2) Load image (BGR)
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image '{image_path}'")
    H, W = img.shape[:2]

    # 3) Threshold blue band
    lowerb = (int(blue_lower[0]), int(blue_lower[1]), int(blue_lower[2]))
    upperb = (int(blue_upper[0]), int(blue_upper[1]), int(blue_upper[2]))
    mask   = cv2.inRange(img, lowerb, upperb)

    # 3a) Clean small noise via morphological open
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # 4) Skeletonize
    skel = skeletonize(mask > 0).astype(np.uint8)
    ys, xs = np.nonzero(skel)
    pts = np.vstack([xs, ys]).T

    # 5) Build KD-tree
    tree = cKDTree(pts)

    # 6) Find an endpoint
    endpoints = []
    for i, p in enumerate(pts):
        if len(tree.query_ball_point(p, np.sqrt(2)+1e-3)) - 1 == 1:
            endpoints.append(i)
    start = endpoints[0] if endpoints else 0

    # 7) Walk the skeleton
    visited = np.zeros(len(pts), bool)
    order   = []
    curr    = start
    for _ in range(len(pts)):
        visited[curr] = True
        order.append(curr)
        dists, idxs = tree.query(pts[curr], k=8)
        next_idx = next((idx for dist,idx in zip(dists[1:], idxs[1:])
                         if not visited[idx]), None)
        if next_idx is None:
            break
        curr = next_idx

    # 8) Convert to map coords
    waypoints = []
    for x_pix, y_pix in pts[order]:
        mx = origin[0] + x_pix * res
        my = origin[1] + (H - y_pix - 1) * res
        waypoints.append((mx, my))

    return waypoints

def overlay_waypoints(image_path, yaml_path, waypoints, out_image_path):
    # 1) Load map + metadata
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image '{image_path}'")
    H, W = img.shape[:2]
    with open(yaml_path, 'r') as f:
        cfg = yaml.safe_load(f)
    res    = float(cfg['resolution'])
    origin = cfg['origin'][:2]

    # 2) Map → pixel conversion
    pix_pts = []
    for x_map, y_map in waypoints:
        col = int(round((x_map - origin[0]) / res))
        row = int(round(H - 1 - (y_map - origin[1]) / res))
        pix_pts.append((col, row))

    # 3) Draw red polyline + dots
    for i in range(len(pix_pts) - 1):
        cv2.line(img, pix_pts[i], pix_pts[i+1], (0,0,255), 2)
    for p in pix_pts:
        cv2.circle(img, p, 3, (0,0,255), -1)

    # 4) Save annotated image
    cv2.imwrite(out_image_path, img)

if __name__ == '__main__':
    # ─── Hard-coded paths ─────────────────────────────
    IMAGE_PATH      = '/home/deepak/Data/f1tenth/mercedes_ws/src/mercedes/maps/levine_updated.png'
    YAML_PATH       = '/home/deepak/Data/f1tenth/mercedes_ws/src/mercedes/maps/levine.yaml'
    CSV_OUT_PATH    = '/home/deepak/Data/f1tenth/mercedes_ws/src/mercedes/storage/waypoints.csv'
    OVERLAY_OUT_PATH= 'overlay.png'
    # ────────────────────────────────────────────────────

    # 1) extract & save CSV
    wps = extract_waypoints(IMAGE_PATH, YAML_PATH)
    with open(CSV_OUT_PATH, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['x', 'y'])
        writer.writerows(wps)
    print(f"Saved {len(wps)} waypoints → {CSV_OUT_PATH}")

    # 2) overlay & save image
    overlay_waypoints(IMAGE_PATH, YAML_PATH, wps, OVERLAY_OUT_PATH)
    print(f"Overlay image written → {OVERLAY_OUT_PATH}")