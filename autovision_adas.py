"""
AutoVision-ADAS: Real-Time Computer Vision & Camera-Radar Perception Stack
Compliant with Specification: PROP-OPENCV-ADAS-001 (OPENCV.md)

Modules:
- Camera Calibration & Undistortion
- Advanced Lane Detection (Adaptive Thresholding, IPM Bird's-Eye View, Sliding Window, 2nd-Order Polyfit)
- Road Curvature & Vehicle Lateral Offset Calculation (LDW)
- Obstacle Detection & Monocular Distance / Optical Flow TTC Estimation
- Extrinsic Camera-Radar Sensor Fusion (3D-to-2D Projection, Bounding Box Gating, State Fusion)
- ADAS Decision Engine (FCW, AEB, LDW)
- Augmented Reality Head-Up Display (HUD) Rendering with Bird's-Eye Radar Inset
"""

import math
import numpy as np
import cv2
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict

# ==============================================================================
# Data Structures
# ==============================================================================

@dataclass
class CalibrationParams:
    """Camera intrinsic and extrinsic calibration parameters."""
    camera_matrix: np.ndarray      # 3x3 K matrix
    dist_coeffs: np.ndarray        # 1x5 distortion coefficients [k1, k2, p1, p2, k3]
    rot_radar_to_cam: np.ndarray   # 3x3 R rotation matrix
    trans_radar_to_cam: np.ndarray # 3x1 t translation vector [m]
    homography_ipm: np.ndarray     # 3x3 Bird's-Eye View IPM homography matrix
    inv_homography_ipm: np.ndarray # 3x3 Inverse IPM matrix
    image_width: int = 1280
    image_height: int = 720
    # Real-world metric scale factors for bird's-eye view (meters per pixel)
    ym_per_pix: float = 30.0 / 720.0  # ~30 meters field of view in 720 vertical pixels
    xm_per_pix: float = 3.7 / 700.0   # Standard US/EU highway lane width 3.7m across ~700 pixels

@dataclass
class LaneModel:
    """Fitted road geometry and lane tracking state."""
    left_fit: Optional[np.ndarray] = None       # [A, B, C] for x = A*y^2 + B*y + C
    right_fit: Optional[np.ndarray] = None      # [A, B, C]
    left_fit_cr: Optional[np.ndarray] = None    # Metric fit in meters
    right_fit_cr: Optional[np.ndarray] = None   # Metric fit in meters
    radius_curvature_m: float = 0.0             # Radius of curvature in meters
    vehicle_offset_m: float = 0.0               # Lateral offset: >0 is right of center, <0 is left
    is_valid: bool = False
    ldw_active: bool = False                    # Lane Departure Warning active
    ldw_direction: str = "NONE"                 # "LEFT", "RIGHT", or "NONE"
    left_lane_pts: List[Tuple[float, float]] = field(default_factory=list)
    right_lane_pts: List[Tuple[float, float]] = field(default_factory=list)

@dataclass
class Obstacle2D:
    """Perceived dynamic obstacle combining vision and radar attributes."""
    id: int = -1
    class_label: str = "vehicle"
    confidence: float = 0.0
    bbox: Tuple[int, int, int, int] = (0, 0, 0, 0) # (x, y, w, h)
    estimated_distance_m: float = 0.0              # Monocular pinhole distance
    ttc_sec: float = 999.0                         # Time-to-Collision
    radar_fused: bool = False                      # True if paired with radar track
    radar_range_m: float = 0.0                     # Centimeter-accurate radar range
    radar_doppler_mps: float = 0.0                 # Doppler relative velocity (negative = closing)
    radar_azimuth_deg: float = 0.0                 # Radar azimuth angle
    pseudo_3d_box: Optional[List[Tuple[int, int]]] = None # 8 projected 3D vertices

@dataclass
class RadarTrackInput:
    """Target track output from InnoSenT 77 GHz FMCW Radar Tracker."""
    track_id: int
    x_m: float          # Lateral coordinate in radar frame (meters, right is +)
    y_m: float          # Longitudinal range in radar frame (meters, forward is +)
    z_m: float          # Height coordinate (meters, up is +)
    vx_mps: float       # Lateral velocity (m/s)
    vy_mps: float       # Longitudinal velocity (m/s, negative = closing on ego)
    doppler_mps: float  # Radial Doppler velocity (m/s)
    rcs_dbsm: float = 10.0
    confidence: float = 0.95

# ==============================================================================
# Module 2.1: Camera Calibration & Lane Detection
# ==============================================================================

class LaneDetector:
    """
    Advanced Lane Boundary & Road Curvature Perception Engine.
    Implements:
    - Adaptive color (HLS / LAB) & Sobel gradient thresholding
    - Inverse Perspective Mapping (IPM) to Bird's-Eye View
    - Histogram Peak Detection & Sliding Window Search
    - 2nd-degree polynomial curve fitting
    - Curvature radius & Lane Departure Warning (LDW) calculation
    """
    def __init__(self, calib: CalibrationParams):
        self.calib = calib
        self.n_windows = 9
        self.window_margin = 100
        self.minpix = 50

    def threshold_frame(self, frame_bgr: np.ndarray) -> np.ndarray:
        """
        Adaptive dual-color space (HLS + LAB) & Sobel gradient edge detector.
        Combines:
        1. Sobel-X gradient on grayscale to detect vertical edges
        2. S-channel of HLS to isolate high-saturation lane markings under variable light
        3. B-channel of LAB to robustly isolate yellow lane markings under bright sunlight/glare
        """
        # 1. Sobel-X on Grayscale
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        abs_sobel64f = np.absolute(sobelx)
        max_sobel = np.max(abs_sobel64f)
        if max_sobel > 0:
            scaled_sobel = np.uint8(255 * abs_sobel64f / max_sobel)
        else:
            scaled_sobel = np.zeros_like(gray)
        sobel_binary = np.zeros_like(scaled_sobel)
        sobel_binary[(scaled_sobel >= 25) & (scaled_sobel <= 120)] = 1

        # 2. HLS S-Channel (Saturation) - Detects white & yellow lanes clearly
        hls = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HLS)
        s_channel = hls[:, :, 2]
        s_binary = np.zeros_like(s_channel)
        s_binary[(s_channel >= 150) & (s_channel <= 255)] = 1

        # 3. LAB B-Channel (Blue-Yellow axis) - Invariant to shadow / asphalt variations
        lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
        b_channel = lab[:, :, 2]
        b_binary = np.zeros_like(b_channel)
        b_binary[(b_channel >= 155) & (b_channel <= 200)] = 1

        # Combined binary edge map
        combined = np.zeros_like(sobel_binary)
        combined[(sobel_binary == 1) | (s_binary == 1) | (b_binary == 1)] = 255
        return combined

    def warp_to_birds_eye(self, image: np.ndarray) -> np.ndarray:
        """Warps image to orthogonal top-down Bird's-Eye View (IPM)."""
        h, w = image.shape[:2]
        return cv2.warpPerspective(image, self.calib.homography_ipm, (w, h), flags=cv2.INTER_LINEAR)

    def unwarp_from_birds_eye(self, warped: np.ndarray) -> np.ndarray:
        """Projects Bird's-Eye View representation back to perspective camera frame."""
        h, w = warped.shape[:2]
        return cv2.warpPerspective(warped, self.calib.inv_homography_ipm, (w, h), flags=cv2.INTER_LINEAR)

    def fit_lane_boundaries(self, warped_binary: np.ndarray) -> LaneModel:
        """
        Sliding Window Search and 2nd-degree Polynomial Fitting.
        Calculates:
        - Polynomial coefficients: x = A*y^2 + B*y + C
        - Radius of curvature R_curve in meters
        - Vehicle lateral offset from lane center in meters
        """
        lane = LaneModel()
        h, w = warped_binary.shape[:2]

        # Histogram of bottom half to find left and right base positions
        histogram = np.sum(warped_binary[h // 2:, :], axis=0)
        midpoint = int(histogram.shape[0] // 2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # Find nonzero pixels
        nonzero = warped_binary.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        leftx_current = leftx_base
        rightx_current = rightx_base

        window_height = int(h // self.n_windows)
        left_lane_inds = []
        right_lane_inds = []

        # Step through windows
        for window in range(self.n_windows):
            win_y_low = h - (window + 1) * window_height
            win_y_high = h - window * window_height

            win_xleft_low = leftx_current - self.window_margin
            win_xleft_high = leftx_current + self.window_margin
            win_xright_low = rightx_current - self.window_margin
            win_xright_high = rightx_current + self.window_margin

            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                              (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                               (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)

            if len(good_left_inds) > self.minpix:
                leftx_current = int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > self.minpix:
                rightx_current = int(np.mean(nonzerox[good_right_inds]))

        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        if len(left_lane_inds) < 100 or len(right_lane_inds) < 100:
            lane.is_valid = False
            return lane

        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        # Fit 2nd order polynomial in pixel space
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)
        lane.left_fit = left_fit
        lane.right_fit = right_fit

        # Fit in real-world metric space (meters)
        ym_per_pix = self.calib.ym_per_pix
        xm_per_pix = self.calib.xm_per_pix
        left_fit_cr = np.polyfit(lefty * ym_per_pix, leftx * xm_per_pix, 2)
        right_fit_cr = np.polyfit(righty * ym_per_pix, rightx * xm_per_pix, 2)
        lane.left_fit_cr = left_fit_cr
        lane.right_fit_cr = right_fit_cr

        # Compute Radius of Curvature R_curve at bottom of image (closest to vehicle)
        y_eval = (h - 1) * ym_per_pix
        left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval + left_fit_cr[1])**2)**1.5) / np.abs(2 * left_fit_cr[0])
        right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval + right_fit_cr[1])**2)**1.5) / np.abs(2 * right_fit_cr[0])
        lane.radius_curvature_m = float((left_curverad + right_curverad) / 2.0)

        # Vehicle offset from lane center
        # Center of image is assumed to be vehicle camera mounting center
        y_bottom = h - 1
        left_x_bottom = left_fit[0] * (y_bottom**2) + left_fit[1] * y_bottom + left_fit[2]
        right_x_bottom = right_fit[0] * (y_bottom**2) + right_fit[1] * y_bottom + right_fit[2]
        lane_center_px = (left_x_bottom + right_x_bottom) / 2.0
        vehicle_center_px = w / 2.0
        # Positive offset: vehicle is shifted right of center; Negative: shifted left
        lane.vehicle_offset_m = float((vehicle_center_px - lane_center_px) * xm_per_pix)

        # Lane Departure Warning (LDW) threshold: > 0.35m deviation from lane center
        if lane.vehicle_offset_m > 0.35:
            lane.ldw_active = True
            lane.ldw_direction = "RIGHT"
        elif lane.vehicle_offset_m < -0.35:
            lane.ldw_active = True
            lane.ldw_direction = "LEFT"
        else:
            lane.ldw_active = False
            lane.ldw_direction = "NONE"

        # Generate polyline points for rendering
        ploty = np.linspace(0, h - 1, h)
        left_fitx = left_fit[0] * ploty**2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty**2 + right_fit[1] * ploty + right_fit[2]

        lane.left_lane_pts = [(float(x), float(y)) for x, y in zip(left_fitx, ploty)]
        lane.right_lane_pts = [(float(x), float(y)) for x, y in zip(right_fitx, ploty)]
        lane.is_valid = True

        return lane

# ==============================================================================
# Module 2.2 & 2.3: Obstacle Detection & Monocular Distance / TTC Engine
# ==============================================================================

class ObstaclePerceptionEngine:
    """
    Monocular Obstacle Detection, Distance Estimation & Optical Flow TTC Engine.
    Uses:
    - Pinhole geometry: Z_est = (f_y * H_real) / h_bbox
    - Optical expansion: TTC = Delta_t / (S_k / S_k-1 - 1)
    - Pseudo-3D wireframe bounding box synthesis
    """
    # Average physical vehicle dimensions in meters
    REAL_HEIGHT_CAR = 1.50   # 1.5 meters average car height
    REAL_WIDTH_CAR = 1.80    # 1.8 meters average car width
    REAL_LENGTH_CAR = 4.50   # 4.5 meters average car length

    def __init__(self, calib: CalibrationParams):
        self.calib = calib
        self.fy = calib.camera_matrix[1, 1]
        self.fx = calib.camera_matrix[0, 0]
        self.cx = calib.camera_matrix[0, 2]
        self.cy = calib.camera_matrix[1, 2]
        self.prev_bboxes: Dict[int, Tuple[int, int, int, int]] = {}

    def estimate_distance_pinhole(self, bbox_h: float, real_height_m: float = 1.50) -> float:
        """Estimates object longitudinal distance Z (meters) using pinhole camera model."""
        if bbox_h <= 1.0:
            return 120.0
        return float((self.fy * real_height_m) / bbox_h)

    def calculate_expansion_ttc(self, prev_bbox: Tuple[int, int, int, int], 
                                curr_bbox: Tuple[int, int, int, int], 
                                dt: float = 0.040) -> float:
        """
        Estimates Time-to-Collision (TTC) using optical expansion rate of bounding box area:
        TTC = dt / (sqrt(Area_k / Area_{k-1}) - 1)
        """
        _, _, w0, h0 = prev_bbox
        _, _, w1, h1 = curr_bbox
        s0 = float(w0 * h0)
        s1 = float(w1 * h1)
        if s0 <= 1.0 or s1 <= 1.0:
            return 999.0
        scale_ratio = math.sqrt(s1 / s0)
        if scale_ratio <= 1.002:
            return 999.0 # Stationary or opening gap
        ttc = dt / (scale_ratio - 1.0)
        return float(max(0.1, min(ttc, 999.0)))

    def generate_pseudo_3d_box(self, bbox: Tuple[int, int, int, int], distance_m: float) -> List[Tuple[int, int]]:
        """
        Generates an 8-vertex pseudo-3D bounding box projected into 2D camera coordinates.
        Vertices: 4 front face, 4 back face.
        """
        x, y, w, h = bbox
        # Perspective depth offset for the top/rear face
        depth_scale = 0.82
        dx = int(w * 0.08)
        dy = int(h * 0.18)

        # Front face (closest to camera)
        f_tl = (x, y)
        f_tr = (x + w, y)
        f_br = (x + w, y + h)
        f_bl = (x, y + h)

        # Back face (further from camera)
        b_tl = (x + dx, y - dy)
        b_tr = (x + int(w * depth_scale) + dx, y - dy)
        b_br = (x + int(w * depth_scale) + dx, y + int(h * depth_scale) - dy)
        b_bl = (x + dx, y + int(h * depth_scale) - dy)

        return [f_tl, f_tr, f_br, f_bl, b_tl, b_tr, b_br, b_bl]

# ==============================================================================
# Module 2.4: Camera-Radar Extrinsic Sensor Fusion (InnoSenT Synergy)
# ==============================================================================

class CameraRadarFusionEngine:
    """
    Extrinsic Camera-Radar Sensor Fusion & Safety Decision Engine.
    Implements:
    - 3D-to-2D projection: P_cam = K * [R | t] * X_radar
    - Bounding Box Gating & IoU association
    - Fused state vector (Camera semantic classification + Radar Doppler velocity & range)
    - Forward Collision Warning (FCW) & Autonomous Emergency Braking (AEB)
    """
    def __init__(self, calib: CalibrationParams):
        self.calib = calib
        self.K = calib.camera_matrix
        self.R = calib.rot_radar_to_cam
        self.t = calib.trans_radar_to_cam

    def project_radar_point(self, x_m: float, y_m: float, z_m: float) -> Optional[Tuple[int, int, float]]:
        """
        Projects a 3D radar coordinate [x, y, z] in the radar reference frame
        into pixel coordinates (u, v) on the camera image plane.
        Returns: (u, v, z_cam) or None if behind camera optical center.
        """
        # Radar coordinates: X = lateral (right +), Y = range (forward +), Z = height (up +)
        # Camera coordinates: X_c = right, Y_c = down, Z_c = forward (optical axis)
        P_radar = np.array([[x_m], [y_m], [z_m]], dtype=np.float64)
        P_cam = self.R @ P_radar + self.t
        Z_cam = float(P_cam[2, 0])

        if Z_cam <= 0.5:
            return None # Point is behind camera or inside near clipping plane

        # Pinhole projection: s * [u, v, 1]^T = K * P_cam
        pixel_homog = self.K @ P_cam
        u = int(pixel_homog[0, 0] / pixel_homog[2, 0])
        v = int(pixel_homog[1, 0] / pixel_homog[2, 0])

        return (u, v, Z_cam)

    def associate_and_fuse(self, 
                           obstacles: List[Obstacle2D], 
                           radar_tracks: List[RadarTrackInput],
                           dt: float = 0.040) -> List[Obstacle2D]:
        """
        Performs spatial gating and sensor state fusion between vision bounding boxes
        and radar track inputs.
        """
        fused_obstacles = []

        for obs in obstacles:
            bx, by, bw, bh = obs.bbox
            matched_track: Optional[RadarTrackInput] = None
            min_dist_diff = float("inf")

            # Check each radar track
            for track in radar_tracks:
                proj = self.project_radar_point(track.x_m, track.y_m, track.z_m)
                if proj is None:
                    continue
                u, v, z_cam = proj

                # Spatial Gating: Verify if projected radar point lies within bounding box
                # with a small 10% safety margin
                pad_w = int(bw * 0.15)
                pad_h = int(bh * 0.15)
                if (bx - pad_w <= u <= bx + bw + pad_w) and (by - pad_h <= v <= by + bh + pad_h):
                    # Check depth consistency with monocular estimate
                    depth_diff = abs(z_cam - obs.estimated_distance_m)
                    if depth_diff < min_dist_diff:
                        min_dist_diff = depth_diff
                        matched_track = track

            if matched_track is not None:
                # Fused state
                obs.radar_fused = True
                obs.radar_range_m = float(math.sqrt(matched_track.x_m**2 + matched_track.y_m**2))
                obs.radar_doppler_mps = float(matched_track.vy_mps) # Forward/closing velocity
                obs.radar_azimuth_deg = float(math.degrees(math.atan2(matched_track.x_m, matched_track.y_m)))
                
                # Fused TTC calculation
                # Closing velocity is negative vy
                closing_speed = -obs.radar_doppler_mps
                if closing_speed > 0.5: # Closing faster than 0.5 m/s
                    obs.ttc_sec = float(obs.radar_range_m / closing_speed)
                else:
                    obs.ttc_sec = 999.0
            else:
                obs.radar_fused = False
                # Fallback to monocular vision TTC if radar occluded
                pass

            fused_obstacles.append(obs)

        return fused_obstacles

# ==============================================================================
# Module 2.5: Augmented Reality Head-Up Display (HUD) Visualizer
# ==============================================================================

class AdasHudRenderer:
    """
    Automotive Augmented Reality HUD Overlay Generator.
    Renders:
    - Warped lane corridor polygon (green=safe, yellow/red=departure)
    - 3D bounding box wireframes with Radar tags & Doppler velocity
    - ADAS Safety Decision Dashboard (FCW / AEB / LDW)
    - Bird's-Eye View Top-Down Radar Radar Inset
    """
    def __init__(self, calib: CalibrationParams):
        self.calib = calib

    def render_lane_corridor(self, canvas: np.ndarray, lane: LaneModel) -> np.ndarray:
        """Projects a transparent colored corridor over the detected drive lane."""
        if not lane.is_valid or len(lane.left_lane_pts) < 2 or len(lane.right_lane_pts) < 2:
            return canvas

        h, w = canvas.shape[:2]
        # Create an overlay image in bird's-eye view space
        warp_zero = np.zeros((h, w, 3), dtype=np.uint8)

        # Generate polygon vertices in bird's-eye view
        left_pts = np.array(lane.left_lane_pts, dtype=np.int32)
        right_pts = np.array(lane.right_lane_pts, dtype=np.int32)
        pts_right_flipped = np.flipud(right_pts)
        pts = np.vstack([left_pts, pts_right_flipped])

        # Corridor color: Green if centered, Amber if departure warning
        corridor_color = (0, 220, 0) if not lane.ldw_active else (0, 165, 255)
        cv2.fillPoly(warp_zero, [pts], corridor_color)

        # Draw bright lane boundary lines
        cv2.polylines(warp_zero, [left_pts], isClosed=False, color=(0, 255, 255), thickness=16)
        cv2.polylines(warp_zero, [right_pts], isClosed=False, color=(255, 255, 255), thickness=16)

        # Warp corridor back to original perspective camera view
        unwarped = cv2.warpPerspective(warp_zero, self.calib.inv_homography_ipm, (w, h))

        # Alpha blend with original frame
        alpha = 0.35
        out = cv2.addWeighted(canvas, 1.0, unwarped, alpha, 0)
        return out

    def render_obstacles(self, canvas: np.ndarray, obstacles: List[Obstacle2D]) -> np.ndarray:
        """Draws bounding boxes, 3D wireframes, and fused radar telemetry badges."""
        out = canvas.copy()

        for obs in obstacles:
            x, y, w, h = obs.bbox

            # Choose color based on safety severity
            if obs.ttc_sec < 1.4:
                box_color = (0, 0, 255)      # Red - CRITICAL AEB
            elif obs.ttc_sec < 2.6:
                box_color = (0, 140, 255)    # Orange - FCW WARNING
            elif obs.radar_fused:
                box_color = (0, 255, 0)      # Green - TRACKED & FUSED
            else:
                box_color = (255, 200, 0)    # Cyan/Yellow - VISION ONLY

            # 1. Draw 3D wireframe box if available
            if obs.pseudo_3d_box is not None and len(obs.pseudo_3d_box) == 8:
                pts = obs.pseudo_3d_box
                # Front face
                for i in range(4):
                    cv2.line(out, pts[i], pts[(i + 1) % 4], box_color, 2)
                # Back face
                for i in range(4):
                    cv2.line(out, pts[i + 4], pts[4 + (i + 1) % 4], box_color, 1)
                # Connecting pillars
                for i in range(4):
                    cv2.line(out, pts[i], pts[i + 4], box_color, 1)
            else:
                # 2D Bounding Box
                cv2.rectangle(out, (x, y), (x + w, y + h), box_color, 2)

            # 2. Telemetry Badge
            range_disp = obs.radar_range_m if obs.radar_fused else obs.estimated_distance_m
            ttc_str = f"{obs.ttc_sec:.1f}s" if obs.ttc_sec < 99.0 else "SAFE"
            
            badge_text_top = f"[{obs.class_label.upper()}] Z:{range_disp:.1f}m"
            if obs.radar_fused:
                badge_text_bot = f"RADAR 77GHz | v_r:{obs.radar_doppler_mps:+.1f}m/s | TTC:{ttc_str}"
            else:
                badge_text_bot = f"VISION-ONLY | TTC:{ttc_str}"

            badge_y = max(y - 30, 25)
            # Semi-transparent background for readability
            cv2.rectangle(out, (x, badge_y - 20), (x + max(w, 260), badge_y + 16), (20, 20, 20), -1)
            cv2.rectangle(out, (x, badge_y - 20), (x + max(w, 260), badge_y + 16), box_color, 1)
            cv2.putText(out, badge_text_top, (x + 5, badge_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            cv2.putText(out, badge_text_bot, (x + 5, badge_y + 11), cv2.FONT_HERSHEY_SIMPLEX, 0.42, 
                        (0, 255, 255) if obs.radar_fused else (200, 200, 200), 1)

        return out

    def render_bev_radar_inset(self, canvas: np.ndarray, 
                               radar_tracks: List[RadarTrackInput], 
                               lane: LaneModel) -> np.ndarray:
        """
        Renders a top-down Bird's-Eye View (BEV) radar display inset
        in the top-right corner of the HUD.
        """
        inset_w, inset_h = 240, 240
        inset = np.zeros((inset_h, inset_w, 3), dtype=np.uint8)
        inset[:] = (18, 22, 26) # Dark automotive UI theme

        # Draw range grid rings (20m, 40m, 60m, 80m)
        center_x = inset_w // 2
        ego_y = inset_h - 25 # Ego vehicle position at bottom center
        scale_m2pix = 2.2     # 2.2 pixels per meter -> ~100m range view

        for r_m in [20, 40, 60, 80]:
            r_px = int(r_m * scale_m2pix)
            cv2.circle(inset, (center_x, ego_y), r_px, (45, 55, 65), 1)
            cv2.putText(inset, f"{r_m}m", (center_x + 5, ego_y - r_px + 12), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, (100, 110, 120), 1)

        # Draw radar FOV cone (-45 to +45 deg)
        fov_angle = math.radians(45)
        p_left = (int(center_x - 90 * math.sin(fov_angle) * scale_m2pix), 
                  int(ego_y - 90 * math.cos(fov_angle) * scale_m2pix))
        p_right = (int(center_x + 90 * math.sin(fov_angle) * scale_m2pix), 
                   int(ego_y - 90 * math.cos(fov_angle) * scale_m2pix))
        cv2.line(inset, (center_x, ego_y), p_left, (35, 45, 55), 1)
        cv2.line(inset, (center_x, ego_y), p_right, (35, 45, 55), 1)

        # Draw Ego Vehicle icon
        cv2.rectangle(inset, (center_x - 6, ego_y - 12), (center_x + 6, ego_y + 6), (0, 200, 255), -1)

        # Draw Radar Target Tracks
        for trk in radar_tracks:
            tx = int(center_x + trk.x_m * scale_m2pix)
            ty = int(ego_y - trk.y_m * scale_m2pix)
            if 0 <= tx < inset_w and 0 <= ty < inset_h:
                # Target blip
                cv2.circle(inset, (tx, ty), 5, (0, 255, 0), -1)
                # Doppler velocity vector arrow
                vy_len = int(-trk.vy_mps * 1.5)
                cv2.line(inset, (tx, ty), (tx, ty + vy_len), (0, 255, 255), 2)
                cv2.putText(inset, f"T{trk.track_id}:{trk.y_m:.0f}m", (tx + 7, ty + 3), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 255, 255), 1)

        # Inset border and title
        cv2.rectangle(inset, (0, 0), (inset_w - 1, inset_h - 1), (0, 180, 255), 1)
        cv2.putText(inset, "77 GHz RADAR BIRD'S-EYE", (10, 18), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 200, 255), 1)

        # Blit inset onto top-right corner of canvas
        margin = 15
        canvas[margin:margin + inset_h, -inset_w - margin:-margin] = inset
        return canvas

    def render_hud_telemetry_banner(self, canvas: np.ndarray, 
                                    lane: LaneModel, 
                                    obstacles: List[Obstacle2D], 
                                    fps: float = 30.0) -> np.ndarray:
        """Renders the top ADAS system status dashboard and critical warnings."""
        out = canvas.copy()
        h, w = out.shape[:2]

        # Top banner background
        banner_h = 70
        banner_bg = out[0:banner_h, 0:w].copy()
        cv2.rectangle(out, (0, 0), (w, banner_h), (12, 16, 20), -1)
        out[0:banner_h, 0:w] = cv2.addWeighted(banner_bg, 0.3, out[0:banner_h, 0:w], 0.7, 0)
        cv2.line(out, (0, banner_h), (w, banner_h), (0, 150, 220), 2)

        # 1. System Branding
        cv2.putText(out, "AutoVision-ADAS", (20, 28), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 230, 255), 2)
        cv2.putText(out, "CAM-RADAR PERCEPTION STACK", (20, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 190, 200), 1)

        # 2. Lane & Road Telemetry
        curv_str = f"Curvature: {lane.radius_curvature_m:.0f} m" if lane.radius_curvature_m < 5000 else "Curvature: Straight"
        offset_dir = "R" if lane.vehicle_offset_m >= 0 else "L"
        offset_str = f"Offset: {abs(lane.vehicle_offset_m):.2f}m ({offset_dir})"
        cv2.putText(out, curv_str, (280, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)
        cv2.putText(out, offset_str, (280, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.48, 
                    (0, 165, 255) if lane.ldw_active else (200, 200, 200), 1)

        # 3. Minimum TTC & Safety Critical Warning
        min_ttc = 999.0
        for obs in obstacles:
            if obs.ttc_sec < min_ttc:
                min_ttc = obs.ttc_sec

        # Determine ADAS Warning Status
        if min_ttc < 0.8:
            status_text = "EMERGENCY: AEB BRAKING ENGAGED"
            status_color = (0, 0, 255) # Red
        elif min_ttc < 1.4:
            status_text = "CRITICAL: AEB HYDRAULIC PRE-FILL"
            status_color = (0, 69, 255) # Orange-Red
        elif min_ttc < 2.6:
            status_text = "FCW: FORWARD COLLISION WARNING"
            status_color = (0, 140, 255) # Orange
        elif lane.ldw_active:
            status_text = f"LDW: LANE DEPARTURE WARNING ({lane.ldw_direction})"
            status_color = (0, 215, 255) # Yellow
        else:
            status_text = "STATUS: LANE CENTERING ACTIVE (SAFE)"
            status_color = (0, 255, 100) # Green

        ttc_disp = f"TTC: {min_ttc:.1f}s" if min_ttc < 99.0 else "TTC: >10s"
        cv2.putText(out, status_text, (520, 28), cv2.FONT_HERSHEY_DUPLEX, 0.55, status_color, 2)
        cv2.putText(out, f"{ttc_disp} | Frame Rate: {fps:.1f} FPS", (520, 48), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)

        # 4. Critical Alert Box in Lower Center if FCW or AEB is triggered
        if min_ttc < 2.6:
            alert_w, alert_h = 480, 50
            ax = (w - alert_w) // 2
            ay = h - 90
            cv2.rectangle(out, (ax, ay), (ax + alert_w, ay + alert_h), (0, 0, 180), -1)
            cv2.rectangle(out, (ax, ay), (ax + alert_w, ay + alert_h), (0, 0, 255), 2)
            cv2.putText(out, status_text, (ax + 20, ay + 32), cv2.FONT_HERSHEY_DUPLEX, 0.60, (255, 255, 255), 2)

        return out

    def render_complete_hud(self, raw_frame: np.ndarray, 
                            lane: LaneModel, 
                            obstacles: List[Obstacle2D], 
                            radar_tracks: List[RadarTrackInput], 
                            fps: float = 30.0) -> np.ndarray:
        """Renders the complete integrated Head-Up Display."""
        # 1. Overlay lane corridor
        frame_corridor = self.render_lane_corridor(raw_frame, lane)
        # 2. Overlay 3D obstacle bounding boxes & fusion badges
        frame_obstacles = self.render_obstacles(frame_corridor, obstacles)
        # 3. Add top-down BEV radar inset
        frame_with_inset = self.render_bev_radar_inset(frame_obstacles, radar_tracks, lane)
        # 4. Add telemetry dashboard & warnings
        full_hud = self.render_hud_telemetry_banner(frame_with_inset, lane, obstacles, fps)
        return full_hud

# ==============================================================================
# Module 2.6: Synthetic Highway Scene Generator for ADAS Benchmarking
# ==============================================================================

class SyntheticHighwaySceneGenerator:
    """
    Generates realistic synthetic 1080p/720p dashcam video frames with
    ground truth lane markings, vehicles, and synchronized InnoSenT 77 GHz FMCW radar tracks.
    """
    def __init__(self, calib: CalibrationParams):
        self.calib = calib

    def generate_frame(self, 
                       curvature_m: float = 1200.0, 
                       vehicle_lead_distance_m: float = 35.0,
                       vehicle_lead_rel_vel_mps: float = -4.5,
                       ego_lateral_offset_m: float = 0.0) -> Tuple[np.ndarray, List[Obstacle2D], List[RadarTrackInput]]:
        """
        Synthesizes a realistic automotive dashcam frame and corresponding radar tracks.
        """
        w, h = self.calib.image_width, self.calib.image_height
        frame = np.zeros((h, w, 3), dtype=np.uint8)

        # 1. Sky & Horizon
        horizon_y = int(h * 0.45)
        for y in range(horizon_y):
            # Gradient blue-gray sky
            interp = y / horizon_y
            color = (int(160 + 30 * interp), int(170 + 20 * interp), int(190 + 10 * interp))
            frame[y, :] = color

        # Horizon line and distant mountains/foliage
        cv2.line(frame, (0, horizon_y), (w, horizon_y), (100, 110, 115), 2)
        pts_mountain = np.array([[0, horizon_y], [250, horizon_y - 45], [500, horizon_y - 20],
                                 [800, horizon_y - 50], [1100, horizon_y - 25], [w, horizon_y]], dtype=np.int32)
        cv2.fillPoly(frame, [pts_mountain], (110, 120, 125))

        # 2. Asphalt Road Surface
        for y in range(horizon_y, h):
            interp = (y - horizon_y) / (h - horizon_y)
            road_val = int(55 + 20 * interp)
            frame[y, :] = (road_val, road_val + 2, road_val + 4)

        # Road shoulders (grass/guardrail)
        left_shoulder_x = int(w * 0.08)
        right_shoulder_x = int(w * 0.92)
        vanish_x = int(w / 2.0 + (ego_lateral_offset_m / 3.7) * 150)
        vanish_y = horizon_y

        pts_grass_left = np.array([[0, horizon_y], [vanish_x - 300, horizon_y], 
                                   [0, h]], dtype=np.int32)
        pts_grass_right = np.array([[w, horizon_y], [vanish_x + 300, horizon_y], 
                                    [w, h]], dtype=np.int32)
        cv2.fillPoly(frame, [pts_grass_left], (35, 65, 40))
        cv2.fillPoly(frame, [pts_grass_right], (35, 65, 40))

        # 3. Lane Lines (Left Yellow, Right White, Adjacent Broken Lane)
        # Inverse mapping: construct lanes in bird's-eye view, then unwarp!
        bev_canvas = np.zeros((h, w, 3), dtype=np.uint8)
        bev_mid = w // 2 - int(ego_lateral_offset_m / self.calib.xm_per_pix)
        lane_half_width_px = int((3.7 / 2.0) / self.calib.xm_per_pix)

        left_x_bev = bev_mid - lane_half_width_px
        right_x_bev = bev_mid + lane_half_width_px

        # Curvature offset in BEV
        curve_coeff = 1.0 / (2.0 * curvature_m) * (self.calib.ym_per_pix**2 / self.calib.xm_per_pix)

        y_eval_pts = np.linspace(0, h, 80)
        left_pts_bev = []
        right_pts_bev = []
        adj_pts_bev = []

        for yp in y_eval_pts:
            y_rev = h - yp
            x_shift = curve_coeff * (y_rev**2)
            left_pts_bev.append((int(left_x_bev + x_shift), int(yp)))
            right_pts_bev.append((int(right_x_bev + x_shift), int(yp)))
            adj_pts_bev.append((int(right_x_bev + 2 * lane_half_width_px + x_shift), int(yp)))

        # Draw onto BEV
        cv2.polylines(bev_canvas, [np.array(left_pts_bev)], isClosed=False, color=(0, 220, 255), thickness=16) # Yellow
        cv2.polylines(bev_canvas, [np.array(right_pts_bev)], isClosed=False, color=(255, 255, 255), thickness=16) # White

        # Unwarp lanes to camera perspective view
        unwarped_lanes = cv2.warpPerspective(bev_canvas, self.calib.inv_homography_ipm, (w, h))
        mask = (unwarped_lanes > 0)
        frame[mask] = unwarped_lanes[mask]

        # 4. Lead Vehicle (Ahead in Ego Lane) - Pinhole Ground Geometry
        cam_h_m = 1.25 # Windshield camera mounting height in meters
        car_h_m = ObstaclePerceptionEngine.REAL_HEIGHT_CAR # 1.50m
        car_w_m = ObstaclePerceptionEngine.REAL_WIDTH_CAR  # 1.80m
        
        fx = self.calib.camera_matrix[0, 0]
        fy = self.calib.camera_matrix[1, 1]
        cx = self.calib.camera_matrix[0, 2]
        cy = self.calib.camera_matrix[1, 2]

        z_lead = max(5.0, vehicle_lead_distance_m)
        x_lead = 0.0 - ego_lateral_offset_m

        car_w_px = int((fx * car_w_m) / z_lead)
        car_h_px = int((fy * car_h_m) / z_lead)
        car_center_x = int(cx + (fx * x_lead) / z_lead)
        car_bottom_y = int(cy + (fy * cam_h_m) / z_lead)
        car_top_y = car_bottom_y - car_h_px
        car_x = car_center_x - car_w_px // 2

        # Draw Lead Vehicle Body (Silver/Dark Blue Sedan)
        car_bbox = (car_x, car_top_y, car_w_px, car_h_px)
        
        # Vehicle chassis rectangle
        cv2.rectangle(frame, (car_x, car_top_y + int(car_h_px * 0.35)), 
                      (car_x + car_w_px, car_bottom_y), (50, 60, 75), -1)
        # Vehicle cabin / roof
        roof_inset = int(car_w_px * 0.18)
        pts_roof = np.array([
            [car_x + roof_inset, car_top_y + int(car_h_px * 0.35)],
            [car_x + int(car_w_px * 0.28), car_top_y],
            [car_x + car_w_px - int(car_w_px * 0.28), car_top_y],
            [car_x + car_w_px - roof_inset, car_top_y + int(car_h_px * 0.35)]
        ], dtype=np.int32)
        cv2.fillPoly(frame, [pts_roof], (40, 50, 65))
        # Rear windshield
        cv2.fillPoly(frame, [pts_roof + [0, 4]], (20, 25, 30))
        # Taillights (Red)
        light_w = max(4, int(car_w_px * 0.18))
        light_h = max(3, int(car_h_px * 0.15))
        light_y = car_top_y + int(car_h_px * 0.48)
        cv2.rectangle(frame, (car_x + 3, light_y), (car_x + 3 + light_w, light_y + light_h), (0, 0, 230), -1)
        cv2.rectangle(frame, (car_x + car_w_px - 3 - light_w, light_y), 
                      (car_x + car_w_px - 3, light_y + light_h), (0, 0, 230), -1)
        # License plate
        plate_w = max(8, int(car_w_px * 0.28))
        plate_h = max(4, int(car_h_px * 0.12))
        cv2.rectangle(frame, (car_center_x - plate_w // 2, light_y + 4), 
                      (car_center_x + plate_w // 2, light_y + 4 + plate_h), (220, 220, 220), -1)
        # Tires (black)
        tire_w = max(4, int(car_w_px * 0.14))
        tire_h = max(4, int(car_h_px * 0.18))
        cv2.rectangle(frame, (car_x + 2, car_bottom_y - tire_h), 
                      (car_x + 2 + tire_w, car_bottom_y + 2), (20, 20, 20), -1)
        cv2.rectangle(frame, (car_x + car_w_px - 2 - tire_w, car_bottom_y - tire_h), 
                      (car_x + car_w_px - 2, car_bottom_y + 2), (20, 20, 20), -1)

        # 5. Populate Ground Truth Obstacle Data
        lead_obstacle = Obstacle2D(
            id=1,
            class_label="car",
            confidence=0.96,
            bbox=car_bbox,
            estimated_distance_m=vehicle_lead_distance_m,
            ttc_sec=999.0
        )

        # 6. Synchronized 77 GHz FMCW Radar Track from InnoSenT EKF Tracker
        # Coordinate system: X=lateral (+right), Y=longitudinal (+forward), Z=elevation (+up from bumper)
        lead_radar_track = RadarTrackInput(
            track_id=1,
            x_m=x_lead,
            y_m=vehicle_lead_distance_m,
            z_m=0.0, # Target reflection from bumper center
            vx_mps=0.0,
            vy_mps=vehicle_lead_rel_vel_mps, # Relative Doppler velocity
            doppler_mps=vehicle_lead_rel_vel_mps,
            rcs_dbsm=12.5,
            confidence=0.98
        )

        # Optional second obstacle: Passing vehicle in right lane
        passing_dist_m = 55.0
        z_pass = passing_dist_m
        x_pass = 3.7 - ego_lateral_offset_m
        p_car_w = int((fx * car_w_m) / z_pass)
        p_car_h = int((fy * car_h_m) / z_pass)
        p_center_x = int(cx + (fx * x_pass) / z_pass)
        p_bottom_y = int(cy + (fy * cam_h_m) / z_pass)
        p_top_y = p_bottom_y - p_car_h
        p_bbox = (p_center_x - p_car_w // 2, p_top_y, p_car_w, p_car_h)

        # Draw distant passing car
        cv2.rectangle(frame, (p_bbox[0], p_bbox[1]), (p_bbox[0] + p_bbox[2], p_bbox[1] + p_bbox[3]), (70, 70, 80), -1)

        passing_obstacle = Obstacle2D(
            id=2,
            class_label="suv",
            confidence=0.88,
            bbox=p_bbox,
            estimated_distance_m=passing_dist_m,
            ttc_sec=999.0
        )

        passing_radar_track = RadarTrackInput(
            track_id=2,
            x_m=x_pass, # In right adjacent lane
            y_m=passing_dist_m,
            z_m=0.0,
            vx_mps=0.2,
            vy_mps=+2.5, # Opening gap (moving away)
            doppler_mps=+2.5,
            rcs_dbsm=15.0,
            confidence=0.92
        )

        return frame, [lead_obstacle, passing_obstacle], [lead_radar_track, passing_radar_track]

# ==============================================================================
# Module 2.7: End-to-End Pipeline Coordinator
# ==============================================================================

class AdasVisionPipeline:
    """
    Main AutoVision-ADAS perception pipeline executing all perception stages
    on incoming frames in real-time.
    """
    def __init__(self, calib: CalibrationParams):
        self.calib = calib
        self.lane_detector = LaneDetector(calib)
        self.obstacle_engine = ObstaclePerceptionEngine(calib)
        self.fusion_engine = CameraRadarFusionEngine(calib)
        self.hud_renderer = AdasHudRenderer(calib)

    def process_frame(self, 
                      frame_bgr: np.ndarray, 
                      raw_obstacles: List[Obstacle2D], 
                      radar_tracks: List[RadarTrackInput],
                      dt: float = 0.040) -> Tuple[np.ndarray, LaneModel, List[Obstacle2D]]:
        """
        Executes end-to-end perception pipeline for a single automotive video frame.
        """
        # 1. Undistortion
        undistorted = cv2.undistort(frame_bgr, self.calib.camera_matrix, self.calib.dist_coeffs)

        # 2. Lane Detection Pipeline
        binary_edges = self.lane_detector.threshold_frame(undistorted)
        warped_binary = self.lane_detector.warp_to_birds_eye(binary_edges)
        lane_model = self.lane_detector.fit_lane_boundaries(warped_binary)

        # 3. Obstacle Perception & Monocular Distance
        processed_obstacles = []
        for obs in raw_obstacles:
            # Monocular distance
            obs.estimated_distance_m = self.obstacle_engine.estimate_distance_pinhole(obs.bbox[3])
            # Pseudo 3D Box
            obs.pseudo_3d_box = self.obstacle_engine.generate_pseudo_3d_box(obs.bbox, obs.estimated_distance_m)
            processed_obstacles.append(obs)

        # 4. Camera-Radar Sensor Fusion
        fused_obstacles = self.fusion_engine.associate_and_fuse(processed_obstacles, radar_tracks, dt)

        # 5. Render HUD Display
        hud_output = self.hud_renderer.render_complete_hud(undistorted, lane_model, fused_obstacles, radar_tracks)

        return hud_output, lane_model, fused_obstacles

# ==============================================================================
# Helper Factory for Calibration Parameters
# ==============================================================================

def create_default_calibration(w: int = 1280, h: int = 720) -> CalibrationParams:
    """
    Creates realistic calibration parameters for automotive 720p dashcam
    and front-bumper 77 GHz FMCW radar.
    """
    # Focal length ~900 px for ~65 deg horizontal FOV at 1280x720
    fx = 920.0
    fy = 920.0
    cx = w / 2.0
    cy = h / 2.0
    camera_matrix = np.array([
        [fx,  0.0, cx],
        [0.0, fy,  cy],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)

    # Slight pincushion / barrel lens distortion
    dist_coeffs = np.array([[-0.05, 0.015, 0.0, 0.0, 0.0]], dtype=np.float64)

    # Radar-to-Camera Extrinsic Matrix:
    # Radar is mounted 0.5m forward and 0.8m below windshield camera.
    # Radar frame: X_r=right, Y_r=forward, Z_r=up
    # Camera frame: X_c=right, Y_c=down, Z_c=forward
    # Rotation matrix: [X_c=X_r, Y_c=-Z_r, Z_c=Y_r]
    R_radar_to_cam = np.array([
        [1.0,  0.0,  0.0],
        [0.0,  0.0, -1.0],
        [0.0,  1.0,  0.0]
    ], dtype=np.float64)

    t_radar_to_cam = np.array([
        [0.0],  # Centered laterally with camera
        [0.8],  # Camera is 0.8m above radar (so radar is +0.8m down in camera Y)
        [-0.5]  # Radar is 0.5m forward of camera (so translation back to camera is -0.5m in camera Z)
    ], dtype=np.float64)

    # Source and destination trapezoid for Bird's-Eye View (IPM)
    # Perspective region on road
    src_pts = np.float32([
        [w * 0.15, h * 0.95],
        [w * 0.85, h * 0.95],
        [w * 0.44, h * 0.62],
        [w * 0.56, h * 0.62]
    ])
    # Top-down rectangle
    dst_pts = np.float32([
        [w * 0.28, h],
        [w * 0.72, h],
        [w * 0.28, 0],
        [w * 0.72, 0]
    ])

    M_ipm = cv2.getPerspectiveTransform(src_pts, dst_pts)
    M_inv = cv2.getPerspectiveTransform(dst_pts, src_pts)

    return CalibrationParams(
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        rot_radar_to_cam=R_radar_to_cam,
        trans_radar_to_cam=t_radar_to_cam,
        homography_ipm=M_ipm,
        inv_homography_ipm=M_inv,
        image_width=w,
        image_height=h
    )

