"""
AutoVision-ADAS: Standalone Pipeline Demonstration
Compliant with Specification: PROP-OPENCV-ADAS-001 (OPENCV.md) Section 4.2

This script runs the complete perception pipeline across multiple highway frames,
prints real-time telemetry metrics, and saves the resulting Head-Up Display (HUD) frames.
"""

import os
import time
import cv2
import numpy as np
import autovision_adas as adas

def run_demo():
    print("=" * 75)
    print(" AutoVision-ADAS: Real-Time Computer Vision & Camera-Radar Perception Stack")
    print(" Specification: PROP-OPENCV-ADAS-001")
    print("=" * 75)

    # 1. Initialize Calibration
    calib = adas.create_default_calibration(w=1280, h=720)
    pipeline = adas.AdasVisionPipeline(calib)
    generator = adas.SyntheticHighwaySceneGenerator(calib)

    output_dir = "output_frames"
    os.makedirs(output_dir, exist_ok=True)

    # Test Scenarios
    scenarios = [
        {
            "name": "Scenario 1: Highway Cruising (Lane Centered)",
            "curvature": 2500.0,
            "dist": 48.0,
            "rel_vel": -1.0,
            "offset": 0.0,
            "filename": "hud_scenario1_normal.png"
        },
        {
            "name": "Scenario 2: Lane Departure Warning (Drifting Right)",
            "curvature": 1200.0,
            "dist": 38.0,
            "rel_vel": -2.5,
            "offset": 0.45,
            "filename": "hud_scenario2_ldw.png"
        },
        {
            "name": "Scenario 3: Forward Collision Warning & AEB Pre-fill",
            "curvature": 1500.0,
            "dist": 15.0,
            "rel_vel": -14.0, # Rapid deceleration
            "offset": 0.0,
            "filename": "hud_scenario3_aeb.png"
        }
    ]

    print("\n[INFO] Executing ADAS Perception Pipeline across test scenarios...")

    for i, sc in enumerate(scenarios, 1):
        t_start = time.perf_counter()
        
        # Synthesize frame
        raw_frame, raw_obstacles, radar_tracks = generator.generate_frame(
            curvature_m=sc["curvature"],
            vehicle_lead_distance_m=sc["dist"],
            vehicle_lead_rel_vel_mps=sc["rel_vel"],
            ego_lateral_offset_m=sc["offset"]
        )

        # Run pipeline
        hud_frame, lane, fused_obs = pipeline.process_frame(raw_frame, raw_obstacles, radar_tracks)
        
        t_elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        fps = 1000.0 / max(0.1, t_elapsed_ms)

        # Save HUD image
        out_path = os.path.join(output_dir, sc["filename"])
        cv2.imwrite(out_path, hud_frame)

        # Telemetry Summary
        lead_obs = fused_obs[0]
        print(f"\n--- {sc['name']} ---")
        print(f" * Frame Resolution:       {calib.image_width}x{calib.image_height}")
        print(f" * Execution Latency:      {t_elapsed_ms:.2f} ms ({fps:.1f} FPS)")
        print(f" * Road Curvature:         {lane.radius_curvature_m:.1f} m")
        print(f" * Vehicle Lateral Offset: {lane.vehicle_offset_m:+.2f} m")
        print(f" * LDW Alert Active:       {lane.ldw_active} (Direction: {lane.ldw_direction})")
        print(f" * Lead Vehicle Radar Dist:{lead_obs.radar_range_m:.1f} m")
        print(f" * Doppler Rel. Velocity:  {lead_obs.radar_doppler_mps:+.1f} m/s")
        print(f" * Fused TTC:              {lead_obs.ttc_sec:.2f} s")
        print(f" * Frame saved to:         {out_path}")

    print("\n" + "=" * 75)
    print(" [SUCCESS] All perception scenarios processed successfully.")
    print("=" * 75)

if __name__ == "__main__":
    run_demo()

