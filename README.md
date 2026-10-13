# AutoVision-ADAS: Real-Time Computer Vision & Camera-Radar Perception Stack

| Metadata | Details |
| :--- | :--- |
| **Document ID** | `PROP-OPENCV-ADAS-001` |
| **Implementation** | Python 3.10+ / C++17 OpenCV Stack & Interactive Jupyter Notebook |
| **Target Domain** | Automotive ADAS (SAE Level 2+ / Level 3 Perception) |
| **Complementary Tracker** | InnoSenT 77 GHz FMCW Radar Multi-Target Tracker ([specification.md](file:///c:/Users/Mourad/Desktop/MOURADCV/InnoSent/specification.md)) |
| **Performance Target** | Real-time processing at $\ge 30\,\text{FPS}$ ($\le 33\,\text{ms}$ per frame) on 1080p/720p automotive streams |

---

## 1. Project Overview & Architecture

**AutoVision-ADAS** delivers an industrial-grade perception pipeline combining high-resolution monocular computer vision with 77 GHz FMCW automotive radar. Designed to meet modern Euro NCAP 5-star active safety requirements, it extracts critical environmental intelligence from front-facing camera feeds and radar point clouds:

1. **Road Geometry & Lane Boundaries (LDW):** Dual-space adaptive thresholding (HLS + LAB + Sobel-X), Inverse Perspective Mapping (Bird's-Eye View), histogram peak sliding window search, 2nd-degree polynomial curve fitting ($x = A y^2 + B y + C$), road radius of curvature, and Lane Departure Warning (LDW).
2. **Obstacle Perception & Pseudo-3D Bounding:** 2D bounding boxes and 8-vertex pseudo-3D wireframes with class tags.
3. **Monocular Distance & Optical Flow TTC:** Pinhole camera geometry ($Z = f_y H / h$) and optical expansion rate Time-to-Collision ($\text{TTC} = \Delta t / (S_k/S_{k-1} - 1)$).
4. **Camera-Radar Extrinsic Sensor Fusion:** Exact 3D-to-2D projection ($P_{\text{cam}} = K [R \mid t] X_R$), spatial bounding box gating, sensor state fusion, and high-precision Doppler velocity integration.
5. **ADAS Safety Decision Engine:** Forward Collision Warning (FCW: $\text{TTC} < 2.6\,\text{s}$), AEB Hydraulic Pre-fill ($\text{TTC} < 1.4\,\text{s}$), Full AEB Deceleration ($\text{TTC} < 0.8\,\text{s}$), and Lane Departure Warning ($|\Delta x_{\text{offset}}| > 0.35\,\text{m}$).
6. **Augmented Reality Head-Up Display (HUD):** Dynamic lane corridor polygon, 3D wireframe boxes, live safety telemetry banner, and top-down Bird's-Eye View (BEV) radar inset.

```
+=======================================================================================+
|                     AUTONOMOUS DRIVING VISION PIPELINE (OPENCV)                       |
|                                                                                       |
|   [Automotive Camera Stream: 1080p @ 30 FPS]                                          |
|                         |                                                             |
|                         v                                                             |
|   +-------------------------------------------------------------------------------+   |
|   | 1. PRE-PROCESSING & CALIBRATION                                               |   |
|   |    cv2.undistort() -> Color Space Transform (HLS/LAB) -> Dual Channel Thresh   |   |
|   +-------------------------------------------------------------------------------+   |
|            |                                              |                           |
|            v                                              v                           |
|   +-------------------------------+              +--------------------------------+   |
|   | 2. LANE GEOMETRY & CURVATURE  |              | 3. OBSTACLE PERCEPTION (DNN)   |   |
|   |    * Inverse Perspective Map  |              |    * 2D & Pseudo-3D Bounding   |   |
|   |      (Bird's Eye View)        |              |    * Monocular Pinhole Model   |   |
|   |    * Sliding Window Search    |              |    * Optical Expansion TTC     |   |
|   |    * 2nd-Order Polyfit        |              |    * Scale Rate Estimation     |   |
|   +-------------------------------+              +--------------------------------+   |
|            |                                              |                           |
|            +----------------------+-----------------------+                           |
|                                   |                                                   |
|                                   v                                                   |
|   +-------------------------------------------------------------------------------+   |
|   | 4. CAMERA-RADAR SPATIAL FUSION (Extrinsics [R|t] & Calibration Matrix K)     |   |
|   |    * Ingest 77 GHz Radar Tracks (x, y, vx, vy) from InnoSenT EKF Tracker      |   |
|   |    * P_proj = K * [R | t] * X_radar                                           |   |
|   |    * IoU Bounding Box Association & Sensor State Fusion                       |   |
|   +-------------------------------------------------------------------------------+   |
|                                   |                                                   |
|                                   v                                                   |
|   +-------------------------------------------------------------------------------+   |
|   | 5. ADAS DECISION ENGINE & AUGMENTED REALITY HUD                               |   |
|   |    * Forward Collision Warning (FCW: TTC < 2.6s)                              |   |
|   |    * Autonomous Emergency Braking (AEB: TTC < 1.4s / 0.8s)                    |   |
|   |    * Lane Departure Warning (LDW: Offset > 0.35m)                             |   |
|   |    * Live Augmented Reality Overlay with Bird's-Eye Radar Inset               |   |
|   +===============================================================================+   |
```

---

## 2. Repository Structure

```
OPENCV/
├── OPENCV.md                    # System Specification & Proposal (PROP-OPENCV-ADAS-001)
├── autovision_adas_demo.ipynb   # Complete, fully executed pedagogical Jupyter Notebook (1.4 MB)
├── autovision_adas.py           # Modular, production-ready Python perception library
├── generate_notebook.py         # Automated notebook synthesis & headless execution script
├── pipeline_demo.py             # Standalone Python CLI & multi-scenario HUD generator
├── requirements.txt             # Python dependencies
├── include/
│   └── AdasVisionPipeline.hpp   # C++17 Core Header matching Section 4.1
├── src/
│   ├── AdasVisionPipeline.cpp   # C++17 OpenCV perception pipeline implementation
│   └── main.cpp                 # C++17 demonstration executable
├── CMakeLists.txt               # Modern CMake build system
└── output_frames/               # Generated high-resolution HUD output frames
    ├── hud_scenario1_normal.png # Scenario 1: Normal cruising (safe lane centering)
    ├── hud_scenario2_ldw.png    # Scenario 2: Lane Departure Warning (+0.45m right drift)
    └── hud_scenario3_aeb.png    # Scenario 3: Critical emergency braking (TTC = 1.07s)
```

---

## 3. Interactive Jupyter Notebook (`autovision_adas_demo.ipynb`)

The repository features a fully evaluated, self-contained **Jupyter Notebook** (`autovision_adas_demo.ipynb`) structured as an IEEE/SAE automotive perception paper:

* **Section 1: Camera Calibration & Rectification:** Pinhole model, focal length scaling, and Brown-Conrady radial/tangential undistortion.
* **Section 2: Dual Color Space Filtering:** Side-by-side decomposition across BGR, Grayscale, Sobel-X ($|\partial I / \partial x|$), HLS $S$-channel (saturation), LAB $B$-channel (yellow chrominance), and combined binary mask.
* **Section 3: Inverse Perspective Mapping (IPM):** Bird's-Eye View top-down transformation and unwarp projection.
* **Section 4: Sliding Window Search & Polynomial Fitting:** Histogram peak detection, sliding window centroid tracking, 2nd-degree polynomial curve fitting ($x = A y^2 + B y + C$), road radius of curvature ($R_{\text{curve}}$), and vehicle lateral offset.
* **Section 5: Monocular Distance & Optical Expansion TTC:** Pinhole distance formula and optical flow area expansion rate.
* **Section 6: Camera-Radar Sensor Fusion (InnoSenT Synergy):** Extrinsic projection $[R \mid t]$, spatial gating, Doppler velocity integration, and state fusion table.
* **Section 7: Augmented Reality HUD Multi-Scenario Simulation:** High-resolution HUD gallery comparing Normal Driving, Lane Departure Warning, and Emergency Collision Imminent (AEB engaged).
* **Section 8: Multi-Frame Dynamic Simulation & TTC Horizon:** 50-frame highway braking sequence with dynamic time-series curves for Range, Doppler Velocity, TTC countdown, and Warning triggers.
* **Section 9: Industrial Benchmarking & Latency Analysis:** Comprehensive latency budget table across all perception stages.

To launch the notebook:
```bash
jupyter notebook autovision_adas_demo.ipynb
```

To programmatically re-generate and execute all figures from scratch:
```bash
python generate_notebook.py
```

---

## 4. Standalone Python Demo (`pipeline_demo.py`)

Run the standalone CLI demonstration to process sample scenarios and output HUD frames:
```bash
python pipeline_demo.py
```

Output:
```
===========================================================================
 AutoVision-ADAS: Real-Time Computer Vision & Camera-Radar Perception Stack
 Specification: PROP-OPENCV-ADAS-001
===========================================================================

[INFO] Executing ADAS Perception Pipeline across test scenarios...

--- Scenario 1: Highway Cruising (Lane Centered) ---
 * Frame Resolution:       1280x720
 * Execution Latency:      28.4 ms (35.2 FPS)
 * Road Curvature:         2633.5 m
 * Vehicle Lateral Offset: +0.00 m
 * LDW Alert Active:       False (Direction: NONE)
 * Lead Vehicle Radar Dist:48.0 m
 * Doppler Rel. Velocity:  -1.0 m/s
 * Fused TTC:              48.00 s
 * Frame saved to:         output_frames\hud_scenario1_normal.png

--- Scenario 2: Lane Departure Warning (Drifting Right) ---
 * Frame Resolution:       1280x720
 * Execution Latency:      24.8 ms (40.3 FPS)
 * Road Curvature:         1230.1 m
 * Vehicle Lateral Offset: +0.45 m
 * LDW Alert Active:       True (Direction: RIGHT)
 * Lead Vehicle Radar Dist:38.0 m
 * Doppler Rel. Velocity:  -2.5 m/s
 * Fused TTC:              15.20 s
 * Frame saved to:         output_frames\hud_scenario2_ldw.png

--- Scenario 3: Forward Collision Warning & AEB Pre-fill ---
 * Frame Resolution:       1280x720
 * Execution Latency:      26.1 ms (38.3 FPS)
 * Road Curvature:         1532.7 m
 * Vehicle Lateral Offset: +0.00 m
 * LDW Alert Active:       False (Direction: NONE)
 * Lead Vehicle Radar Dist:15.0 m
 * Doppler Rel. Velocity:  -14.0 m/s
 * Fused TTC:              1.07 s
 * Frame saved to:         output_frames\hud_scenario3_aeb.png
```

---

## 5. C++17 Production Build (CMake)

The C++17 implementation provides zero-overhead execution using native OpenCV C++ data structures.

### Build Instructions:
```bash
mkdir build && cd build
cmake ..
cmake --build . --config Release
./autovision_adas_demo
```

---

## 6. Mathematical Foundations

### 6.1 Pinhole Camera Model & Extrinsic Transformation
$$\mathbf{K} = \begin{bmatrix}
f_x & 0 & c_x \\
0 & f_y & c_y \\
0 & 0 & 1
\end{bmatrix}, \quad
[\mathbf{R} \mid \mathbf{t}] = \begin{bmatrix}
1 & 0 & 0 & t_x \\
0 & 0 & -1 & t_y \\
0 & 1 & 0 & t_z
\end{bmatrix}$$

$$s \begin{bmatrix} u \\ v \\ 1 \end{bmatrix} = \mathbf{K} \cdot [\mathbf{R} \mid \mathbf{t}] \cdot \begin{bmatrix} X_R \\ Y_R \\ Z_R \\ 1 \end{bmatrix}$$

### 6.2 Road Curvature & Vehicle Lateral Offset
$$R_{\text{curve}} = \frac{\left(1 + (2 A_m y_m + B_m)^2\right)^{3/2}}{|2 A_m|}$$
$$\Delta x_{\text{offset}} = \left(\frac{W}{2} - \frac{x_{\text{left, base}} + x_{\text{right, base}}}{2}\right) \cdot s_x$$

### 6.3 Time-to-Collision (TTC) Formulation
$$\text{TTC}_{\text{fused}} = \begin{cases}
\frac{R_{\text{radar}}}{|v_{r,\text{radar}}|}, & \text{if Radar Track Confirmed } (v_r < 0) \\[1em]
\frac{Z_{\text{vision}}}{v_{\text{vision, optical}}}, & \text{if Radar Occluded / Vision Only}
\end{cases}$$

---

## 7. Synergy with InnoSenT 77 GHz FMCW Radar Multi-Target Tracker

By fusing front-facing monocular computer vision with the **InnoSenT 77 GHz FMCW Radar Multi-Target Tracker** ([specification.md](file:///c:/Users/Mourad/Desktop/MOURADCV/InnoSent/specification.md)), this system overcomes the primary weaknesses of standalone sensors:

* **Camera Alone:** High angular resolution and semantic classification, but noisy longitudinal range and susceptibility to blinding direct glare, fog, and rain.
* **Radar Alone:** Centimeter range accuracy and direct Doppler velocity unaffected by weather, but coarse angular resolution and absence of semantic classification.
* **Fused Synergy:** Direct spatial gating associates 77 GHz radar track vectors with camera 2D/3D bounding boxes, providing **centimeter-accurate range, instantaneous Doppler velocity, semantic classification, and high-confidence TTC calculations** for ASIL B / ASIL D automotive safety systems.

