#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <vector>
#include <string>
#include <memory>

/**
 * @file AdasVisionPipeline.hpp
 * @brief Production C++17 Real-Time Computer Vision & Camera-Radar Perception Stack
 * @details Compliant with Specification PROP-OPENCV-ADAS-001 (OPENCV.md)
 */

namespace adas::vision {

struct LaneModel {
    std::vector<cv::Point2f> left_lane_pts;
    std::vector<cv::Point2f> right_lane_pts;
    float radius_curvature_m{0.0f};
    float vehicle_offset_m{0.0f}; // Positive: right of center, Negative: left
    bool is_valid{false};
    bool ldw_active{false};
    std::string ldw_direction{"NONE"};
};

struct Obstacle2D {
    int id{-1};
    std::string class_label{"vehicle"};
    float confidence{0.0f};
    cv::Rect2d bbox{0, 0, 0, 0};
    float estimated_distance_m{0.0f};
    float ttc_sec{999.0f};
    bool radar_fused{false};
    float radar_doppler_mps{0.0f};
    float radar_range_m{0.0f};
};

struct CalibrationParams {
    cv::Mat camera_matrix;      // 3x3 K
    cv::Mat dist_coeffs;        // 1x5 [k1, k2, p1, p2, k3]
    cv::Mat rot_radar_to_cam;   // 3x3 R
    cv::Mat trans_radar_to_cam; // 3x1 t
    cv::Mat homography_ipm;     // 3x3 IPM Homography
    cv::Mat inv_homography_ipm; // 3x3 Inverse IPM Homography
    int image_width{1280};
    int image_height{720};
    float ym_per_pix{30.0f / 720.0f};
    float xm_per_pix{3.7f / 700.0f};
};

class AdasVisionPipeline {
public:
    AdasVisionPipeline();
    ~AdasVisionPipeline();

    bool initialize(const std::string& model_onnx_path, 
                    const CalibrationParams& calib);

    // Process a single video frame
    void processFrame(const cv::Mat& input_bgr,
                      LaneModel& out_lane,
                      std::vector<Obstacle2D>& out_obstacles);

    // Fuse external radar tracks with detected vision obstacles
    void fuseRadarTracks(const std::vector<cv::Point3f>& radar_positions_m,
                         const std::vector<float>& radar_velocities_mps,
                         std::vector<Obstacle2D>& obstacles);

    // Render augmented reality HUD overlay
    cv::Mat renderHUD(const cv::Mat& raw_frame,
                      const LaneModel& lane,
                      const std::vector<Obstacle2D>& obstacles,
                      const std::vector<cv::Point3f>& radar_positions_m = {},
                      const std::vector<float>& radar_velocities_mps = {});

    // Utility to project 3D radar point to 2D image coordinates (u, v)
    bool projectRadarPoint(const cv::Point3f& radar_pt_m, cv::Point2i& out_pixel, float& out_z_cam) const;

private:
    cv::dnn::Net net_;
    CalibrationParams calib_;
    cv::Mat bird_view_cache_;
    bool has_model_{false};

    void extractLaneEdges(const cv::Mat& undistorted, cv::Mat& out_binary);
    void fitLanePolynomials(const cv::Mat& warped_binary, LaneModel& out_lane);
    void detectObjects(const cv::Mat& undistorted, std::vector<Obstacle2D>& out_obstacles);
};

// Factory function for default automotive calibration
CalibrationParams createDefaultCalibration(int width = 1280, int height = 720);

} // namespace adas::vision

