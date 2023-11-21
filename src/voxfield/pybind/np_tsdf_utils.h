#pragma once

// cv2
#include <opencv2/core/mat.hpp>

// voxblox
#include "voxblox/core/voxel.h"
#include "voxblox/core/tsdf_map.h"
#include "voxblox/core/esdf_map.h"
#include "voxblox/integrator/np_tsdf_integrator.h"
#include "voxblox/mesh/mesh_integrator.h"
#include "voxblox/mesh/mesh.h"
#include "voxblox/io/sdf_ply.h"

// Use this if need more includes
// #include <voxblox/alignment/icp.h>
// #include <voxblox/core/tsdf_map.h>
// #include <voxblox/integrator/np_tsdf_integrator.h>
// #include <voxblox/io/layer_io.h>
// #include <voxblox/io/mesh_ply.h>
// #include <voxblox/mesh/mesh_integrator.h>
// #include <voxblox/utils/color_maps.h>
// #include <voxblox_msgs/FilePath.h>
// #include <voxblox_msgs/Mesh.h>

// This file implements some minimal utilities of np_tsdf_server.h
// without having a dependency on ROS


class NpTsdfServerConfig {
 public:
  // Sensor specification
  int width_;
  int height_;
  // float max_range_;
  // float min_range_;
  float smooth_thre_ratio_ = 1.0f;
  bool sensor_is_lidar_ = false;

  // Camera
  int vx_;
  int vy_;
  int fx_;
  int fy_;

  // LiDAR
  float fov_up_;
  float fov_down_;

  // For preprocessing noise filter (mianly for KITTI)
  float min_dist_ = 0.1f; // 2.75 for KITTI
  float min_z_ = -1000.0f;// -3.0 for KITTI
};

// point should be in the LiDAR's coordinate system
float projectPointToImageLiDAR(
    const voxblox::Point& p_C, int* u, int* v, const NpTsdfServerConfig& config) {
  // All values are ceiled and floored to guarantee that the resulting points
  // will be valid for any integer conversion.
  float depth =
      std::sqrt(p_C.x() * p_C.x() + p_C.y() * p_C.y() + p_C.z() * p_C.z());
  float yaw = std::atan2(p_C.y(), p_C.x());
  float pitch = std::asin(p_C.z() / depth);
  // projections in image coordinates (percentage)
  float proj_x = 0.5 * (yaw / M_PI + 1.0);
  float fov = std::abs(config.fov_down_) + std::abs(config.fov_up_);
  float fov_down_rad_ = config.fov_down_ / 180.0f * M_PI;
  float fov_rad_ = fov / 180.0f * M_PI;
  float proj_y = 1.0 - (pitch - fov_down_rad_) / fov_rad_;
  // scale to image size
  proj_x *= config.width_;
  proj_y *= config.height_;
  // round for integer index
  CHECK_NOTNULL(u);
  *u = std::round(proj_x);
  if (*u == config.width_)
    *u = 0;

  CHECK_NOTNULL(v);
  *v = std::round(proj_y);
  if (std::ceil(proj_y) > config.height_ - 1 || std::floor(proj_y) < 0) {
    return (-1.0);
  }
  return depth;
}

bool projectPointToImageCamera(
    const voxblox::Point& p_C, int* u, int* v, const NpTsdfServerConfig& config) {
  CHECK_NOTNULL(u);
  *u = std::round(p_C.x() * config.fx_ / p_C.z() + config.vx_);
  if (*u >= config.width_ || *u < 0) {
    return false;
  }
  CHECK_NOTNULL(v);
  *v = std::round(p_C.y() * config.fy_ / p_C.z() + config.vy_);
  if (*v >= config.height_ || *v < 0) {
    return false;
  }
  return true;
}
bool projectPointCloudToImage(
    const voxblox::Pointcloud& points_C, const voxblox::Colors& colors,
    cv::Mat& vertex_map,   // corresponding point // NOLINT
    cv::Mat& depth_image,  // Float depth image (CV_32FC1). // NOLINT
    cv::Mat& color_image,
    float min_z, 
    float min_d, const NpTsdfServerConfig& config) {
  // TODO(py): consider to calculate in parallel to speed up
  for (size_t i = 0; i < points_C.size(); i++) {
    int u, v;
    float depth;
    if (config.sensor_is_lidar_)
      depth = projectPointToImageLiDAR(points_C[i], &u, &v, config);
    else
      depth = projectPointToImageCamera(points_C[i], &u, &v, config);
    if (depth > min_d && points_C[i].z() > min_z) {
      float old_depth = depth_image.at<float>(v, u);
      // save only nearest point for each pixel
      if (old_depth <= 0.0 || old_depth > depth) {
        for (int k = 0; k <= 2; k++) {
          vertex_map.at<cv::Vec3f>(v, u)[k] = points_C[i](k);
        }
        depth_image.at<float>(v, u) = depth;
        // BGR default order
        color_image.at<cv::Vec3b>(v, u)[0] = colors[i].b;
        color_image.at<cv::Vec3b>(v, u)[1] = colors[i].g;
        color_image.at<cv::Vec3b>(v, u)[2] = colors[i].r;
      }
    }
  }
  return false;
}

cv::Mat computeNormalImage(
    const cv::Mat& vertex_map, const cv::Mat& depth_image, const NpTsdfServerConfig& config) {
  cv::Mat normal_image(depth_image.size(), CV_32FC3, 0.0);
  for (int u = 0; u < config.width_; u++) {
    for (int v = 0; v < config.height_; v++) {
      voxblox::Point p;
      p << vertex_map.at<cv::Vec3f>(v, u)[0], vertex_map.at<cv::Vec3f>(v, u)[1],
          vertex_map.at<cv::Vec3f>(v, u)[2];

      float d_p = depth_image.at<float>(v, u);
      // sign of the normal vector
      float sign = 1.0;

      if (d_p > 0) {
        // neighbor x (in ring)
        int n_x_u;
        if (u == config.width_ - 1)
          n_x_u = 0;
        else
          n_x_u = u + 1;
        voxblox::Point n_x;
        n_x << vertex_map.at<cv::Vec3f>(v, n_x_u)[0],
            vertex_map.at<cv::Vec3f>(v, n_x_u)[1],
            vertex_map.at<cv::Vec3f>(v, n_x_u)[2];
        float d_n_x = depth_image.at<float>(v, n_x_u);
        if (d_n_x < 0)
          continue;
        // on the boundary, not continous
        if (std::abs(d_n_x - d_p) > config.smooth_thre_ratio_ * d_p)
          continue;

        // neighbor y
        int n_y_v;
        if (v == config.height_) {
          n_y_v = v - 1;
          sign *= -1.0;
        } else {
          n_y_v = v + 1;
        }
        voxblox::Point n_y;
        n_y << vertex_map.at<cv::Vec3f>(n_y_v, u)[0],
            vertex_map.at<cv::Vec3f>(n_y_v, u)[1],
            vertex_map.at<cv::Vec3f>(n_y_v, u)[2];

        float d_n_y = depth_image.at<float>(n_y_v, u);
        if (d_n_y < 0)
          continue;
        // on the boundary, not continous
        if (std::abs(d_n_y - d_p) > config.smooth_thre_ratio_ * d_p)
          continue;
        voxblox::Point dx = n_x - p;
        voxblox::Point dy = n_y - p;

        voxblox::Point normal = (dx.cross(dy)).normalized() * sign;
        cv::Vec3f& normals = normal_image.at<cv::Vec3f>(v, u);
        for (int k = 0; k <= 2; k++)
          normals[k] = normal(k);
      }
    }
  }
  return normal_image;
}

voxblox::Pointcloud extractPointCloud(
    const cv::Mat& vertex_map, const cv::Mat& depth_image) {
  voxblox::Pointcloud points_C;
  for (int v = 0; v < vertex_map.rows; v++) {
    for (int u = 0; u < vertex_map.cols; u++) {
      cv::Vec3f vertex = vertex_map.at<cv::Vec3f>(v, u);
      if (depth_image.at<float>(v, u) > 0) {
        voxblox::Point p_C(vertex[0], vertex[1], vertex[2]);
        points_C.push_back(p_C);
      }
    }
  }
  return points_C;
}

voxblox::Colors extractColors(
    const cv::Mat& color_image, const cv::Mat& depth_image) {
  voxblox::Colors colors;
  for (int v = 0; v < color_image.rows; v++) {
    for (int u = 0; u < color_image.cols; u++) {
      // BGR
      cv::Vec3b color = color_image.at<cv::Vec3b>(v, u);
      if (depth_image.at<float>(v, u) > 0) {
        // RGB
        voxblox::Color c_C(color[2], color[1], color[0]);
        colors.push_back(c_C);
      }
    }
  }
  return colors;
}

voxblox::Pointcloud extractNormals(
    const cv::Mat& normal_image, const cv::Mat& depth_image) {
  voxblox::Pointcloud normals_C;
  for (int v = 0; v < normal_image.rows; v++) {
    for (int u = 0; u < normal_image.cols; u++) {
      cv::Vec3f vertex = normal_image.at<cv::Vec3f>(v, u);
      if (depth_image.at<float>(v, u) > 0) {
        voxblox::Ray n_C(vertex[0], vertex[1], vertex[2]);
        normals_C.push_back(n_C);
      }
    }
  }
  return normals_C;
}