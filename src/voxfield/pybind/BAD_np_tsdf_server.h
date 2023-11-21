// projectPointCloudToImage
// computeNormalImage
// extractPointCloud
// extractNormals
// extractColors
// integratePointCloud

#include <memory>
#include <queue>
#include <string>

#include <opencv2/core/mat.hpp>
#include <pcl/conversions.h>
#include <pcl/filters/filter.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl_ros/point_cloud.h>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <std_srvs/Empty.h>
#include <tf/transform_broadcaster.h>
#include <visualization_msgs/MarkerArray.h>

#include <voxblox/alignment/icp.h>
#include <voxblox/core/tsdf_map.h>
#include <voxblox/integrator/np_tsdf_integrator.h>
#include <voxblox/io/layer_io.h>
#include <voxblox/io/mesh_ply.h>
#include <voxblox/mesh/mesh_integrator.h>
#include <voxblox/utils/color_maps.h>
#include <voxblox_msgs/FilePath.h>
#include <voxblox_msgs/Mesh.h>

#include "voxblox_ros/mesh_vis.h"
#include "voxblox_ros/ptcloud_vis.h"
#include "voxblox_ros/transformer.h"

constexpr float kDefaultMaxIntensity = 100.0;

class NpTsdfServerConfig {
  // Sensor specification
  int width_;
  int height_;
  float max_range_;
  float min_range_;
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
  float fov_down_rad_;
  float fov_rad_;

  // For preprocessing noise filter (mianly for KITTI)
  float min_dist_ = 0.1f; // 2.75 for KITTI
  float min_z_ = -1000.0f;// -3.0 for KITTI
};


  void voxblox::getServerConfigFromRosParam(const ros::NodeHandle& nh_private);


  bool projectPointCloudToImage(
      const Pointcloud& points_C, const Colors& colors,
      cv::Mat& vertex_map,
      cv::Mat& depth_image,
      cv::Mat& color_image,
      float min_z,
      float min_d) const;
  float projectPointToImageLiDAR(const Point& p_C, int* u, int* v) const;
  bool projectPointToImageCamera(const Point& p_C, int* u, int* v) const;
  cv::Mat computeNormalImage(
      const cv::Mat& vertex_map, const cv::Mat& depth_image) const;
  // from range image to point cloud
  voxblox::Pointcloud extractPointCloud(
      const cv::Mat& vertex_map,
      const cv::Mat& depth_image) const;
  voxblox::Pointcloud extractNormals(
      const cv::Mat& normal_image,
      const cv::Mat& depth_image) const;
  voxblox::Colors extractColors(
      const cv::Mat& color_image,
      const cv::Mat& depth_image) const;

 protected:


};