#include "np_tsdf_server.h"

#include <minkindr_conversions/kindr_msg.h>
#include <minkindr_conversions/kindr_tf.h>

#include "voxblox_ros/conversions.h"
#include "ros_params.h"

// projectPointCloudToImage
// computeNormalImage
// extractPointCloud
// extractNormals
// extractColors
// integratePointCloud


NpTsdfServer::NpTsdfServer(
    const ros::NodeHandle& nh, const ros::NodeHandle& nh_private)
    : NpTsdfServer(
          nh, nh_private, getTsdfMapConfigFromRosParam(nh_private),
          getNpTsdfIntegratorConfigFromRosParam(nh_private),
          getMeshIntegratorConfigFromRosParam(nh_private)) {}

NpTsdfServer::NpTsdfServer(
    const ros::NodeHandle& nh, const ros::NodeHandle& nh_private,
    const TsdfMap::Config& config,
    const NpTsdfIntegratorBase::Config& integrator_config,  // NOLINT
    const MeshIntegratorConfig& mesh_config)
    : nh_(nh),
      nh_private_(nh_private),
      verbose_(true),
      world_frame_("world"),
      icp_corrected_frame_("icp_corrected"),
      pose_corrected_frame_("pose_corrected"),
      max_block_distance_from_body_(std::numeric_limits<FloatingPoint>::max()),
      slice_level_(0.5),
      use_freespace_pointcloud_(false),
      color_map_(new RainbowColorMap()),
      publish_pointclouds_on_update_(false),
      publish_slices_(false),
      publish_pointclouds_(false),
      publish_tsdf_map_(false),
      cache_mesh_(false),
      enable_icp_(false),
      accumulate_icp_corrections_(true),
      pointcloud_queue_size_(1),
      num_subscribers_tsdf_map_(0),
      transformer_(nh, nh_private) {
  getServerConfigFromRosParam(nh_private);

bool NpTsdfServer::projectPointCloudToImage(
    const Pointcloud& points_C, const Colors& colors,
    cv::Mat& vertex_map,   // corresponding point // NOLINT
    cv::Mat& depth_image,  // Float depth image (CV_32FC1). // NOLINT
    cv::Mat& color_image,
    float min_z, 
    float min_d) const {
  // TODO(py): consider to calculate in parallel to speed up
  for (size_t i = 0; i < points_C.size(); i++) {
    int u, v;
    float depth;
    if (sensor_is_lidar_)
      depth = projectPointToImageLiDAR(points_C[i], &u, &v);
    else
      depth = projectPointToImageCamera(points_C[i], &u, &v);
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