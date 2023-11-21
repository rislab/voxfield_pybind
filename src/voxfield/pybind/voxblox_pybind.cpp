// pybind11
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

// std stuff
#include <Eigen/Core>
#include <ios>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

// cv2
#include <opencv2/core/mat.hpp>

// Speed up Pyhton <-> C++ API
#include "stl_vector_eigen.h"

// A portion of voxfield/voxblox_ros/include/voxblox_ros/np_tsdf_server
// with ROS dependencies removed
#include "np_tsdf_utils.h"

// voxblox stuff
#include "voxblox/integrator/np_tsdf_integrator.h"
#include "voxblox/io/sdf_ply.h"
PYBIND11_MAKE_OPAQUE(std::vector<Eigen::Vector3d>);
PYBIND11_MAKE_OPAQUE(std::vector<Eigen::Vector3i>);

namespace py = pybind11;
using namespace py::literals;

namespace voxblox {

template <typename>
struct to_string {
    static char const* value();
};

#define REGISTER_INTEGRATOR_TYPE(T)               \
    template <>                                   \
    struct to_string<T> {                         \
        static char const* value() { return #T; } \
    };

REGISTER_INTEGRATOR_TYPE(NpSimpleTsdfIntegrator);
REGISTER_INTEGRATOR_TYPE(NpFastTsdfIntegrator);
REGISTER_INTEGRATOR_TYPE(NpMergedTsdfIntegrator);

}  // namespace voxblox

// TODO: Move this to a seprate util-conversions file
namespace {

auto PointcloudToVoxblox(const std::vector<Eigen::Vector3d>& points) {
    // Convert data to voxblox format
    voxblox::Pointcloud points_C(points.size());
    for (const auto& p : points) {
        points_C.emplace_back(p.cast<float>());
    }
    // Create empty colors vector
    voxblox::Colors colors(points.size());
    for (int i = 0; i < points.size(); i++) {
        colors.emplace_back(voxblox::Color{255, 255, 255});
    }

    return std::make_tuple(points_C, colors);
}

/// Extract the mest as vertices and triangles
auto ExtractMeshFromVoxbloxLayer(const voxblox::Layer<voxblox::TsdfVoxel>* layer) {
    voxblox::MeshIntegratorConfig config;
    auto mesh = std::make_unique<voxblox::Mesh>();
    voxblox::io::convertLayerToMesh(*layer, config, mesh.get());
    std::vector<Eigen::Vector3d> vertices;
    for (auto& v : mesh->vertices) {
        vertices.emplace_back(v.cast<double>());
    }

    std::vector<Eigen::Vector3i> triangles;
    for (size_t i = 0; i < mesh->indices.size(); i += 3) {
        Eigen::Vector3i triangle;
        for (int j = 0; j < 3; j++) {
            triangle[j] = mesh->indices.at(i + j);
        }
        triangles.emplace_back(triangle);
    }
    return std::make_tuple(vertices, triangles);
}
}  // namespace

namespace voxblox {

template <class IntegratorBase = NpTsdfIntegratorBase>
class PyIntegratorBase : public IntegratorBase {
public:
    using IntegratorBase::IntegratorBase;
    void integratePointCloud(const Transformation& T_G_C,
                             const Pointcloud& points_C,
                             const Pointcloud& normals_C,
                             const Colors& colors,
                             const bool freespace_points = false) override {
        PYBIND11_OVERRIDE_PURE(void, IntegratorBase, T_G_C, points_C, normals_C, colors, freespace_points);
    }
};

NpTsdfIntegratorBase::Config GetNpTsdfIntegratorConfigFromYaml(const py::dict& cfg) {
    NpTsdfIntegratorBase::Config config;

    // Only override NpTsdfIntegratorBase::Config members that are set in yaml/rosparams
    config.curve_assumption = cfg["curve_assumption"].cast<bool>();
    config.max_ray_length_m = cfg["max_ray_length_m"].cast<float>();
    config.min_ray_length_m = cfg["min_ray_length_m"].cast<float>();
    config.merge_with_clear = cfg["merge_with_clear"].cast<bool>();
    config.normal_available = cfg["normal_available"].cast<bool>();
    config.reliable_band_ratio = cfg["reliable_band_ratio"].cast<float>();
    config.reliable_normal_ratio_thre = cfg["reliable_normal_ratio_thre"].cast<float>();
    config.use_const_weight = cfg["use_const_weight"].cast<bool>();
    config.use_weight_dropoff = cfg["use_weight_dropoff"].cast<bool>();
    config.weight_reduction_exp = cfg["weight_reduction_exp"].cast<float>();

    // config.max_weight = cfg["max_weight"].cast<float>();
    // config.voxel_carving_enabled = cfg["voxel_carving_enabled"].cast<bool>();
    // config.min_ray_length_m = cfg["min_ray_length_m"].cast<FloatingPoint>();
    // config.max_ray_length_m = cfg["max_ray_length_m"].cast<FloatingPoint>();
    // config.use_const_weight = cfg["use_const_weight"].cast<bool>();
    // config.allow_clear = cfg["allow_clear"].cast<bool>();
    // config.use_weight_dropoff = cfg["use_weight_dropoff"].cast<bool>();
    // config.use_sparsity_compensation_factor = cfg["use_sparsity_compensation_factor"].cast<bool>();
    // config.sparsity_compensation_factor = cfg["sparsity_compensation_factor"].cast<float>();
    // config.integrator_threads = cfg["integrator_threads"].cast<int>();
    // config.integration_order_mode = cfg["integration_order_mode"].cast<std::string>();
    // config.enable_anti_grazing = cfg["enable_anti_grazing"].cast<bool>();
    // config.start_voxel_subsampling_factor = cfg["start_voxel_subsampling_factor"].cast<float>();
    // config.max_consecutive_ray_collisions = cfg["max_consecutive_ray_collisions"].cast<int>();
    // config.clear_checks_every_n_frames = cfg["clear_checks_every_n_frames"].cast<int>();
    return config;
}


NpTsdfServerConfig GetNpTsdfServerConfigFromYaml(const py::dict& cfg) {
    NpTsdfServerConfig config;

    if (cfg.contains("width")) config.width_ = cfg["width"].cast<int>();
    if (cfg.contains("height")) config.height_ = cfg["height"].cast<int>();
    if (cfg.contains("smooth_thre_ratio")) config.smooth_thre_ratio_ = cfg["smooth_thre_ratio"].cast<float>();
    if (cfg.contains("sensor_is_lidar")) config.sensor_is_lidar_ = cfg["sensor_is_lidar"].cast<bool>();
    if (cfg.contains("vx")) config.vx_ = cfg["vx"].cast<int>();
    if (cfg.contains("vy")) config.vy_ = cfg["vy"].cast<int>();
    if (cfg.contains("fx")) config.fx_ = cfg["fx"].cast<int>();
    if (cfg.contains("fy")) config.fy_ = cfg["fy"].cast<int>();
    if (cfg.contains("fov_up")) config.fov_up_ = cfg["fov_up"].cast<float>();
    if (cfg.contains("fov_down")) config.fov_down_ = cfg["fov_down"].cast<float>();
    if (cfg.contains("min_dist")) config.min_dist_ = cfg["min_dist"].cast<float>();
    if (cfg.contains("min_z")) config.min_z_ = cfg["min_z"].cast<float>();

    return config;
}


template <typename Integrator>
void pybind_integrator(py::module& m) {
    std::string integrator_id = std::string("_") + std::string(to_string<Integrator>::value());
    py::class_<Integrator, PyIntegratorBase<Integrator>, std::shared_ptr<Integrator>>
        python_integrator(m, integrator_id.c_str(),
                          "This is the low level C++ binding, all the methods and constructor "
                          "defined within this module (starting with a ``_`` should not be used. "
                          "Please reffer to the python Procesor class to check how to use the API");
    python_integrator
        .def(py::init([](float voxel_size, float sdf_trunc, const py::dict& cfg) {
                 auto config = GetNpTsdfIntegratorConfigFromYaml(cfg);
                 int voxels_per_side = cfg["voxels_per_side"].cast<int>();
                 config.default_truncation_distance = sdf_trunc;
                 auto* layer = new Layer<TsdfVoxel>(voxel_size, voxels_per_side);
                 return std::make_shared<Integrator>(config, layer);
             }),
             "voxel_size"_a, "sdf_trunc"_a, "config"_a)
        .def("_integrate",
             [](Integrator& self, const std::vector<Eigen::Vector3d>& points, 
             const Eigen::Matrix4f& extrinsics, const py::dict& cfg) {
                auto config = GetNpTsdfServerConfigFromYaml(cfg);
                auto [points_C, colors] = PointcloudToVoxblox(points);
                auto T_G_C = voxblox::Transformation(extrinsics);
                // Preprocess the point cloud: convert to range images
                cv::Mat vertex_map = cv::Mat::zeros(config.height_, config.width_, CV_32FC3);
                cv::Mat depth_image(vertex_map.size(), CV_32FC1, -1.0);
                cv::Mat color_image = cv::Mat::zeros(vertex_map.size(), CV_8UC3);
                projectPointCloudToImage(points_C, colors, vertex_map, depth_image, 
                    color_image, config.min_z_, config.min_dist_, config);
                cv::Mat normal_image = computeNormalImage(vertex_map, depth_image, config);
                // Back project to point cloud from range images
                points_C = extractPointCloud(vertex_map, depth_image);
                normals_C = extractNormals(normal_image, depth_image);
                colors = extractColors(color_image, depth_image);
                self.integratePointCloud(T_G_C, points_C, colors);
             })
        .def("_extract_triangle_mesh",
             [=](const Integrator& self) { return ExtractMeshFromVoxbloxLayer(self.getLayer()); });
}

PYBIND11_MODULE(voxblox_pybind, m) {
    auto vector3dvector = pybind_eigen_vector_of_vector<Eigen::Vector3d>(
        m, "_VectorEigen3d", "std::vector<Eigen::Vector3d>",
        py::py_array_to_vectors_double<Eigen::Vector3d>);

    auto vector3ivector = pybind_eigen_vector_of_vector<Eigen::Vector3i>(
        m, "_VectorEigen3i", "std::vector<Eigen::Vector3i>",
        py::py_array_to_vectors_int<Eigen::Vector3i>);

    pybind_integrator<NpSimpleTsdfIntegrator>(m);
    pybind_integrator<NpFastTsdfIntegrator>(m);
    pybind_integrator<NpMergedTsdfIntegrator>(m);
};
}  // namespace voxblox
