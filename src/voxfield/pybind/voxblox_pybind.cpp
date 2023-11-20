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

// Speed up Pyhton <-> C++ API
#include "stl_vector_eigen.h"

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

template <class IntegratorBase = TsdfIntegratorBase>
class PyIntegratorBase : public IntegratorBase {
public:
    using IntegratorBase::IntegratorBase;
    void integratePointCloud(const Transformation& T_G_C,
                             const Pointcloud& points_C,
                             const Colors& colors,
                             const bool freespace_points = false) override {
        PYBIND11_OVERRIDE_PURE(void, IntegratorBase, T_G_C, points_C, colors, freespace_points);
    }
};

TsdfIntegratorBase::Config GetConfigFromYaml(const py::dict& cfg) {
    NpTsdfIntegratorBase::Config config;


    // Modify NpTsdfIntegratorBase default parameters that are set via yaml/rosparams
    if (cfg.contains("curve_assumption")) config.curve_assumption = cfg["curve_assumption"].cast<bool>();
    if (cfg.contains("max_ray_length_m")) config.max_ray_length_m = cfg["max_ray_length_m"].cast<float>();
    if (cfg.contains("min_ray_length_m")) config.min_ray_length_m = cfg["min_ray_length_m"].cast<float>();
    if (cfg.contains("merge_with_clear")) config.merge_with_clear = cfg["merge_with_clear"].cast<bool>();
    if (cfg.contains("normal_available")) config.normal_available = cfg["normal_available"].cast<bool>();
    if (cfg.contains("reliable_band_ratio")) config.reliable_band_ratio = cfg["reliable_band_ratio"].cast<float>();
    if (cfg.contains("reliable_normal_ratio_thre")) config.reliable_normal_ratio_thre = cfg["reliable_normal_ratio_thre"].cast<float>();
    if (cfg.contains("use_const_weight")) config.use_const_weight = cfg["use_const_weight"].cast<bool>();
    if (cfg.contains("use_weight_dropoff")) config.use_weight_dropoff = cfg["use_weight_dropoff"].cast<bool>();
    if (cfg.contains("weight_reduction_exp")) config.weight_reduction_exp = cfg["weight_reduction_exp"].cast<float>();

    config.max_weight = cfg["max_weight"].cast<float>();
    config.voxel_carving_enabled = cfg["voxel_carving_enabled"].cast<bool>();
    config.min_ray_length_m = cfg["min_ray_length_m"].cast<FloatingPoint>();
    config.max_ray_length_m = cfg["max_ray_length_m"].cast<FloatingPoint>();
    config.use_const_weight = cfg["use_const_weight"].cast<bool>();
    config.allow_clear = cfg["allow_clear"].cast<bool>();
    config.use_weight_dropoff = cfg["use_weight_dropoff"].cast<bool>();
    config.use_sparsity_compensation_factor = cfg["use_sparsity_compensation_factor"].cast<bool>();
    config.sparsity_compensation_factor = cfg["sparsity_compensation_factor"].cast<float>();
    config.integrator_threads = cfg["integrator_threads"].cast<int>();
    config.integration_order_mode = cfg["integration_order_mode"].cast<std::string>();
    config.enable_anti_grazing = cfg["enable_anti_grazing"].cast<bool>();
    config.start_voxel_subsampling_factor = cfg["start_voxel_subsampling_factor"].cast<float>();
    config.max_consecutive_ray_collisions = cfg["max_consecutive_ray_collisions"].cast<int>();
    config.clear_checks_every_n_frames = cfg["clear_checks_every_n_frames"].cast<int>();
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
                 auto config = GetConfigFromYaml(cfg);
                 int voxels_per_side = cfg["voxels_per_side"].cast<int>();
                 config.default_truncation_distance = sdf_trunc;
                 auto* layer = new Layer<TsdfVoxel>(voxel_size, voxels_per_side);
                 return std::make_shared<Integrator>(config, layer);
             }),
             "voxel_size"_a, "sdf_trunc"_a, "config"_a)
        .def("_integrate",
             [](Integrator& self, const std::vector<Eigen::Vector3d>& points,
                const Eigen::Matrix4f& extrinsics) {
                 auto [points_C, colors] = PointcloudToVoxblox(points);
                 auto T_G_C = voxblox::Transformation(extrinsics);
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
