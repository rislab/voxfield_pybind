from dataclasses import asdict, dataclass


# @dataclass
# class VoxBloxConfig:
#     # SimpleTsdfIntegrator
#     voxels_per_side: int = 16
#     max_weight: float = 10000.0
#     voxel_carving_enabled: bool = True
#     min_ray_length_m: float = 2.0
#     max_ray_length_m: float = 25.0
#     use_const_weight: bool = True
#     allow_clear: bool = True
#     use_weight_dropoff: bool = True
#     use_sparsity_compensation_factor: bool = True
#     sparsity_compensation_factor: float = 1.0
#     integrator_threads: int = 16

#     # Custom integrators configs

#     # Mode of the ThreadSafeIndex, determines the integration order of the
#     # rays. Options: "mixed", "sorted"
#     integration_order_mode: str = "mixed"

#     # merge integrator specific
#     enable_anti_grazing: bool = False

#     # fast integrator specific
#     start_voxel_subsampling_factor: float = 2.0
#     max_consecutive_ray_collisions: int = 2
#     clear_checks_every_n_frames: int = 1




@dataclass
class VoxBloxConfig:
    # Only contains members of NpTsdfIntegratorBase::Config that appear when
    # rosparam dumping while running launch files.
    curve_assumption: bool = False
    max_ray_length_m: float = 5.0
    min_ray_length_m: float = 0.1
    merge_with_clear: bool = True
    normal_available: bool = False
    reliable_band_ratio: float = 2.0
    reliable_normal_ratio_thre: float = 0.1
    use_const_weight: bool = False
    use_weight_dropoff: bool = True
    weight_reduction_exp: float = 1.0


    ### For NpTsdfServerConfig
    # Sensor specification (all but sensor_is_lidar have no default)
    width_: int = -1
    height_: int = -1
    max_range_: float = -1
    min_range_: float = -1
    smooth_thre_ratio_: float = -1
    sensor_is_lidar_: float = False
    # Camera (no default)
    vx_: int = -1
    vy_: int = -1
    fx_: int = -1
    fy_: int = -1
    # LiDAR (no default)
    fov_up_: float = -1
    fov_down_: float = -1
    fov_down_rad_: float = -1
    fov_rad_: float = -1
    # For preprocessing noise filter (mianly for KITTI)
    min_dist_: float = 0.1
    min_z_: float = -1000.0

    # voxels_per_side: int = 16
    # max_weight: float = 10000.0
    # voxel_carving_enabled: bool = True
    # allow_clear: bool = True
    # use_sparsity_compensation_factor: bool = True
    # sparsity_compensation_factor: float = 1.0
    # integrator_threads: int = 16

#     # Custom integrators configs

#     # Mode of the ThreadSafeIndex, determines the integration order of the
#     # rays. Options: "mixed", "sorted"
#     integration_order_mode: str = "mixed"

#     # merge integrator specific
#     enable_anti_grazing: bool = False

#     # fast integrator specific
#     start_voxel_subsampling_factor: float = 2.0
#     max_consecutive_ray_collisions: int = 2
#     clear_checks_every_n_frames: int = 1

# __default_config__ = asdict(VoxBloxConfig())
__default_config__ = asdict(VoxBloxConfig())
