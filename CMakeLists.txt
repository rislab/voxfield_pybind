cmake_minimum_required(VERSION 2.8.3)
project(voxfield_pybind)

set(CMAKE_CXX_STANDARD 17)

# finds dependencies for this package
find_package(catkin REQUIRED COMPONENTS
        pybind11_catkin
        voxblox
        voxblox_ros
            )

# declares dependencies for packages that depend on this package
catkin_package(LIBRARIES
        CATKIN_DEPENDS
	      pybind11_catkin
          voxblox
          voxblox_ros
        )

catkin_python_setup()

include_directories(
  ${catkin_INCLUDE_DIRS}
  ${pybind11_catkin_INCLUDE_DIRS}
)

# pybind modules
pybind_add_module(voxfield_pybind src/voxblox/pybind/voxblox_pybind.cpp)
target_include_directories(voxfield_pybind PUBLIC ${pybind11_catkin_INCLUDE_DIRS})