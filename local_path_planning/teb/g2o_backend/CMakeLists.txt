cmake_minimum_required(VERSION 3.16)
project(g2o_teb_native LANGUAGES CXX)

find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)
find_package(g2o CONFIG REQUIRED)

pybind11_add_module(_g2o_teb_native g2o_native.cpp)
target_compile_features(_g2o_teb_native PRIVATE cxx_std_17)
target_link_libraries(_g2o_teb_native PRIVATE g2o::core g2o::solver_eigen)
set_target_properties(_g2o_teb_native PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/.."
  BUILD_RPATH "/opt/homebrew/opt/g2o/lib"
  INSTALL_RPATH "/opt/homebrew/opt/g2o/lib"
)
