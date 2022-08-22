cmake_minimum_required(VERSION 3.13)
project(compute_cluster_stat)

#########################################################
# Include VTK-m
#########################################################
find_package(VTKm REQUIRED QUIET)

add_executable(gaussian_similarity main.cxx)
target_link_libraries(gaussian_similarity PRIVATE vtkm_filter vtkm_cont vtkm_rendering)
vtkm_add_target_information(gaussian_similarity DROP_UNUSED_SYMBOLS MODIFY_CUDA_FLAGS DEVICE_SOURCES main.cxx)
