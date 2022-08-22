cmake_minimum_required(VERSION 3.13)
project(SLIC_based_Statistical_Feature_Detection)

#########################################################
# Include VTK-m
#########################################################
add_executable(stat_feature_detect main.cxx)
#target_compile_options(stat_feature_detect PUBLIC -fsanitize=address)
#target_link_options(stat_feature_detect PUBLIC -fsanitize=address)
target_link_libraries(stat_feature_detect PRIVATE vtkm_filter vtkm_cont vtkm_rendering)
vtkm_add_target_information(stat_feature_detect DROP_UNUSED_SYMBOLS MODIFY_CUDA_FLAGS DEVICE_SOURCES main.cxx)
