cmake_minimum_required(VERSION 2.8)

if(POLICY CMP0020)
  cmake_policy(SET CMP0020 NEW)
endif()
 
PROJECT(Fuzzy_Multivariate_Sampling_Histogram)

include_directories(${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR})

set(CMAKE_CXX_FLAGS "-w -o3")

#########################################################
# Include GLM
#########################################################
include_directories("/Users/sdutta/glm/")

#########################################################
# Include VTK
#########################################################
find_package(VTK REQUIRED)
include(${VTK_USE_FILE})

file(GLOB CXX_FILES *.cxx)

add_executable(sample_data ${CXX_FILES})

target_link_libraries(sample_data ${VTK_LIBRARIES})
