cmake_minimum_required(VERSION 2.8)

if(POLICY CMP0020)
  cmake_policy(SET CMP0020 NEW)
endif()
 
PROJECT(feature_analysis)

include_directories(${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR})
set(CMAKE_CXX_FLAGS "-w -o3")


#########################################################
# VTK
#########################################################
find_package(VTK REQUIRED)
include(${VTK_USE_FILE})

file(GLOB CXX_FILES *.cpp)
add_executable(feature_analysis ${CXX_FILES})
target_link_libraries(feature_analysis ${VTK_LIBRARIES})
