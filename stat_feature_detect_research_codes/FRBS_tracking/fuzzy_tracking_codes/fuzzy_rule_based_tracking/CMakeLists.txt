cmake_minimum_required(VERSION 2.8)

if(POLICY CMP0020)
  cmake_policy(SET CMP0020 NEW)
endif()
 
PROJECT(Fuzzy_Rule_Based_Tracking)

include_directories(${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR})

set(CMAKE_CXX_FLAGS "-w -o3")

#########################################################
# Include Python
#########################################################
#find_package(PythonLibs REQUIRED)
#include_directories(${PYTHON_INCLUDE_DIRS})

#find_package( OpenCV REQUIRED )

#########################################################
# Include GLM
#########################################################
include_directories("/Users/sdutta/glm")

#########################################################
# VTK
#########################################################
find_package(VTK REQUIRED)
include(${VTK_USE_FILE})

file(GLOB CXX_FILES *.cxx)

add_executable(fuzzy_tracking ${CXX_FILES})

target_link_libraries(fuzzy_tracking ${VTK_LIBRARIES})
