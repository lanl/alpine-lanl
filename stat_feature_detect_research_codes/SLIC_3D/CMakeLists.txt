cmake_minimum_required(VERSION 2.8)

if(POLICY CMP0020)
  cmake_policy(SET CMP0020 NEW)
endif()
 
PROJECT(SLIC_3D)

include_directories(${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -w -O3")

#########################################################
# Include VTK
#########################################################
find_package(VTK REQUIRED)
include(${VTK_USE_FILE})

#########################################################
# Include OpenCV
#########################################################
#find_package( OpenCV REQUIRED )

#########################################################
# Include Python
#########################################################
#find_package(PythonLibs REQUIRED)
#include_directories(${PYTHON_INCLUDE_DIRS})

file(GLOB CXX_FILES *.cxx *.h)

add_executable(slic ${CXX_FILES})

target_link_libraries(slic ${VTK_LIBRARIES})
