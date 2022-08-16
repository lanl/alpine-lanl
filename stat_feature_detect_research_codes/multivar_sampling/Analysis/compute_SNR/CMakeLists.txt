cmake_minimum_required(VERSION 2.8)

if(POLICY CMP0020)
  cmake_policy(SET CMP0020 NEW)
endif()
 
PROJECT(SNR_Analysis)

include_directories(${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR})

set(CMAKE_CXX_FLAGS "-w -O3 -luuid")

#########################################################
# Include VTK
#########################################################
find_package(VTK REQUIRED)
include(${VTK_USE_FILE})

#########################################################
# Include OpenCV
#########################################################
#find_package( OpenCV REQUIRED )

file(GLOB CXX_FILES *.cxx *.h)

add_executable(snr_analysis ${CXX_FILES})

target_link_libraries(snr_analysis  ${VTK_LIBRARIES})
