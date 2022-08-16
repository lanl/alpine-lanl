cmake_minimum_required (VERSION 2.8)

if(POLICY CMP0020)
  cmake_policy(SET CMP0020 NEW)
endif()

project (Draw_tracking_graph)

set(CMAKE_CXX_FLAGS "-w") 

include_directories(${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR})

##Include GLM library
include_directories("/Users/sdutta/glm")

#########################################################
# FIND GLUT
#########################################################
find_package(GLUT REQUIRED)
include_directories(${GLUT_INCLUDE_DIRS})
link_directories(${GLUT_LIBRARY_DIRS})
add_definitions(${GLUT_DEFINITIONS})
if(NOT GLUT_FOUND)
    message(ERROR " GLUT not found!")
endif(NOT GLUT_FOUND)

#########################################################
# FIND OPENGL
#########################################################
find_package(OpenGL REQUIRED)
include_directories(${OpenGL_INCLUDE_DIRS})
link_directories(${OpenGL_LIBRARY_DIRS})
add_definitions(${OpenGL_DEFINITIONS})
if(NOT OPENGL_FOUND)
    message(ERROR " OPENGL not found!")
endif(NOT OPENGL_FOUND)

#########################################################
# FIND GLUI
#########################################################
FIND_LIBRARY (GLUI_LIBRARY NAMES glui
	    PATHS
	    ENV LD_LIBRARY_PATH
	    ENV LIBRARY_PATH
	    /usr/lib64
	    /usr/lib
	    /usr/local/lib64
	    /usr/include/GL
	    /usr/local/lib
	    /opt/local/lib
	    )

FIND_PATH (GLUI_INCLUDES GL/glui.h
	    ENV CPATH
	    /usr/include
		/usr/include/GL
	    /usr/local/include
	    /opt/local/include
	    )
	
IF(GLUI_INCLUDES AND GLUI_LIBRARY)
	    SET(GLUI_FOUND TRUE)
ENDIF(GLUI_INCLUDES AND GLUI_LIBRARY)

# add the executable
file(GLOB CXX_FILES *.cxx)

add_executable(main ${CXX_FILES})


#########################################################
# Link the libraries to the executables
#########################################################
TARGET_LINK_LIBRARIES(main ${OPENGL_LIBRARIES} ${GLUT_LIBRARIES})