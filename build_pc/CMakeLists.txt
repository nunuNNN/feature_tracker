cmake_minimum_required(VERSION 2.8)
project(feature_tracker)

set( CMAKE_CXX_FLAGS "-std=c++11" )


set( EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin )

SET(CMAKE_BUILD_TYPE Release) # Release, RelWithDebInfo , Debug
SET(CMAKE_VERBOSE_MAKEFILE OFF)
SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/CMakeModules/")

FIND_PACKAGE(OpenCV 4 REQUIRED)
FIND_PACKAGE(Eigen REQUIRED)

message("Eigen is found: ${Eigen_INCLUDE_DIRS}")
message("OpenCV is found: ${OpenCV_INCLUDE_DIRS}")

include_directories(
    ${PROJECT_SOURCE_DIR}/../src
    ${Eigen_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${Boost_INCLUDE_DIRS}
)

# Set link libraries
LIST(APPEND thirdparty_libraries 
  ${OpenCV_LIBS}
  ${Boost_LIBRARIES}
)
message("OpenCV_LIBS : ${OpenCV_LIBS}")

add_executable( test_tracking ${PROJECT_SOURCE_DIR}/../src/test_tracking.cpp )
target_link_libraries( test_tracking ${thirdparty_libraries} )
