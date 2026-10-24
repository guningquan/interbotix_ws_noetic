cmake_minimum_required(VERSION 2.8.3)
project(aloha)

## Compile as C++11, supported in ROS Kinetic and newer
add_compile_options(-std=c++11)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  interbotix_xs_msgs
  interbotix_xsarm_control
  rosbag
  roscpp
  tf2_ros
  rviz
  sensor_msgs
  std_msgs
  std_srvs
)

###################################
## catkin specific configuration ##
###################################
## The catkin_package macro generates cmake config files for your package
## Declare things to be passed to dependent projects
catkin_package(
 CATKIN_DEPENDS interbotix_xs_msgs interbotix_xsarm_control rosbag roscpp tf2_ros rviz sensor_msgs std_msgs std_srvs
)

#############
## Install ##
#############

## This package ships two files for the XENSE tactile publisher:
##   1. xense_tactile_publisher.py  — the actual Python implementation
##   2. xense_tactile_runner.sh     — bash launcher that invokes it via the
##      lerobot conda-env Python (which has xensesdk installed).
##
## ROS Noetic's catkin generates Python wrappers with #!/usr/bin/python3,
## ignoring custom shebangs, so we publish the bash runner as the ROS
## executable that roslaunch calls.

install(PROGRAMS
  scripts/xense_tactile_runner.sh
  scripts/xense_tactile_publisher.py
  scripts/xense_test_runner.sh
  scripts/xense_test_viewer.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
  ${catkin_INCLUDE_DIRS}
)

## Ensure the bash runner is reachable in the devel space as well.
## install(PROGRAMS ...) only installs to the install space, so we
## symlink it into <devel>/lib/aloha/ so roslaunch can find it.
add_custom_target(aloha_devel_tactile_runner ALL
  COMMAND ${CMAKE_COMMAND} -E remove -f
    "${CATKIN_DEVEL_PREFIX}/lib/${PROJECT_NAME}/xense_tactile_runner.sh"
  COMMAND ${CMAKE_COMMAND} -E create_symlink
    "${CMAKE_CURRENT_SOURCE_DIR}/scripts/xense_tactile_runner.sh"
    "${CATKIN_DEVEL_PREFIX}/lib/${PROJECT_NAME}/xense_tactile_runner.sh"
  WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
  COMMENT "Symlinking xense_tactile_runner.sh into devel space"
  VERBATIM
)
