cmake_minimum_required(VERSION 3.0.2)
project(mppi_cuda_node)

## Find catkin macros and libraries
find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  nav_msgs
  mav_msgs
  roscpp
  rospy
  sensor_msgs
  std_msgs
  message_generation
  dynamic_reconfigure
)

# Optionally set the CMP0148 policy to suppress Python-related warnings.
if(POLICY CMP0148)
  cmake_policy(SET CMP0148 NEW)
endif()

## This macro ensures that modules and global scripts declared in setup.py get installed
catkin_python_setup()

## Generate dynamic reconfigure parameters in the 'cfg' folder
## (Must be called after catkin_python_setup() and before catkin_package())
generate_dynamic_reconfigure_options(
  cfg/MPCParams.cfg
  cfg/MPPIParams.cfg
)
# add_dependencies(MPCParams ${PROJECT_NAME}_gencfg)

###################################
## catkin specific configuration ##
###################################
catkin_package(
  LIBRARIES mppi_cuda_node
  CATKIN_DEPENDS geometry_msgs nav_msgs mav_msgs roscpp rospy sensor_msgs std_msgs message_runtime
)

## Specify additional locations of header files
include_directories(
  ${catkin_INCLUDE_DIRS}
)

## Mark executable scripts (Python etc.) for installation
file(GLOB_RECURSE PYTHON_SCRIPTS "scripts/**/*.py")

catkin_install_python(PROGRAMS ${PYTHON_SCRIPTS}
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

# (Optional) Install additional directories if needed.
# install(DIRECTORY controllers misc
#   DESTINATION ${CATKIN_PACKAGE_PYTHON_DESTINATION}
#   FILES_MATCHING PATTERN "*.py"
# )
