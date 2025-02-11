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
)

## Uncomment this if the package has a setup.py. This macro ensures
## modules and global scripts declared therein get installed
## See http://ros.org/doc/api/catkin/html/user_guide/setup_dot_py.html
catkin_python_setup()

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

# install(DIRECTORY controllers misc
  # DESTINATION ${CATKIN_PACKAGE_PYTHON_DESTINATION}
  # FILES_MATCHING PATTERN "*.py"
# )
