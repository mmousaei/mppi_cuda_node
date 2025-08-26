# with-ros.sh
#!/bin/bash
set -eu

# 1) Source system ROS (pick one)
[ -f /opt/ros/noetic/setup.bash ] && source /opt/ros/noetic/setup.bash
# [ -f /opt/ros/humble/setup.bash ] && source /opt/ros/humble/setup.bash

# 2) Source your workspace (ROS 1 uses devel/, ROS 2 uses install/)
[ -f "$PWD/devel/setup.bash" ] && source ~/workspace/aerial_manipulation_mppi_realworld/devel/setup.bash
# [ -f "$PWD/install/setup.bash" ] && source "$PWD/install/setup.bash"

exec "$@"
