#!/bin/bash
# Minimal setup.sh file for aerial_manipulation_mppi_realworld workspace
# This file is required by ROS setup.bash

# Set workspace path
export ROS_WORKSPACE="/home/dream_reaper/workspace/aerial_manipulation_mppi_realworld"

# Add packages to ROS package path
export ROS_PACKAGE_PATH="$ROS_WORKSPACE/src:$ROS_PACKAGE_PATH"

# Add Python path
export PYTHONPATH="$ROS_WORKSPACE/src:$PYTHONPATH"

echo "aerial_manipulation_mppi_realworld workspace setup complete"
