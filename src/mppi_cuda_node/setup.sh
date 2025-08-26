#!/bin/bash
# Minimal setup.sh file for mppi_cuda_node package
# This file is required by the workspace setup.bash

# Set package path
export ROS_PACKAGE_PATH="/home/dream_reaper/workspace/aerial_manipulation_mppi_realworld/src:$ROS_PACKAGE_PATH"

# Add the package to Python path
export PYTHONPATH="/home/dream_reaper/workspace/aerial_manipulation_mppi_realworld/src/mppi_cuda_node:$PYTHONPATH"

echo "mppi_cuda_node package setup complete"
