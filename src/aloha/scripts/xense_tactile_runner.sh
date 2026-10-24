#!/bin/bash
# Launcher for xense_tactile_publisher.py that forces the lerobot conda env
# which has xensesdk + rospy + cv_bridge installed.
#
# ROS Noetic's catkin ignores the script shebang and uses system python3,
# so we provide this shell wrapper that invokes the correct interpreter.

LEROBOT_ENV="/home/robot/miniforge3/envs/lerobot"

# xensesdk's inference engine (onnxruntime-gpu) needs CUDA 12.x + cuDNN 9.x
# shared libraries at runtime. The system-wide CUDA_HOME (/usr/local/cuda-12.1)
# is incomplete on this machine, so onnxruntime silently falls back to CPU
# ("infer session using CPU" in the logs) instead of erroring. The correct
# libs are already present via pip (nvidia-*-cu12, pulled in by torch) inside
# this conda env's site-packages -- just not on LD_LIBRARY_PATH by default.
# Prepend them so the CUDA execution provider actually loads.
for _nvidia_lib_dir in "$LEROBOT_ENV"/lib/python3.*/site-packages/nvidia/*/lib; do
    [ -d "$_nvidia_lib_dir" ] && LD_LIBRARY_PATH="$_nvidia_lib_dir:$LD_LIBRARY_PATH"
done
export LD_LIBRARY_PATH
unset _nvidia_lib_dir

exec "$LEROBOT_ENV/bin/python" -u \
    "$(dirname "$0")/xense_tactile_publisher.py" \
    "$@"
