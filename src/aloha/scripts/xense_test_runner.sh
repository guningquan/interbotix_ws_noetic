#!/bin/bash
# Launcher for xense_test_viewer.py. Same reasons and same environment setup
# as xense_tactile_runner.sh: catkin ignores the script shebang, so this
# wrapper picks the lerobot conda env (which has xensesdk), and puts the
# pip-installed CUDA/cuDNN libs on LD_LIBRARY_PATH so xensesdk's onnxruntime
# runs on GPU instead of silently falling back to CPU.

LEROBOT_ENV="/home/robot/miniforge3/envs/lerobot"

for _nvidia_lib_dir in "$LEROBOT_ENV"/lib/python3.*/site-packages/nvidia/*/lib; do
    [ -d "$_nvidia_lib_dir" ] && LD_LIBRARY_PATH="$_nvidia_lib_dir:$LD_LIBRARY_PATH"
done
export LD_LIBRARY_PATH
unset _nvidia_lib_dir

# readlink -f: resolve to scripts/ even if this file is reached via a symlink
# (e.g. one in devel/lib/aloha), where the .py does not live.
exec "$LEROBOT_ENV/bin/python" -u \
    "$(dirname "$(readlink -f "$0")")/xense_test_viewer.py" \
    "$@"
