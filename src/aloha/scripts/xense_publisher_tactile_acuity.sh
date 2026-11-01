#!/bin/bash
# Runs tactile_acuity's xense_publisher.py (the /xense/<sensor>/frame publisher that
# record_episodes.py, eval_episodes.py and run_baseline.py subscribe to) from roslaunch,
# in the lerobot conda env that has xensesdk + rospy. Same reason as xense_tactile_runner.sh:
# catkin ignores the script's shebang. The publisher puts the bundled CUDA libraries on the
# loader path itself (xense_recorder.ensure_cuda_library_path) before touching the SDK.
LEROBOT_ENV="/home/robot/miniforge3/envs/lerobot"
REPO="/mnt/tao_ssd1/Programs_server/tactile_acuity"
exec "$LEROBOT_ENV/bin/python" -u "$REPO/aloha_scripts/xense_publisher.py" "$@"
