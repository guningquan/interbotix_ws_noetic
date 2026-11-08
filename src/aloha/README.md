# ALOHA: A Low-cost Open-source Hardware System for Bimanual Teleoperation

#### Project Website: https://tonyzhaozh.github.io/aloha/

This codebase contains implementation for teleoperation and data collection with the ALOHA hardware.
To build ALOHA, follow the [Hardware Assembly Tutorial](https://docs.google.com/document/d/1sgRZmpS7HMcZTPfGy3kAxDrqFMtNNzmK-yVtX5cKYME/edit?usp=sharing) and the quick start guide below.
To train imitation learning algorithms, you would also need to install [ACT](https://github.com/tonyzhaozh/act).

### Repo Structure
- ``config``: a config for each robot, designating the port they should bind to, more details in quick start guide.
- ``launch``: a ROS launch file for all 4 cameras and all 4 robots.
- ``aloha_scripts``: python code for teleop and data collection
- ``scripts``: the XENSE tactile publisher and viewer used by the tactile launch files; see
  [XENSE tactile sensing](#xense-tactile-sensing) at the end of this file.

## Quick start guide

### Hardware selection 

We suggest using a "heavy-duty" computer if possible. 

*In particular, at least 6 USB3 ports are needed. 4 ports for robot connections and 2 ports for cameras.* We have seen cases that a machine was not able to stably connect to all 4 robot arms simultaneously over USB, especially when USB hubs are used.

### Software selection -- OS:

Currently tested and working configurations: 
- :white_check_mark: Ubuntu 18.04 + ROS 1 noetic
- :white_check_mark: Ubuntu 20.04 + ROS 1 noetic

Ongoing testing (compatibility effort underway):
- :construction: ROS 2

### Software installation - ROS:
1. Install ROS and interbotix software following https://docs.trossenrobotics.com/interbotix_xsarms_docs/
2. This will create the directory ``~/interbotix_ws`` which contains ``src``.
3. git clone this repo inside ``~/interbotix_ws/src``
4. ``source /opt/ros/noetic/setup.sh && source ~/interbotix_ws/devel/setup.sh``
5. ``sudo apt-get install ros-noetic-usb-cam && sudo apt-get install ros-noetic-cv-bridge``
6. run ``catkin_make`` inside ``~/interbotix_ws``, make sure the build is successful
7. go to ``~/interbotix_ws/src/interbotix_ros_toolboxes/interbotix_xs_toolbox/interbotix_xs_modules/src/interbotix_xs_modules/arm.py``, find function ``publish_positions``.
   Change ``self.T_sb = mr.FKinSpace(self.robot_des.M, self.robot_des.Slist, self.joint_commands)`` to ``self.T_sb = None``.
   This prevents the code from calculating FK at every step which delays teleoperation.
### Hardware installation:

The goal of this section is to run ``roslaunch aloha 4arms_teleop.launch``, which starts
communication with 4 robots and 4 cameras. It should work after finishing the following steps:

Step 1: Connect 4 robots to the computer via USB, and power on. *Do not use extension cable or usb hub*.
- To check if the robot is connected, install dynamixel wizard [here](https://emanual.robotis.com/docs/en/software/dynamixel/dynamixel_wizard2/)
- Dynamixel wizard is a very helpful debugging tool that connects to individual motors of the robot. It allows
things such as rebooting the motor (very useful!), torque on/off, and sending commands.
However, it has no knowledge about the kinematics of the robot, so be careful about collisions.
The robot *will* collapse if motors are torque off i.e. there is no automatically engaged brakes in joints.
- Open Dynamixel wizard, go into ``options`` and select:
  - Protocal 2.0
  - All ports
  - 1000000 bps
  - ID range from 0-10
- Note: repeat above everytime before you scan.
- Then hit ``Scan``. There should be 4 devices showing up, each with 9 motors.


- One issue that arises is the port each robot binds to can change over time, e.g. a robot that
is initially ``ttyUSB0`` might suddenly become ``ttyUSB5``. To resolve this, we bind each robot to a fixed symlink
port with the following mapping:
  - ``ttyDXL_master_right``: right master robot (master: the robot that the operator would be holding)
  - ``ttyDXL_puppet_right``: right puppet robot (puppet: the robot that performs the task)
  - ``ttyDXL_master_left``: left master robot
  - ``ttyDXL_puppet_left``: left puppet robot
- Take ``ttyDXL_master_right``: right master robot as an example:
  1. Find the port that the right master robot is currently binding to, e.g. ``ttyUSB0``
  2. run ``udevadm info --name=/dev/ttyUSB0 --attribute-walk | grep serial`` to obtain the serial number. Use the first one that shows up, the format should look similar to ``FT6S4DSP``.
  3. ``sudo vim /etc/udev/rules.d/99-fixed-interbotix-udev.rules`` and add the following line: 

         SUBSYSTEM=="tty", ATTRS{serial}=="<serial number here>", ENV{ID_MM_DEVICE_IGNORE}="1", ATTR{device/latency_timer}="1", SYMLINK+="ttyDXL_master_right"

  4. This will make sure the right master robot is *always* binding to ``ttyDXL_master_right``
  5. Repeat with the rest of 3 arms.
- To apply the changes, run ``sudo udevadm control --reload && sudo udevadm trigger``
- If successful, you should be able to find ``ttyDXL*`` in your ``/dev``

Step 2: Set max current for gripper motors
- Open Dynamixel Wizard, and select the wrist motor for puppet arms. The name of it should be ```[ID:009] XM430-W350```
- Tip: the LED on the base of robot will flash when it is talking to Dynamixel Wizard. This will help determine which robot is selected. 
- Find ``38 Current Limit``, enter ``200``, then hit ``save`` at the bottom.
- Repeat this for both puppet robots.
- This limits the max current through gripper motors, to prevent overloading errors.


Step 3: Setup 4 cameras
- You may use usb hub here, but *maximum 2 cameras per hub for reasonable latency*.
- To make sure all 4 cameras are binding to a consistent port, similar steps are needed.
- Cameras are by default binding to ``/dev/video{0, 1, 2...}``, while we want to have symlinks ``{CAM_RIGHT_WRIST, CAM_LEFT_WRIST, CAM_LOW, CAM_HIGH}``
- Take ``CAM_RIGHT_WRIST`` as an example, and let's say it is now binding to ``/dev/video0``. run ``udevadm info --name=/dev/video0 --attribute-walk | grep serial`` to obtain it's serial. Use the first one that shows up, the format should look similar to ``0E1A2B2F``.
- Then ``sudo vim /etc/udev/rules.d/99-fixed-interbotix-udev.rules`` and add the following line 

      SUBSYSTEM=="video4linux", ATTRS{serial}=="<serial number here>", ATTR{index}=="0", ATTRS{idProduct}=="085c", ATTR{device/latency_timer}="1", SYMLINK+="CAM_RIGHT_WRIST"

- Repeat this for ``{CAM_LEFT_WRIST, CAM_LOW, CAM_HIGH}`` in additional to ``CAM_RIGHT_WRIST``
- To apply the changes, run ``sudo udevadm control --reload && sudo udevadm trigger``
- If successful, you should be able to find ``{CAM_RIGHT_WRIST, CAM_LEFT_WRIST, CAM_LOW, CAM_HIGH}`` in your ``/dev``

At this point, have a new terminal
    
    conda deactivate # if conda shows up by default
    source /opt/ros/noetic/setup.sh && source ~/interbotix_ws/devel/setup.sh
    roslaunch aloha 4arms_teleop.launch

If no error message is showing up, the computer should be successfully connected to all 4 cameras and all 4 robots.

#### Trouble shooting
- Make sure Dynamixel Wizard is disconnected, and no app is using webcam's stream. It will prevent ROS from connecting to
these devices.

### Software installation - Conda:

    conda create -n aloha python=3.8.10
    conda activate aloha
    pip install torchvision
    pip install torch
    pip install pyquaternion
    pip install pyyaml
    pip install rospkg
    pip install pexpect
    pip install mujoco==2.3.7
    pip install dm_control==1.0.14
    pip install opencv-python
    pip install matplotlib
    pip install einops
    pip install packaging
    pip install h5py

### Testing teleoperation

**Notice**: Before running the commands below, be sure to place all 4 robots in their sleep positions, and open master robot's gripper. 
All robots will rise to a height that is easy for teleoperation.

    # ROS terminal
    conda deactivate
    source /opt/ros/noetic/setup.sh && source ~/interbotix_ws/devel/setup.sh
    roslaunch aloha 4arms_teleop.launch
    
    # Right hand terminal
    conda activate aloha
    cd ~/interbotix_ws/src/aloha/aloha_scripts
    python3 one_side_teleop.py right
    
    # Left hand terminal
    conda activate aloha
    cd ~/interbotix_ws/src/aloha/aloha_scripts
    python3 one_side_teleop.py left

The teleoperation will start when the master side gripper is closed.


## Example Usages

To set up a new terminal, run:

    conda activate aloha
    cd ~/interbotix_ws/src/aloha/aloha_scripts


The ``one_side_teleop.py`` we ran is for testing teleoperation and has no data collection. To collect data for an episode, run:

    python3 record_episodes.py --dataset_dir <data save dir> --episode_idx 0

This will store a hdf5 file at ``<data save dir>``.
To change episode length and other params, edit ``constants.py`` directly.

To visualize the episode collected, run:

    python3 visualize_episodes.py --dataset_dir <data save dir> --episode_idx 0

To replay the episode collected with real robot, run:

    python3 replay_episodes.py --dataset_dir <data save dir> --episode_idx 0

To lower 4 robots before e.g. cutting off power, run:

    python3 sleep.py


## XENSE tactile sensing

Four XENSE G1-WS tactile sensors sit on the follower grippers. `scripts/xense_tactile_publisher.py`
reads them through the XENSE SDK (xensesdk 1.7.0) and publishes them on ROS for the recording, force
feedback and policy code in act-triple-plus-aloha2. ROS starts it through `scripts/xense_tactile_runner.sh`,
which runs it in the `lerobot` conda env (`/home/robot/miniforge3/envs/lerobot`, Python 3.10).

| Sensor | Serial |
| --- | --- |
| `left_gripper_inner` | OG000707 |
| `left_gripper_outer` | OG000635 |
| `right_gripper_inner` | OG000614 |
| `right_gripper_outer` | OG000706 |

### Launch files

| Launch file | What it starts |
| --- | --- |
| `4arms_tactile_teleop.launch` | 4 arms, 4 cameras, the XENSE publisher and the camera + tactile preview window. Use this with act-triple-plus-aloha2. |
| `4arms_tactile.launch` | 4 cameras and the XENSE publisher, no arm drivers; the publisher runs with its default settings. |
| `4arms_act_tactile.launch` | 4 arms, 4 cameras and tactile_acuity's own XENSE publisher (`/xense/<sensor>/frame`), for the tactile_acuity code. |
| `xense_test.launch` | A viewer for one sensor (rectify, difference and depth side by side). Publishes nothing. Do not run it while a publisher is using the sensors. |

### Starting it

```sh
# Default: rectify + force, sent raw. This is what aloha2 recording and force feedback need.
roslaunch aloha 4arms_tactile_teleop.launch extra_outputs:=force

# All three images (rectify, difference, depth) + force, sent raw
roslaunch aloha 4arms_tactile_teleop.launch extra_outputs:=difference,depth,force

# All three images + force; rectify/difference as JPEG (quality 90), depth zlib-compressed (lossless)
roslaunch aloha 4arms_tactile_teleop.launch extra_outputs:=difference,depth,force image_codec:=jpeg
```

Add `launch_driver:=false` to start the cameras and sensors without the arm drivers, and
`stats_log_period_s:=5` to print the frame-rate heartbeat every 5 s while comparing settings.

**Do not touch the gels while the launch starts:** the publisher zeroes every sensor before it sends
the first frame (`calibrate_on_start:=true`).

### Launch arguments

| Argument | Default | Meaning |
| --- | --- | --- |
| `primary_output` | `rectify` | The always-on image: `rectify`, `difference` or `depth`. It is also the image on the preview topic. |
| `extra_outputs` | empty | Comma-separated extra fields: `difference`, `depth`, `force`, `force_norm`, `force_resultant`, `marker2d`, `mesh3d`, `mesh3d_init`, `mesh3d_flow`, `timestamp`. Force feedback needs `force`. |
| `publish_mode` | `on_new_frame` | Each sensor's own thread sends every new SDK frame as soon as it is read. `timer` restores the old behaviour: the newest frame of every sensor re-sent every 1/60 s. |
| `max_rate_hz` | `60` | `on_new_frame` only: at most this many messages per second per sensor. Frames above the cap are dropped, never delayed; 0 = no cap. |
| `image_codec` | `raw` | `raw`: rectify/difference as RGB uint8 arrays. `jpeg`: as JPEG bytes, with depth zlib-compressed. |
| `jpeg_quality` | `90` | JPEG quality used with `image_codec:=jpeg`. |
| `calibrate_on_start` | `true` | Zero every sensor before the first frame. |
| `stats_log_period_s` | `60` | Seconds between heartbeat lines; 0 = off. |
| `tactile_rotation` | `cw` | Preview window only (`cw` or `ccw`): turns the portrait 700×400 frames to landscape. The published data is never rotated. |
| `launch_driver` | `true` | `false` = cameras and tactile sensors only, no arm drivers. |

### Topics

- `/tactile/<sensor>/raw` (`std_msgs/UInt8MultiArray`): `pickle.dumps({"frames": {...}, "meta": {...}})`.
  This is the topic to record from.
  - `frames` holds the primary output plus `extra_outputs`, all from one SDK call, so every field is
    from the same sensor frame. Everything is at native resolution, with no resizing: rectify and
    difference (700, 400, 3) uint8 RGB, depth (700, 400) float32 in mm, force (35, 20, 3).
    With `image_codec:=jpeg`, rectify/difference are 1-D uint8 JPEG bytes (`cv2.imdecode` returns BGR)
    and depth is zlib-compressed float32; `meta["encoding"]` lists the encoded fields.
  - `meta`: `local_read_start` / `local_read_end` (host `time.time()` around the SDK call, so
    `now - local_read_end` is the frame's age), `timestamp` (the SDK's own TimeStamp),
    `timestamp_source`, `seq` (frames read from the SDK; gaps are frames dropped by the cap),
    `image_codec`, `encoding`.
- `/tactile/<sensor>/<primary_output>` (`sensor_msgs/Image`): a preview for the monitor window, every
  2nd sent frame (rgb8, or 32FC1 in mm for depth).
- `/tactile/runtime_config` (latched): every sensor's SDK runtime config, exported right after the last
  zeroing, plus the SDK version and the publisher's settings. The recorders store it with each episode.
- `/tactile/calibrate` (`std_srvs/Trigger`): zero all sensors again. Nothing may touch the gels while
  it runs. aloha2's recorders call it at the start of every episode.

In act-triple-plus-aloha2, `aloha_scripts/tactile_stream.py` reads `/raw` in both codecs.

### Heartbeat

The publisher prints one line every `stats_log_period_s` seconds:

```text
xense_tactile_publisher: left_gripper_inner read <ms> ms, new <n>/s, sent <n>/s; left_gripper_outer ...
```

`read` is the median SDK call time, `new` the number of new frames per second read from the SDK, and
`sent` the messages per second. `(capped <n>/s)` appears when `max_rate_hz` drops frames. `new` is the
number to compare between settings. The first line or two include start-up, so read the later ones.

### raw or jpeg?

Measured on this machine on 2026-11-08:

- **rectify + force (the default), raw:** a 50 Hz recording loop got a new frame on 98% of its steps;
  frame age p50 9 ms, p95 19 ms, max 37 ms. Keep `raw`.
- **All three images:** about 43 new frames/s per sensor with `raw`, 52–60 with `jpeg`. Each message
  is about 2.8 MB raw and about 0.15 MB as jpeg. Use `jpeg`.
- `jpeg` also shrinks what the recorder keeps in RAM during an episode (rectify: 0.84 MB to about
  28 KB per frame) and removes the JPEG encoding when an episode is saved. The saved HDF5 is the same
  either way, because aloha2 stores tactile images as JPEG quality 90.

### Cameras

`usb_cam` publishes raw `image_raw` (rgb8, 640×480, 0.92 MB per frame). Keep the cameras raw when the
program reading them runs on this PC. Measured with all cameras and sensors running, sending them
compressed (ROS `image_raw/compressed`, JPEG quality 90) instead:

| | raw | compressed (q90) |
| --- | --- | --- |
| Size per frame | 0.92 MB | 0.07 MB |
| Latency p50 / p95 / max | 1.9 / 5.4 / 30 ms | 3.7 / 6.2 / 36 ms |
| Decoding at the receiver | none | 1.4 ms per frame |
| CPU of the 4 camera nodes | 28% | 96% |

Compress only to send images to another machine. The compressed topic is produced only while something
subscribes to it, and its quality defaults to 80. To set 90:

```sh
for c in high low left_wrist right_wrist; do
  rosrun dynamic_reconfigure dynparam set /usb_cam_$c/image_raw/compressed jpeg_quality 90
done
```

aloha2's recorders read `image_raw` only.

