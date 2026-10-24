#!/usr/bin/env python3

import sys
import rospy
import pyrealsense2 as rs  # Intel RealSense cross-platform open-source API
from aloha_pro.msg import RGBGrayscaleImage
from cv_bridge import CvBridge
import time
import numpy as np
from collections import defaultdict

TIMEOUT_MS = 100
FPS = 60

# import IPython
# e = IPython.embed

# Single D415 camera configuration
CAMERA_NAME = "cam_high_rs"
D415_SERIAL_NUMBER = None  # Will be auto-detected

# Wait for camera initialization
time.sleep(2)


cv_bridge = CvBridge()
rospy.init_node("realsense_publisher")

print(f"\n\nSTARTING REALSENSE PUBLISHER FOR D415 CAMERA: {CAMERA_NAME}\n\n")

ctx = rs.context()
devices = ctx.query_devices()

# Auto-detect D415 camera
D415_SERIAL_NUMBER = None
for device in devices:
    device_info = device.get_info(rs.camera_info.serial_number)
    device_name = device.get_info(rs.camera_info.name)
    print(f"Found device: {device_name}, Serial: {device_info}")
    if "D415" in device_name:
        D415_SERIAL_NUMBER = device_info
        print(f"D415 camera detected with serial: {D415_SERIAL_NUMBER}")
        break

if D415_SERIAL_NUMBER is None:
    print("ERROR: No D415 camera found!")
    exit(1)

# Hardware reset for D415
for dev in devices:
    if dev.get_info(rs.camera_info.serial_number) == D415_SERIAL_NUMBER:
        dev.hardware_reset()
        break

# Single camera configuration
pipe = rs.pipeline()
cfg = rs.config()
profile = None
depth_scale = None

# D415 specific configuration
D415_INTENSITY_SET_POINT = 500  # Optimized for D415 lighting conditions

# Configure single D415 camera
print(f"Configuring D415 camera with serial: {D415_SERIAL_NUMBER}")
cfg.enable_device(D415_SERIAL_NUMBER)
cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, FPS)
cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, FPS)
profile = pipe.start(cfg)
device = profile.get_device()

# D415 camera configuration (D415 does not support advanced mode)
print("D415 camera detected - using standard configuration")
print("Note: D415 does not support rs400_advanced_mode, using default settings")

# Optional: Set basic camera options if needed
try:
    # Set basic exposure control for D415
    color_sensor = device.first_color_sensor()
    if color_sensor:
        
        # Enable auto exposure (this is supported by D415)
        color_sensor.set_option(rs.option.enable_auto_exposure, True)
        print("D415 auto exposure enabled")
        
        # Set exposure value if needed (optional)
        # color_sensor.set_option(rs.option.exposure, 100)
        # color_sensor.set_option(rs.option.enable_auto_exposure, False)
        # color_sensor.set_option(rs.option.exposure, 1200)  # 12 ms
        # color_sensor.set_option(rs.option.gain, 16)
        # print("Manual Exposure enabled (auto exposure disabled)")
        
except Exception as e:
    print(f"D415 basic configuration failed: {e}")
    print("Continuing with default camera settings")

# Get depth scale for D415
depth_sensor = device.first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print(f"D415 Depth Scale is: {depth_scale}")

print("\nWAITING FOR D415 FRAMES\n")
for _ in range(3):
    t = time.time()
    try:
        pipe.wait_for_frames()
        print(f"{CAMERA_NAME} waited {time.time() - t}s")
    except:
        print(f"{CAMERA_NAME} waited too long: {time.time() - t}s\n\n")
        raise Exception

# Single publisher for D415 camera
publisher = rospy.Publisher(CAMERA_NAME, RGBGrayscaleImage, queue_size=1)

# i = 0

print("\n\nD415 REALSENSE PUBLISHER RUNNING\n\n")
no_error = True
t = time.time()
while not rospy.is_shutdown():
    t0 = time.time()
    
    try:
        frameset = pipe.wait_for_frames(timeout_ms=TIMEOUT_MS)
    except:
        print(f"\n\n{CAMERA_NAME} failed\n")
        no_error = False
        pipe.stop()
        break

    color_frame = np.array(frameset.get_color_frame().get_data())
    depth_frame = np.array(frameset.get_depth_frame().get_data())

    # Create message for D415 camera
    msg = RGBGrayscaleImage()
    msg.header.stamp = rospy.Time.now()
    msg.images.append(cv_bridge.cv2_to_imgmsg(color_frame, encoding="bgr8"))
    msg.images.append(cv_bridge.cv2_to_imgmsg(depth_frame, encoding="mono16"))
    
    # Publish message
    publisher.publish(msg)
    
    # Optional: Print depth information for debugging
    # h, w = depth_frame.shape
    # half_block_size = 5
    # if i % 15 == 0:
    #     pixel_depth = depth_frame[h//2-half_block_size:h//2+half_block_size, w//2-half_block_size:w//2+half_block_size].mean()
    #     print(f"{CAMERA_NAME} depth mean (m): {pixel_depth * depth_scale}")
    
    # print(f"D415 realsense_publisher {time.time() - t0:.4f}s")
    # i += 1
