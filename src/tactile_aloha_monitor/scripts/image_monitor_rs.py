#!/usr/bin/env python3
import numpy as np
import time
import cv2
import rospy
from aloha_pro.msg import RGBGrayscaleImage

class ImageRecorder:
    def __init__(self, init_node=True, is_debug=False):
        from collections import deque
        import rospy
        from cv_bridge import CvBridge
        from sensor_msgs.msg import Image

        self.is_debug = is_debug
        self.bridge = CvBridge()
        self.camera_names = ["cam_high_rs"]

        if init_node:
            rospy.init_node("image_recorder", anonymous=True)
        for cam_name in self.camera_names:
            setattr(self, f"{cam_name}_rgb_image", None)
            setattr(self, f"{cam_name}_depth_image", None)
            setattr(self, f"{cam_name}_timestamp", 0.0)
            if cam_name == "cam_high":
                callback_func = self.image_cb_cam_high
            elif cam_name == "cam_low":
                callback_func = self.image_cb_cam_low
            elif cam_name == "cam_left_wrist":
                callback_func = self.image_cb_cam_left_wrist
            elif cam_name == "cam_right_wrist":
                callback_func = self.image_cb_cam_right_wrist
            elif cam_name == "cam_high_rs":
                callback_func = self.image_cb_cam_high_rs
            else:
                raise NotImplementedError
            rospy.Subscriber(f"/{cam_name}", RGBGrayscaleImage, callback_func)
            if self.is_debug:
                setattr(self, f"{cam_name}_timestamps", deque(maxlen=50))

        self.cam_last_timestamps = {cam_name: 0.0 for cam_name in self.camera_names}
        time.sleep(0.5)

    def image_cb(self, cam_name, data):
        setattr(
            self,
            f"{cam_name}_rgb_image",
            self.bridge.imgmsg_to_cv2(data.images[0], desired_encoding="bgr8"),
        )
        setattr(
            self,
            f"{cam_name}_depth_image",
            self.bridge.imgmsg_to_cv2(data.images[1], desired_encoding="mono16"),
        )
        setattr(
            self,
            f"{cam_name}_timestamp",
            data.header.stamp.secs + data.header.stamp.nsecs * 1e-9,
        )
        # setattr(self, f'{cam_name}_secs', data.images[0].header.stamp.secs)
        # setattr(self, f'{cam_name}_nsecs', data.images[0].header.stamp.nsecs)
        # cv2.imwrite('/home/lucyshi/Desktop/sample.jpg', cv_image)
        if self.is_debug:
            getattr(self, f"{cam_name}_timestamps").append(
                data.images[0].header.stamp.secs
                + data.images[0].header.stamp.nsecs * 1e-9
            )

    def image_cb_cam_high(self, data):
        cam_name = "cam_high"
        return self.image_cb(cam_name, data)

    def image_cb_cam_low(self, data):
        cam_name = "cam_low"
        return self.image_cb(cam_name, data)

    def image_cb_cam_left_wrist(self, data):
        cam_name = "cam_left_wrist"
        return self.image_cb(cam_name, data)

    def image_cb_cam_right_wrist(self, data):
        cam_name = "cam_right_wrist"
        return self.image_cb(cam_name, data)

    def image_cb_cam_high_rs(self, data):
        cam_name = "cam_high_rs"
        return self.image_cb(cam_name, data)

    def get_images(self):
        image_dict = dict()
        for cam_name in self.camera_names:
            while (
                getattr(self, f"{cam_name}_timestamp")
                <= self.cam_last_timestamps[cam_name]
            ):
                time.sleep(0.00001)
            rgb_image = getattr(self, f"{cam_name}_rgb_image")
            depth_image = getattr(self, f"{cam_name}_depth_image")
            self.cam_last_timestamps[cam_name] = getattr(self, f"{cam_name}_timestamp")
            image_dict[cam_name] = rgb_image
            image_dict[f"{cam_name}_depth"] = depth_image
        return image_dict

    def print_diagnostics(self):
        def dt_helper(l):
            l = np.array(l)
            diff = l[1:] - l[:-1]
            return np.mean(diff)

        for cam_name in self.camera_names:
            image_freq = 1 / dt_helper(getattr(self, f"{cam_name}_timestamps"))
            print(f"{cam_name} {image_freq=:.2f}")
        print()


class Recorder:
    def __init__(self, side, init_node=True, is_debug=False):
        from collections import deque
        import rospy
        from sensor_msgs.msg import JointState
        from interbotix_xs_msgs.msg import JointGroupCommand, JointSingleCommand

        self.secs = None
        self.nsecs = None
        self.qpos = None
        self.effort = None
        self.arm_command = None
        self.gripper_command = None
        self.is_debug = is_debug

        if init_node:
            rospy.init_node("recorder", anonymous=True)
        rospy.Subscriber(
            f"/puppet_{side}/joint_states", JointState, self.puppet_state_cb
        )
        rospy.Subscriber(
            f"/puppet_{side}/commands/joint_group",
            JointGroupCommand,
            self.puppet_arm_commands_cb,
        )
        rospy.Subscriber(
            f"/puppet_{side}/commands/joint_single",
            JointSingleCommand,
            self.puppet_gripper_commands_cb,
        )
        if self.is_debug:
            self.joint_timestamps = deque(maxlen=50)
            self.arm_command_timestamps = deque(maxlen=50)
            self.gripper_command_timestamps = deque(maxlen=50)
        time.sleep(0.1)

    def puppet_state_cb(self, data):
        self.qpos = data.position
        self.qvel = data.velocity
        self.effort = data.effort
        self.data = data
        if self.is_debug:
            self.joint_timestamps.append(time.time())

    def puppet_arm_commands_cb(self, data):
        self.arm_command = data.cmd
        if self.is_debug:
            self.arm_command_timestamps.append(time.time())

    def puppet_gripper_commands_cb(self, data):
        self.gripper_command = data.cmd
        if self.is_debug:
            self.gripper_command_timestamps.append(time.time())

    def print_diagnostics(self):
        def dt_helper(l):
            l = np.array(l)
            diff = l[1:] - l[:-1]
            return np.mean(diff)

        joint_freq = 1 / dt_helper(self.joint_timestamps)
        arm_command_freq = 1 / dt_helper(self.arm_command_timestamps)
        gripper_command_freq = 1 / dt_helper(self.gripper_command_timestamps)

        print(
            f"{joint_freq=:.2f}\n{arm_command_freq=:.2f}\n{gripper_command_freq=:.2f}\n"
        )


def process_depth_image(depth_image, max_depth=1000):
    """
    Process depth image for display
    Args:
        depth_image: 16-bit depth image
        max_depth: Maximum depth value for normalization (in mm)
    Returns:
        Processed 8-bit depth image with colormap
    """
    if depth_image is None:
        return None
    
    # Convert to 8-bit for display
    depth_8bit = np.uint8(depth_image / 16)  # Convert from 16-bit to 8-bit
    
    # Apply colormap for better visualization
    depth_colormap = cv2.applyColorMap(depth_8bit, cv2.COLORMAP_JET)
    
    return depth_colormap


def main():
    # Initialize the image recorder for D415 camera
    recorder = ImageRecorder(init_node=True)
    
    print("D415 RealSense Camera Monitor Started")
    print("Press 'q' to quit")
    print("Displaying RGB and Depth images side by side")
    
    # Main display loop
    while not rospy.is_shutdown():
        try:
            # Get images from D415 camera
            images = recorder.get_images()
            
            # Get RGB and depth images from cam_high_rs
            rgb_image = images.get('cam_high_rs')
            depth_image = images.get('cam_high_rs_depth')
            
            # Process images for display
            if rgb_image is not None and depth_image is not None:
                # Resize images for display
                rgb_display = cv2.resize(rgb_image, (640, 480))
                depth_display = cv2.resize(depth_image, (640, 480))
                
                # Process depth image with colormap
                depth_colormap = process_depth_image(depth_display)
                
                # Add titles to images
                cv2.putText(rgb_display, 'D415 RGB Camera', (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(depth_colormap, 'D415 Depth Camera', (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Combine images side by side
                combined_image = np.hstack((rgb_display, depth_colormap))
                
                # Display the combined image
                cv2.imshow('D415 RealSense Monitor - RGB & Depth', combined_image)
                
            elif rgb_image is not None:
                # Only RGB available
                rgb_display = cv2.resize(rgb_image, (640, 480))
                cv2.putText(rgb_display, 'D415 RGB Camera (No Depth)', (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Create blank depth image
                blank_depth = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(blank_depth, 'NO DEPTH SIGNAL', (150, 240), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)
                
                # Combine images
                combined_image = np.hstack((rgb_display, blank_depth))
                cv2.imshow('D415 RealSense Monitor - RGB & Depth', combined_image)
                
            else:
                # No signal
                blank_image = np.zeros((480, 1280, 3), dtype=np.uint8)  # Wider for side-by-side
                cv2.putText(blank_image, 'NO SIGNAL', (500, 240), 
                           cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
                cv2.imshow('D415 RealSense Monitor - RGB & Depth', blank_image)
            
            # Exit when the 'q' key is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        except Exception as e:
            print(f"Error in display loop: {e}")
            time.sleep(0.1)
    
    # Clean up
    cv2.destroyAllWindows()
    print("D415 RealSense Monitor Stopped")


if __name__ == "__main__":
    main()

