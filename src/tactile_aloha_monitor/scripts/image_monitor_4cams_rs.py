#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2, time
import numpy as np
from aloha_pro.msg import RGBGrayscaleImage

class ImageRecorder:
    def __init__(self, init_node=True, is_debug=False):
        from collections import deque
        import rospy
        from cv_bridge import CvBridge
        from sensor_msgs.msg import Image

        self.is_debug = is_debug
        self.bridge = CvBridge()

        # Camera configuration: 4 Logitech cameras + 1 RealSense camera
        self.logitech_cameras = [
            'cam_high',
            'cam_low', 
            'cam_left_wrist',
            'cam_right_wrist',
        ]
        self.realsense_cameras = ['cam_high_rs']
        self.camera_names = self.logitech_cameras + self.realsense_cameras

        if init_node:
            rospy.init_node('image_recorder_4cams_rs', anonymous=True)

        # Initialize Logitech cameras
        for cam_name in self.logitech_cameras:
            setattr(self, f'{cam_name}_image', None)
            setattr(self, f'{cam_name}_secs', None)
            setattr(self, f'{cam_name}_nsecs', None)

            # Select callback and topic based on camera name
            if cam_name == 'cam_high':
                callback_func = self.image_cb_cam_high
                topic_name = "/usb_cam_high/image_raw"
            elif cam_name == 'cam_low':
                callback_func = self.image_cb_cam_low
                topic_name = "/usb_cam_low/image_raw"
            elif cam_name == 'cam_left_wrist':
                callback_func = self.image_cb_cam_left_wrist
                topic_name = "/usb_cam_left_wrist/image_raw"
            elif cam_name == 'cam_right_wrist':
                callback_func = self.image_cb_cam_right_wrist
                topic_name = "/usb_cam_right_wrist/image_raw"
            else:
                raise NotImplementedError

            # Subscribe to Logitech camera topics
            rospy.Subscriber(topic_name, Image, callback_func)

            # If debug, store timestamps for frequency diagnostics
            if self.is_debug:
                setattr(self, f'{cam_name}_timestamps', deque(maxlen=50))

        # Initialize RealSense camera
        for cam_name in self.realsense_cameras:
            setattr(self, f"{cam_name}_rgb_image", None)
            setattr(self, f"{cam_name}_depth_image", None)
            setattr(self, f"{cam_name}_timestamp", 0.0)
            
            if cam_name == "cam_high_rs":
                callback_func = self.image_cb_cam_high_rs
            else:
                raise NotImplementedError
                
            # Subscribe to RealSense camera topic
            rospy.Subscriber(f"/{cam_name}", RGBGrayscaleImage, callback_func)
            
            if self.is_debug:
                setattr(self, f"{cam_name}_timestamps", deque(maxlen=50))

        self.cam_last_timestamps = {cam_name: 0.0 for cam_name in self.realsense_cameras}
        time.sleep(0.5)

    def logitech_image_cb(self, cam_name, data):
        """Callback for Logitech cameras using standard Image messages"""
        cv_image = self.bridge.imgmsg_to_cv2(data, desired_encoding='passthrough')
        
        # Convert color channels from BGR to RGB
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        
        setattr(self, f'{cam_name}_image', cv_image)
        setattr(self, f'{cam_name}_secs', data.header.stamp.secs)
        setattr(self, f'{cam_name}_nsecs', data.header.stamp.nsecs)
        
        if self.is_debug:
            ts_list = getattr(self, f'{cam_name}_timestamps')
            ts_list.append(data.header.stamp.secs + data.header.stamp.nsecs * 1e-9)

    def realsense_image_cb(self, cam_name, data):
        """Callback for RealSense camera using RGBGrayscaleImage messages"""
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
        
        if self.is_debug:
            getattr(self, f"{cam_name}_timestamps").append(
                data.images[0].header.stamp.secs
                + data.images[0].header.stamp.nsecs * 1e-9
            )

    # Logitech camera callbacks
    def image_cb_cam_high(self, data):
        return self.logitech_image_cb('cam_high', data)

    def image_cb_cam_low(self, data):
        return self.logitech_image_cb('cam_low', data)

    def image_cb_cam_left_wrist(self, data):
        return self.logitech_image_cb('cam_left_wrist', data)

    def image_cb_cam_right_wrist(self, data):
        return self.logitech_image_cb('cam_right_wrist', data)

    # RealSense camera callback
    def image_cb_cam_high_rs(self, data):
        return self.realsense_image_cb('cam_high_rs', data)

    def get_images(self):
        """Get images from all cameras"""
        image_dict = {}
        
        # Get Logitech camera images
        for cam_name in self.logitech_cameras:
            image_dict[cam_name] = getattr(self, f'{cam_name}_image')
        
        # Get RealSense camera images
        for cam_name in self.realsense_cameras:
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
        """Print camera frequency diagnostics"""
        def dt_helper(list_of_timestamps):
            arr = np.array(list_of_timestamps)
            diff = arr[1:] - arr[:-1]
            return np.mean(diff)

        print("=== Camera Frequency Diagnostics ===")
        for cam_name in self.camera_names:
            ts_list = getattr(self, f'{cam_name}_timestamps', [])
            if len(ts_list) > 1:
                image_freq = 1 / dt_helper(ts_list)
                print(f'{cam_name} frequency: {image_freq:.2f} Hz')
            else:
                print(f'{cam_name} no timestamps recorded yet.')
        print()


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
    # Initialize the image recorder for all cameras
    recorder = ImageRecorder(init_node=True)
    
    print("Multi-Camera Monitor Started (4 Logitech + 1 RealSense)")
    print("Press 'q' to quit")
    print("Layout: 2x3 Grid - 4 Logitech cameras + RealSense RGB+Depth")
    
    # Create window once before the loop (allows resizing)
    window_name = "Multi-Camera Monitor (2x3 Grid)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 960, 480)
    
    # Main display loop
    while not rospy.is_shutdown():
        try:
            # Get images from all cameras
            images = recorder.get_images()
            
            # Define target size for all images
            target_size = (320, 240)  # Size for each camera in 2x3 grid
            
            # Create blank image for missing signals
            blank_image = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
            
            # Process all camera images
            processed_images = {}
            
            # Process Logitech cameras
            for cam_name in recorder.logitech_cameras:
                img = images.get(cam_name)
                if img is not None:
                    resized_img = cv2.resize(img, target_size)
                    # Add camera name as title
                    cv2.putText(resized_img, cam_name, (10, 25), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    processed_images[cam_name] = resized_img
                else:
                    # Create blank image with "NO SIGNAL"
                    blank_img = blank_image.copy()
                    cv2.putText(blank_img, 'NO SIGNAL', (50, 120), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    processed_images[cam_name] = blank_img
            
            # Process RealSense camera
            rs_rgb = images.get('cam_high_rs')
            rs_depth = images.get('cam_high_rs_depth')
            
            if rs_rgb is not None:
                rs_rgb_resized = cv2.resize(rs_rgb, target_size)
                cv2.putText(rs_rgb_resized, 'D415 RGB', (10, 25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                processed_images['rs_rgb'] = rs_rgb_resized
            else:
                blank_rs_rgb = blank_image.copy()
                cv2.putText(blank_rs_rgb, 'NO RGB', (80, 120), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                processed_images['rs_rgb'] = blank_rs_rgb
            
            if rs_depth is not None:
                rs_depth_resized = cv2.resize(rs_depth, target_size)
                # Process depth image with colormap
                depth_colormap = process_depth_image(rs_depth_resized)
                cv2.putText(depth_colormap, 'D415 Depth', (10, 25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                processed_images['rs_depth'] = depth_colormap
            else:
                blank_rs_depth = blank_image.copy()
                cv2.putText(blank_rs_depth, 'NO DEPTH', (80, 120), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                processed_images['rs_depth'] = blank_rs_depth
            
            # Create 2x3 grid layout
            # Top row: cam_high, cam_low, D415 RGB
            top_row = np.hstack((
                processed_images['cam_high'], 
                processed_images['cam_low'], 
                processed_images['rs_rgb']
            ))
            
            # Bottom row: cam_left_wrist, cam_right_wrist, D415 Depth
            bottom_row = np.hstack((
                processed_images['cam_left_wrist'], 
                processed_images['cam_right_wrist'], 
                processed_images['rs_depth']
            ))
            
            # Combine into 2x3 grid
            final_output = np.vstack((top_row, bottom_row))
            
            # Display the combined image (window already created outside loop)
            cv2.imshow(window_name, final_output)
            
            # Exit when the 'q' key is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        except Exception as e:
            print(f"Error in display loop: {e}")
            time.sleep(0.1)
    
    # Clean up
    cv2.destroyAllWindows()
    print("Multi-Camera Monitor Stopped")


if __name__ == "__main__":
    main()
