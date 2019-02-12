# -*- coding: utf-8 -*-
"""
Created on Thu Jan 24 11:46:45 2019

This script aims to implement image clustering, segmentation technique
to achieve the feetTracking function

@author: jschang
"""

# First import library
import pyrealsense2 as rs
# Import Numpy for easy array manipulation
import numpy as np
# Import OpenCV for easy image rendering
import cv2
# Import PCL for filtering
import pcl
# Import time packages for pointcloud rendering
import time
# Import pointcloud viewer.py
import pointcloudViewer
# Import open3d for visualization
import open3d

import math




#############################
# Point Cloud Visualzation  #
#    Setting                #
#############################

# Create AppState object for pointcloud visualization
state = pointcloudViewer.AppState()    

# Define mouse motion event
def mouse_cb(event, x, y, flags, param):

    if event == cv2.EVENT_LBUTTONDOWN:
        state.mouse_btns[0] = True

    if event == cv2.EVENT_LBUTTONUP:
        state.mouse_btns[0] = False

    if event == cv2.EVENT_RBUTTONDOWN:
        state.mouse_btns[1] = True

    if event == cv2.EVENT_RBUTTONUP:
        state.mouse_btns[1] = False

    if event == cv2.EVENT_MBUTTONDOWN:
        state.mouse_btns[2] = True

    if event == cv2.EVENT_MBUTTONUP:
        state.mouse_btns[2] = False

    if event == cv2.EVENT_MOUSEMOVE:

        h, w = out.shape[:2]
        dx, dy = x - state.prev_mouse[0], y - state.prev_mouse[1]

        if state.mouse_btns[0]:
            state.yaw += float(dx) / w * 2
            state.pitch -= float(dy) / h * 2

        elif state.mouse_btns[1]:
            dp = np.array((dx / w, dy / h, 0), dtype=np.float32)
            state.translation -= np.dot(state.rotation, dp)

        elif state.mouse_btns[2]:
            dz = math.sqrt(dx**2 + dy**2) * math.copysign(0.01, -dy)
            state.translation[2] += dz
            state.distance -= dz

    if event == cv2.EVENT_MOUSEWHEEL:
        dz = math.copysign(0.1, flags)
        state.translation[2] += dz
        state.distance -= dz

    state.prev_mouse = (x, y)




####################################################
##           Initialization -                     ##      
##  Read the video and make ready the pipeline    ##
###################################################

# Flag setting
SAVE_IMAGE = False
IS_DEBUG = False



# .bag file location
FILE_LOC = "D:\\Jen\\Projects\\RealSense Camera\\Recordings\\feetTest1.bag"
#FILE_LOC = "D:\\Jen\\Projects\\RealSense Camera\\Recordings\\d415_1500.bag"


# Create the context object that holds the handle of all the connected devices
pipeline = rs.pipeline()


# Create the config object and configure the stream
cfg = rs.config()
# Tell config that we will use a recorded device from filem to be used by the pipeline through playback.
rs.config.enable_device_from_file(cfg, FILE_LOC)
# Configure the pipeline to stream the depth stream (resolution, format, and frame rate)
cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

# Start streaming from file and obtain the returned profile
profile = pipeline.start(cfg)


# Obtain the depth stream profile and camera intrinsics
depth_profile = rs.video_stream_profile(profile.get_stream(rs.stream.depth))
depth_intrinsics = depth_profile.get_intrinsics()

# Get the width and height of the frame from the camera intrinsics
w, h = depth_intrinsics.width, depth_intrinsics.height

# Create pointcloud object
pc = rs.pointcloud()

# Decimate the point cloud
decimate = rs.decimation_filter()
decimate.set_option(rs.option.filter_magnitude, 2 ** state.decimate)

# Create realsense colorizer object
colorizer = rs.colorizer()


# Initialize the opencv window
cv2.namedWindow(state.WIN_NAME, cv2.WINDOW_AUTOSIZE)
cv2.resizeWindow(state.WIN_NAME, w, h)
cv2.setMouseCallback(state.WIN_NAME, mouse_cb)

# Create param 'out' to store the frame data
out = np.empty((h, w, 3), dtype=np.uint8)





#######################
#    Stream loop      #
#######################


while True:
    # Grab camera data
    if not state.paused:
        # Get the frames from pipeline
        frames = pipeline.wait_for_frames()
        
        # Get depth frame
        depth_frame = frames.get_depth_frame()
        
        depth_frame = decimate.process(depth_frame)
        
        # Grab new intrinsics (may be changed by decimation)
        depth_intrinsics = rs.video_stream_profile(depth_frame.profile).get_intrinsics()
        w, h = depth_intrinsics.width, depth_intrinsics.height
        
        
        # Get the depth frame data
        depth_image = np.asanyarray(depth_frame.get_data())
        
        
        # Get the depth color map
        depth_colormap = np.asanyarray(colorizer.colorize(depth_frame).get_data())
        
        # Obtain the mapped frame and color_source
        mapped_frame, color_source = depth_frame, depth_colormap    
        
        # Calculate the points from the depth_frame
        points = pc.calculate(depth_frame)
        pc.map_to(mapped_frame)
        
        # Pointcloud data to numpy array
        v,t = points.get_vertices(), points.get_texture_coordinates()
        verts = np.asarray(v).view(np.float32).reshape(-1,3) #xyz
        texcoords = np.asarray(t).view(np.float32).reshape(-1,2) #uv
        
        
        
    
        
        ######################################
        #   Downsample the point cloud       #
        #     with Voxel Grid Filter         #
        ######################################
        
        # Create a PCL pointcloud
        oriCloud = pcl.PointCloud(verts)
    
     
        # Starting time for downsampling
        now = time.time()
        
        # Create the Voxel Grid Filter Object
        vox = oriCloud.make_voxel_grid_filter()
        # Choose the voxel (leaf) size
        LEAF_SIZE = 0.005
        # Set the voxel size on the vox object
        vox.set_leaf_size(LEAF_SIZE, LEAF_SIZE, LEAF_SIZE)
        
        # Call the voxel_grid_filter to obtain the downsampled cloud, called VGCloud (voxel_grid cloud)
        vgCloud = vox.filter()  
        
        
        # Downsampling time
        dt = time.time() - now   
        
                    
        # Debug
        if IS_DEBUG:
            # Print the size of the original point cloud
            print("The size of the original pointcloud: ", oriCloud.size) 
            # Print process time
            print("Downsampleing time: %10.9f", dt)
            # Print the size of the downsampled point cloud
            print("The size of the downsampled pointcloud: ", vgCloud.size)  
        
        
        # Save the image for visualization
        if SAVE_IMAGE:
            pcl.save(oriCloud, "oriCloud.pcd")
            pcl.save(vgCloud, "vgCloud.pcd")    
        
        
        
        
        ###############################
        #   Apply Passthrough Filter   #
        #    (crop the image)         #
        ###############################
        
        # Starting time for downsampling
        now1 = time.time()
        
        
        # Create a passthrough filter object
        passthrough = vgCloud.make_passthrough_filter()
            
        # Assign axis and range to the passthrough filter()
        FILTER_AXIS = 'z'
        passthrough.set_filter_field_name(FILTER_AXIS)
        AXIS_MIN = 0
        AXIS_MAX = 0.6
        passthrough.set_filter_limits(AXIS_MIN, AXIS_MAX)
            
        # Call the passthrough filter to obtain the resultant pointcloud
        ptCloud = passthrough.filter()
         
        # Downsampling time
        dt1 = time.time() - now1 
    
    
        # Debug
        if IS_DEBUG:
            # Print the size of the cropped point cloud
            print("The size of the cropped pointcloud: ", ptCloud.size)
            # Print process time
            print("Cropping time: %10.9f", dt1)
    
            
            
            
        # Save the image for visualization
        if SAVE_IMAGE:
            pcl.save(ptCloud, "passthroughCloud.pcd")
        
        
        
        
        ###################################
        #   Ground Segmentation (remove)  #
        #          via RANSAC             #
        ###################################
        
        # Starting time for ground segmentation
        now2 = time.time()
        
        # Create the segmentation object
        seg = ptCloud.make_segmenter()
    
        # Set the model you wish to fit
        seg.set_model_type(pcl.SACMODEL_PLANE)
        #seg.set_model_type(pcl.SAC_RANSAC)
                           
        # Max distance for the point to be consider fitting this model
        max_distance = 0.015
        seg.set_distance_threshold(max_distance)
    
        # Obtain a set of inlier indices ( who fit the plane) and model coefficients
        inliers, coefficients = seg.segment()
        
        # Extract Inliers obtained from previous step
        gdRemovedCloud = ptCloud.extract(inliers, negative=True)
    
        # Ground segmentation time
        dt2 = time.time() - now2
        
       
        
        # Debug
        if IS_DEBUG:
            # Print the size of the ground removed point cloud
            print("The size of the ground removed pointcloud: ", gdRemovedCloud.size)
            # Print process time
            print("Ground segmentation time: %10.9f", dt2)
           
        
        
        # Save the image for visualization
        if SAVE_IMAGE:
            pcl.save(gdRemovedCloud, "gdRemovedCloud.pcd")
        
        
        
        
        #################################
        #   Outlier removal Filter-     #
        #  Statistical Outlier Removal  #
        #################################
        
        # Starting time for outlier removal 
        now3 = time.time()
        
        
        # Create a statistical outlier filter object
        outlier = gdRemovedCloud.make_statistical_outlier_filter()
    
        # Set the number of neighboring points to analyze for any given point
        outlier.set_mean_k(100)
        
        # Set threshold scale factor
        outlier_threshold = 0.05
    
        # Eliminate the points whose mean distance is larger than global
        # (global dis = mean_dis + threshold * std_dev)               
        outlier.set_std_dev_mul_thresh(outlier_threshold)
    
        # Apply the statistical outlier removal filter
        olRemovedCloud = outlier.filter()
    
        # Outlier removal time
        dt3 = time.time() - now3
        
        
        # Convert the pcl pointcloud object to array type
        displayCloud = olRemovedCloud
        display_verts = np.asarray(displayCloud).view(np.float32).reshape(-1,3) #xyz
        verts = display_verts    
        
    
    
        # Debug
        if IS_DEBUG:
            # Print the size of the ground removed point cloud
            print("The size of the outlier removed pointcloud: ", olRemovedCloud.size)   
            # Print process time
            print("Outlier removal time: %10.9f", dt3)
    
    
        # Save the image for visualization
        if SAVE_IMAGE:
            pcl.save(olRemovedCloud, "outlierRemovedCloud.pcd")
            
    ##############################
    # Point Cloud Visuzalization #
    ##############################
    
    # Render
    now = time.time()

    out.fill(0)

    pointcloudViewer.grid(state, out, (0, 0.5, 1), size=1, n=10)
    pointcloudViewer.frustum(state, out, depth_intrinsics)
    pointcloudViewer.axes(out, pointcloudViewer.view(state, [0, 0, 0]), state.rotation, size=0.1, thickness=1)

    if not state.scale or out.shape[:2] == (h, w):
        pointcloudViewer.pointcloud(state, out, verts, texcoords, color_source)
    else:
        tmp = np.zeros((h, w, 3), dtype=np.uint8)
        pointcloudViewer.pointcloud(state, tmp, verts, texcoords, color_source)
        tmp = cv2.resize(
            tmp, out.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)
        np.putmask(out, tmp > 0, tmp)

    if any(state.mouse_btns):
        pointcloudViewer.axes(out, pointcloudViewer.view(state, state.pivot), state.rotation, thickness=4)

    dt = time.time() - now

    cv2.setWindowTitle(
        state.WIN_NAME, "RealSense (%dx%d) %dFPS (%.2fms) %s" %
        (w, h, 1.0/dt, dt*1000, "PAUSED" if state.paused else ""))

    cv2.imshow(state.WIN_NAME, out)
    key = cv2.waitKey(1)

    if key == ord("r"):
        state.reset()

    if key == ord("p"):
        state.paused ^= True

    if key == ord("d"):
        state.decimate = (state.decimate + 1) % 3
        decimate.set_option(rs.option.filter_magnitude, 2 ** state.decimate)

    if key == ord("z"):
        state.scale ^= True

    if key == ord("c"):
        state.color ^= True

    if key == ord("s"):
        cv2.imwrite('./out.png', out)

    if key == ord("e"):
        points.export_to_ply('./out.ply', mapped_frame)

    if key in (27, ord("q")) or cv2.getWindowProperty(state.WIN_NAME, cv2.WND_PROP_AUTOSIZE) < 0:
        break   
        
        
        
        
        
        
# Stop streaming
pipeline.stop()
