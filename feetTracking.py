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
# Import k-mean classifier from scikit learn
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.cluster import SpectralClustering
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering

from matplotlib import pyplot as plt


import math
import myMath




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







#####################
##      FloodFill  ##
#####################
def floodfillClassifier (verts, loDiff, upDiff):
    
    # Convert the pointcloud data into 2D image with approximated width and height
    w, h, image_array = myMath.forced_project_to_2Dimage(verts)   
        
    # Reshape the datapoint and obtain the number of datapoints
    dataPoints = image_array.reshape(-1, 3)
    dataNo = image_array.shape[0] * image_array.shape[1]


    # Mark the points in the image that are chosen as points in the clusters
    ptsToBeClassified = (dataPoints[:, 2] / 4 * 1000).reshape(int(h), int(w)) # Convert the depth to a larger scale
    ptsToBeClassified_copy = (dataPoints[:, 2] / 4 * 1000).reshape(int(w), int(h))
    
    # Mask with the size of w+2, h+2 of the target image
    mask = np.zeros([ptsToBeClassified.shape[0] + 2, ptsToBeClassified.shape[1]+2], np.uint8)
   
    # Manually assign the seedPoints (Did not use anymore, change to randomly assign seedpoints)
#    seedPoint = tuple(image_array[10, 10, 0:2])
#    
#    seedPoints = [tuple(image_array[w/4, h/4, 0:2]), tuple(image_array[w/2, h/4, 0:2]), tuple(image_array[3 * w/4, h/4, 0:2]),
#                tuple(image_array[w/4, h/2, 0:2]), tuple(image_array[w/2, h/2, 0:2]), tuple(image_array[3 * w/4, h/2, 0:2]),
#                tuple(image_array[w/4, 3 * h/4, 0:2]), tuple(image_array[w/2, 3* h/4, 0:2]), tuple(image_array[3* w/4, 3* h/4, 0:2])]
#    seedPoint = tuple(seedPoints)

    
    # Randomly select 1% of total datapoints as seedPoints
    seedPointsNo = int(dataNo/100)
    seedPoints = np.zeros([seedPointsNo, 2]) # store the x and y index number
    seeds = np.zeros([seedPointsNo, 2]) # store the acutal x and y value of the seedpoints
    for i in range(0, seedPointsNo):

        # Here is a tricky way to avoid the dimension overfit problem which I haven't solve.
        # (If i select the seedpoints based on the max number of rows and column individaully,
        # later in the cv.floodfill will incounter the problem suggesting that the seeds are outside the image)
        if h > w:
            random_index1 = np.random.randint(1, w-1) # rows
            random_index2 = np.random.randint(1, w-1) # columns
        else:
            random_index1 = np.random.randint(1, h-1) # rows
            random_index2 = np.random.randint(1, h-1) # columns            
            

        seedPoints[i, :] = [random_index1, random_index2]
        seeds[i, :] = dataPoints[random_index2 + random_index1 * w, 0:2]
    

    

    
    # Debug Plot
    # Plot the projected pointcloud 2D image
    myMath.plot_points(dataPoints)
    # Plot the seedPoints
    plt.plot(seeds[:, 0], seeds[:, 1], 'r+')
    
  
    
    # Prepare to run floodfill from all the selected seedpoints
    skippedNo=0         # the number of seedpoints that are being skipped
    groupedNo = 0       # the number of groups that is clustered (ideally, should be two)     
    retval_array = np.zeros(seedPointsNo)   # store the number of points in each cluster    
    
    for i in range(seedPointsNo):
        
        # Extract the seedPoint from the seedPoints array
        seedPoint = int(seedPoints[i,0]), int(seedPoints[i, 1])
        # Clear the retval in case the selected seedPoint is skipped
        retval = 0
              
        #print ptsToBeClassified[seedPoint[0], seedPoint[1]]        # the depth value of the seedpoint itself
        
        # Skip the seedpoints that already been clustered        
        if i > 0: # afer the first iteration, skip the seedpoint if it's already assigned to a group           
            if ptsToBeClassified[seedPoint[0], seedPoint[1]] > seedPointsNo: # Meaning that the seedpoint haven't been clustered yet

                newVal = tuple([groupedNo+1])
                retval, image, mask, rect = cv2.floodFill(ptsToBeClassified, mask, seedPoint, newVal=newVal, loDiff=loDiff, upDiff=upDiff)
                #print(i, retval)
                
            else: # Skip the seedpoint when it is already been clustered
                skippedNo = skippedNo + 1
        else: # Run the first floodfill with the first seedpoint
            retval, image, mask, rect = cv2.floodFill(ptsToBeClassified, mask, seedPoint, newVal=1, loDiff=loDiff, upDiff=upDiff)
            #print(i, retval)
        
        # Calculate the number of groups (what if the first seedpoint failed to group?)
        if retval != 0:
            groupedNo = groupedNo + 1
        
        # Break the loop when all the datapoints are grouped
        retval_array[i] = retval
        if sum(retval_array) == dataNo :
            break
     ## End of running floodfill ##   
        
            
                
    # Display the floodfill result
    if groupedNo == 0:
        print('Did not find any group')
    elif groupedNo == 1:
        print('Only one group is found')
    elif groupedNo == 2:
        print('Should found both legs!')
    else:
        print('More than two groups are found')




#####################
##  Floodfill Test  #
#####################













####################################################
##           Initialization -                     ##      
##  Read the video and make ready the pipeline    ##
###################################################

# Flag setting
SAVE_IMAGE = False
IS_DEBUG = False



# .bag file location
#FILE_LOC = "D:\\Jen\\Projects\\RealSense Camera\\Recordings\\feetTest1.bag"
FILE_LOC = "D:\\Jen\\Projects\\RealSense Camera\\Recordings\\d415_1500.bag"


# Create the context object that holds the handle of all the connected devices
pipeline = rs.pipeline()


# Create the config object and configure the stream
cfg = rs.config()
# Tell config that we will use a recorded device from filem to be used by the pipeline through playback.
rs.config.enable_device_from_file(cfg, FILE_LOC)
# Configure the pipeline to stream the depth stream (resolution, format, and frame rate)
#cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
cfg.enable_stream(rs.stream.depth, 640, 360, rs.format.z16, 60)

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
    
    # Render
    now = time.time()
    
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
        
#        # 2D image display
#        points2d_x = verts[:, 0]
#        points2d_y = verts[:, 1]
#        plt.plot(points2d_x.tolist(), points2d_y.tolist(), 'o')

    
        
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
        LEAF_SIZE = 0.005  # unit is in [meter] (also can set 0.043)
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
        AXIS_MAX = 0.9
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
        seg.set_method_type(pcl.SAC_RANSAC)
                           
        # Max distance for the point to be consider fitting this model
        max_distance = 0.010
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
        outlier.set_mean_k(10)
        
        # Set threshold scale factor
        outlier_threshold = 0.01
    
        # Eliminate the points whose mean distance is larger than global
        # (global dis = mean_dis + threshold * std_dev)               
        outlier.set_std_dev_mul_thresh(outlier_threshold)
    
        # Apply the statistical outlier removal filter
        olRemovedCloud = outlier.filter()
    
        # Outlier removal time
        dt3 = time.time() - now3
        
        
        # Debug
        if IS_DEBUG:
            # Print the size of the ground removed point cloud
            print("The size of the outlier removed pointcloud: ", olRemovedCloud.size)   
            # Print process time
            print("Outlier removal time: %10.9f", dt3)
    
    
        # Save the image for visualization
        if SAVE_IMAGE:
            pcl.save(olRemovedCloud, "outlierRemovedCloud.pcd")
        
        
        

        
        
        
        
        ###################################
        #    Segment left/right leg   #
        ###################################
        # Hough Transform
        # cylinders = cv2.HoughCylinders()
        
        
        # --> unable to segment via RANSAC (maybe because there are two cylinders)
#        # Create the segmentation object
#        seg_leg = olRemovedCloud.make_segmenter()
#    
#        # Set the model you wish to fit
#        seg_leg.set_model_type(pcl.SACMODEL_CYLINDER)
#        seg_leg.set_method_type(pcl.SAC_RANSAC)
#        
#        # Set the threshold value
#        max_distance = 100
#        seg_leg.set_distance_threshold(max_distance)
#        
#        # Obtain a set of inlier indices ( who fit the plane) and model coefficients
#        inliers_leg, coefficients = seg_leg.segment()
#        
#        # Extract Inliers obtained from previous step
#        LegRemovedCloud = ptCloud.extract(inliers_leg, negative=False)
  

        ##################################
        #     Conver the pcl pointcloud  #
        #    object to numpy array type  #
        ##################################               
        
        # Convert the pcl pointcloud object to array type
        displayCloud = olRemovedCloud
        #displayCloud = LegRemovedCloud
        display_verts = np.asarray(displayCloud).view(np.float32).reshape(-1,3) #xyz
        verts = display_verts
        
        # Extimate the image width and height
        w, h, image_2d = myMath.forced_project_to_2Dimage(verts)
        
        
#        # 2D image display
#        points2d_x = verts[:, 0]
#        points2d_y = verts[:, 1]
#        estimatedImage = verts[0: int(w * h), :]
#        points2d_x = estimatedImage[:, 0]
#        points2d_y = estimatedImage[:, 1]
#        datapoint = estimatedImage.reshape(int(w), int(h), 3)
#        #plt.plot(points2d_x.tolist(), points2d_y.tolist(), 'o')


      
        
        ############################
        ## Clustering Testing     #
        ###########################

#        # Ward hierachical clustering method
#        ward = AgglomerativeClustering(n_clusters=2, linkage='ward')
#        ward.fit(verts)
#        label = ward.labels_
        
        
        # Floodfill from opencv
        #floodfillClassifier()
        

        
#        # Spectral Cluster -> Too slow
#        spectral_cluster = SpectralClustering(n_clusters = 2)
#        spectral_cluster.fit(verts)
#        label = spectral_cluster.labels_
        
        
        # DBSCAN --> unacceptable and too slow
#        db = DBSCAN()
#        db.fit(verts)
#        label = db.labels_
        

        
        
        
        #########################
        #  K-Mean Clustering   #
        ########################
#        # Implement k-mean for classification --> not accurate enough
#        init_idx = [verts.shape[0] / 2 - 150, verts.shape[0] / 2 + 150]
##        init_idx = [10  , verts.shape[0] - 50]
#        init_idx = np.array(init_idx)
#        init = np.array([verts[init_idx[0], :], verts[init_idx[1], :]])
#        k_means = KMeans(n_clusters=2, max_iter=10000, random_state=0, init = init)
#        k_means.fit(verts)
#        label = k_means.labels_
        
#        # Implement minibatch k-means --> same as k-mean
#        mbk = MiniBatchKMeans(n_clusters = 2)
#        mbk.fit(verts)
#        label = mbk.labels_
#        centers = mbk.cluster_centers_
        
        
        
        
        
        ##############################
        # Point Cloud Visuzalization #
        ##############################
        
        
#    # Draw k-mean initial guess point
#    #    p0 = tuple([verts[init_idx[0], :]])
#    #    p1 = tuple([verts[init_idx[1], :]])
#    p0 = verts[init_idx[0], :]
#    p1 = verts[init_idx[1], :]
#    #color = (0, 0, 255)
#    pointcloudViewer.line3d_exa(out, p0, p1,  thickness=50)
        
        
        
        
    # Remove the points which below to group 1 to visualize the clustering result
#    for idx, val in enumerate (label):
#        if val == 1:
#            verts[idx, :] = np.zeros(3)

    
    

            

    
    

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
