# FeetTracker
This project aims to develop an python real-time feet tracking function. In addition, it support playback recorded .bag file.
Point cloud pre-processing are done. Next, it should target the feet segmentation and identification.



## PointCloudViewer.py
It is extracted and from the realsense python sample code [Here] (https://github.com/IntelRealSense/librealsense/blob/master/wrappers/python/examples/opencv_pointcloud_viewer.py).It has been slightly modified in order to call from other script. Import the pointcloudViewer in the main script and add the following mouse_cb part so as to register it later.

