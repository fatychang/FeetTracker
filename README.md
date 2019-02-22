# FeetTracker
This project aims to develop an python real-time feet tracking function. In addition, it support playback recorded .bag file.
Point cloud pre-processing are done. Next, it should target the feet segmentation and identification.

## FeetTracking.py
The main script to perform the feet tracking algorithm. Currently, it only supports playback.
Point cloud pre-processing includes downsampling, image cropping, ground segmentation, and outlier removal.
The next step should cluster the left and right foot via k-mean or other unsupervised clustering method.

k-Mean is unacceptable since it will classify the legs wrongly to up and down instead of left and right. Therefore, I am trying to use directly segment the leg with cylinder model via Hough Transform.

## PointCloudViewer.py
It is extracted and from the realsense python sample code [Here](https://github.com/IntelRealSense/librealsense/blob/master/wrappers/python/examples/opencv_pointcloud_viewer.py).It has been slightly modified in order to call from other script. Import the pointcloudViewer in the main script and add the following **mouse_cb** part for easy registration in the main script.
```
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
```
## myMath.py
This script handles some self-defined functions related to simple math. The following are the callable functions:

- **forced_project_to_2Dimage()**: From the point cloud datapoint (which belongs to dataPoints * 3) to obtain a 2D image by using the estimated image width and height.

- **plot_points()**: A simple visualization method to plot the 2D image (x and y) converted from point cloud

