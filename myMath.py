# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 13:14:34 2019

@author: jschang
"""

import math



'''
Estimate the width and height of the corresponding 2D image
from the pointcloud datapoints

The pointcloud_array argument should be a numpy array with size of [num_of_point, 3]
Return the width(w), height(h), and the image_array itself
'''
def forced_project_to_2Dimage (pointcloud_array):
    
    # Get the max and min value in the point cloud along x, y, and z axis
    boundary_max = pointcloud_array.max(axis=0) # return the max of [x, y, z] in all datapoints
    boundary_min = pointcloud_array.min(axis=0) # return the min of [x, y, z] in all datapoints
    
    # Calculate the width and height of the 2D image
    diff = boundary_max - boundary_min # return the legth in all three dimensions [x, y, z]
    
    # Calculate the ratio of x/y (width/height)
    ratio = diff[0]/diff[1]
    
    # find the approximated width of the image (RX^2 = numofpoints)
    x_square = pointcloud_array.shape[0]/ratio
    x = math.sqrt(x_square)
    
    # find y via y = ratio * x
    y = ratio * x
    
    # Round the values to int and return (width, height)
    w = int(x)
    h = int(y)
    image = pointcloud_array.reshape(int(w), int(h), 3)
    return (w, h, image)
    


