# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 13:14:34 2019

@author: jschang
"""

import math



'''
Estimate the width and height from the pointcloud datapoint
'''
def findwidthHeight (verts):
    
    # Get the max and min value in the point cloud along x, y, and z axis
    boundary_max = verts.max(axis=0)
    boundary_min = verts.min(axis=0)
    
    # Calculate the width and height of the 2D image
    diff = boundary_max - boundary_min
    
    # Calculate the ratio of x/y (width/height)
    ratio = diff[0]/diff[1]
    
    # find the approximated width of the image (RX^2 = numofpoints)
    x_square = verts.shape[0]/ratio
    x = math.sqrt(x_square)
    
    # find y via y = ratio * x
    y = ratio * x
    
    # Round the values and return (width, height)
    w = round(x, 0)
    h = round(y, 0)
    return (w, h)
    


