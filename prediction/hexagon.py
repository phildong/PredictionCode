import numpy as np
from numpy.linalg import norm
import pickle

import matplotlib.pyplot as plt

a = np.array([[350, 0], [-175, 100]]).T

def hex_pts(center, micron, angle, max_n = 10):
    c, s = np.cos(angle), np.sin(angle)
    aa = micron*np.dot(np.array([[c, -s], [s, c]]), a)
    pts = []
    for i in range(-max_n, max_n + 1):
        for j in range(-max_n, max_n + 1):
            pts.append(center + np.dot(aa, np.array([i, j])))
    return np.array(pts)

def min_dist(x, pts):
    return min([norm(x-p) for p in pts])

def dist(p1, p2):
    return hypot(*(np.array(p1)-np.array(p2)))

# with open('circles.pkl', 'rb') as f:
#     circles = pickle.load(f)

# times = [c[0][0] for c in circles]
# start = circles[0][0][1:]

params = ([start for t in times], 0.5, 0)

def cost(centers, micron, angle):
    mismatch = 0
    for i in range(len(times)):
        hexa = hex_pts(centers[i], micron, angle)
        for j in range(len(circles[i])):
            mismatch += min_dist(circles[i][j][1:], hexa)**2
    
    vel = 0
    for i in range(len(times)-1):
        vel += dist(centers[i][1:], centers[j][1:])**2
    
    return mismatch + vel


    
