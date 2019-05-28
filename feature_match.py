"""
opencv-python-3.4.2.16
opencv-contrib-python-3.4.2.16
"""
import numpy as np
import cv2  as cv
import matplotlib.pyplot as plt
from skimage import io

# img1 = cv.imdecode(np.fromfile('data/0.jpg', dtype=np.uint8), -1)
# img2 = cv.imdecode(np.fromfile('data/5.png', dtype=np.uint8), -1)

# img2 = cv.imdecode(np.fromfile('data/0.jpg', dtype=np.uint8), cv.IMREAD_GRAYSCALE)

img1 = cv.imread('data/0.jpg')          # queryImage
#img2 = cv.imread('data/5.png')    # trainImage
img2 = io.imread('data/5.png')

#cv.imshow('123', img1)
io.imshow(img2)
#cv.waitKey(0)
#cv.destroyAllWindows()

# Initiate SIFT detector
sift = cv.xfeatures2d.SIFT_create()
# sift = cv.xfeatures2d.SURF_create(400)
# sift = cv.HOGDescriptor()
# orb = cv.ORB_create()
# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

# find the keypoints and descriptors with ORB
#kp1, des1 = orb.detectAndCompute(img1,None)
#kp2, des2 = orb.detectAndCompute(img2,None)

# BFMatcher with default params
bf = cv.BFMatcher()
matches = bf.knnMatch(des1,des2,k=2)
# Apply ratio test
good = []
for m,n in matches:
    if m.distance < 0.8*n.distance:
        good.append([m])
# cv.drawMatchesKnn expects list of lists as matches.
img3 = cv.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
plt.imshow(img3),plt.show()
