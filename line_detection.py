import os
import cv2 
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage.morphology import skeletonize


IMG_PATH = "data/SG"

def binarization(img):
    thres = 128
    for i in range(256):
        for j in range(256):
            if img[i, j] > thres:
                img[i, j] = 0
            else:
                img[i, j] = 1
    return img

for root, dirs, fs in os.walk(IMG_PATH):
    for f in fs:
        p = os.path.join(root, f)
        img = Image.open(p)
        img = img.resize((256, 256), Image.BILINEAR)

        gray = img.convert('L')
        gray = np.array(gray)
        kernel_size = 5
        blur_gray = cv2.GaussianBlur(gray,(kernel_size,kernel_size),0)
        # low_threshold = 50
        # high_threshold = 150
        # masked_edges = cv2.Canny(blur_gray,low_threshold,high_threshold)
        blur_gray = np.array(blur_gray)
        blur_gray = binarization(blur_gray)
        skeleton = skeletonize(blur_gray).astype(np.uint8)

        #plt.figure()
        #plt.imshow(skeleton)
        #plt.show()

        rho = 1
        theta = np.pi/180
        threshold = 1
        min_line_length = 2
        max_line_gap = 1
        line_image = np.copy(img)*0 #creating a blank to draw lines on

        lines = cv2.HoughLinesP(skeleton,rho,theta,threshold,np.array([]),
                                min_line_length,max_line_gap)

        lines = np.squeeze(lines)

        for x1,y1,x2,y2 in lines:
            cv2.line(line_image,(x1,y1),(x2,y2),(0,0,255),1)

        color_edges = np.dstack((skeleton,skeleton,skeleton))
        combo = cv2.addWeighted(color_edges, 0.8, line_image, 1, 0) 
        save_name = 'Hough_%s.jpg' % root[-1]
        cv2.imencode('.jpg', combo)[1].tofile('result/SG' + '/' + root[-1] + '/' + save_name)
        #plt.axis('off')
        #plt.imshow(combo)
        #plt.imsave('result/SG/')
        #plt.show()

