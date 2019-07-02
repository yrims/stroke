import os
import cv2 
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage.morphology import skeletonize

IMG_PATH = "data/SG"
SAVE_PATH = "result"

def binarization(img):
    thres = 128
    img[np.where(img <= thres)] = 1
    img[np.where(img > thres)] = 0
    return img

def de_binary(img):
    img[np.where(img > 0)] = 255
    return img

def load_data(img_path):
    img = Image.open(img_path)
    img = img.resize((256, 256), Image.BILINEAR)
    gray = img.convert('L')
    gray = np.array(gray)
    kernel_size = 3
    blur_gray = cv2.GaussianBlur(gray,(kernel_size,kernel_size),0)
    blur_gray = np.array(blur_gray)
    blur_gray = binarization(blur_gray)
    return blur_gray

def thin(img):
    skeleton = skeletonize(img).astype(np.uint8)
    return skeleton

def dil_ero(img):
    kernel = np.ones((5,5),np.uint8)
    dilation = cv2.dilate(img, kernel, iterations = 1)
    erosion = cv2.erode(dilation, kernel, iterations = 1)
    return erosion

def line_detect_possible_demo(image):
    
    lines = cv2.HoughLinesP(image, 1, np.pi / 180, 50, minLineLength=50, maxLineGap=5)
    
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
    #cv2.imshow("line_detect_possible_demo",image)
    return image
    
def make_dir(name):
    if not os.path.exists(SAVE_PATH + '/' + 'SG' + '/' + name):
        os.mkdir(SAVE_PATH + '/' + 'SG' + '/' + name)
   

for root, dirs, fs in os.walk(IMG_PATH):
    for f in fs:
        p = os.path.join(root, f)
        img = load_data(p)
        img = thin(img)
        img = dil_ero(img)
        img = thin(img)
        img = de_binary(img)
        save_name = 'SG_%s.jpg' % root[-1]
        print(save_name)
        make_dir(root[-1])
        cv2.imencode('.jpg', img)[1].tofile('result/SG' + '/' + root[-1] + '/' + save_name)

        