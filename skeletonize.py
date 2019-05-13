import os
import cv2
import math
from skimage.morphology import skeletonize
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

TYPE_1 = 'SG'
TYPE_2 = 'BK'
IMG_PATH = 'data'
SAVE_PATH = 'output'

class Data():
    
    def __init__(self, img_name, TYPE):
        self.img = None
        self.ori_img = None
        self.type = TYPE
        self.img_name = img_name
        self.skeleton = None
        self.table = None
        self.stroke_len = 0
        self.start_end = []


    def load_data(self):
        self.img = Image.open(IMG_PATH + '/' + self.type + '/' + self.img_name)
        self.img = self.img.resize((256, 256), Image.BILINEAR)
        self.ori_img = np.array(self.img)
        self.img = self.img.convert('L')
        self.img = np.array(self.img)
        thres = 128
        for i in range(256):
            for j in range(256):
                if self.img[i, j] > thres:
                    self.img[i, j] = 0
                else:
                    self.img[i, j] = 1

    def thin(self):
        # perform skeletonization
        self.skeleton = skeletonize(self.img)
    
    def split(self):

        # store each strock
        self.table = np.zeros((256, 256), dtype=np.int8)
        # split stroke order
        count = 1 
        surround = []
        # print(self.skeleton)
        for j in range(256):
            # print('i:',i)
            for i in range(256):
                # print('j:',j)
                if self.skeleton[i, j] == 1 and self.table[i, j] == 0:
                    
                    print('#############################')
                    print('     stroke order:', count)

                    # print('########################')
                    
                    # compute length in a stroke
                    # stroke_len = 1

                    self.table[i, j] = count
                        
                    tmp_i, tmp_j = i, j
                    
                    self.table[tmp_i, tmp_j] = count
                    
                    adjacent_idx = [(tmp_i + 1, tmp_j), (tmp_i + 1, tmp_j + 1), (tmp_i, tmp_j + 1), (tmp_i - 1, tmp_j + 1),
                                    (tmp_i - 1, tmp_j), (tmp_i - 1, tmp_j - 1), (tmp_i, tmp_j - 1), (tmp_i + 1, tmp_j - 1)]
                    
                    adjacent_pixel = [self.skeleton[i+1, j], self.skeleton[i+1, j+1], self.skeleton[i, j+1], self.skeleton[i-1, j+1],
                                      self.skeleton[i-1, j], self.skeleton[i-1, j-1], self.skeleton[i, j-1], self.skeleton[i+1, j-1]]

                    n_adjacent = np.count_nonzero(adjacent_pixel)
                    start_i = tmp_i
                    start_j = tmp_j
                    self.start_end.append((count, start_i, start_j))
                    print('start : ( %3d, %3d )' % (start_i, start_j))
                    # print('adj idx:', adjacent_idx)
                    
                    while(1):
                        # print('(%d, %d)' % (tmp_i, tmp_j))
                        FLAG = 1
                        for idx in adjacent_idx:
                            adj_i, adj_j = idx
                            # print('(%d, %d)' % (adj_i, adj_j))
                            if self.skeleton[adj_i, adj_j] == 1 and self.table[adj_i, adj_j] == 0:
                                FLAG = 0    
                                # stroke_len += 1
                                self.table[adj_i, adj_j] = count
                                tmp_i = adj_i
                                tmp_j = adj_j
                                
                                if tmp_i == 255 or tmp_j == 255 or tmp_i == 0 or tmp_j == 0:
                                    break    
                                
                                adjacent_idx = [(tmp_i + 1, tmp_j), (tmp_i + 1, tmp_j + 1), (tmp_i, tmp_j + 1), (tmp_i - 1, tmp_j + 1),
                                                (tmp_i - 1, tmp_j), (tmp_i - 1, tmp_j - 1), (tmp_i, tmp_j - 1), (tmp_i + 1, tmp_j - 1)]

                                adjacent_pixel = [self.skeleton[tmp_i+1, tmp_j], self.skeleton[tmp_i+1, tmp_j+1], self.skeleton[tmp_i, tmp_j+1], self.skeleton[tmp_i-1, tmp_j+1], 
                                                  self.skeleton[tmp_i-1, tmp_j], self.skeleton[tmp_i-1, tmp_j-1], self.skeleton[tmp_i, tmp_j-1], self.skeleton[tmp_i+1, tmp_j-1]]

                                computed_adjacent_pixel =  [self.table[tmp_i+1, tmp_j], self.table[tmp_i+1, tmp_j+1], self.table[tmp_i, tmp_j+1], self.table[tmp_i-1, tmp_j+1], 
                                                            self.table[tmp_i-1, tmp_j], self.table[tmp_i-1, tmp_j-1], self.table[tmp_i, tmp_j-1], self.table[tmp_i+1, tmp_j-1]] 

                                n_adjacent = np.count_nonzero(adjacent_pixel) - np.count_nonzero(computed_adjacent_pixel)
                                print('(%d, %d)' % (tmp_i, tmp_j), 'num of adjacent:', n_adjacent)
                                
                                if n_adjacent > 1:
                                    self.table[tmp_i, tmp_j] = count
                                    break
                                
                                continue
                        if FLAG:
                            break

                    # add mid pixel
                    end_i = tmp_i
                    end_j = tmp_j
                    mid_i = int((start_i + end_i) / 2)
                    mid_j = int((start_j + end_j) / 2)
                    self.start_end.append((count, mid_i, mid_j))
                   
                    # add end pixel
                    self.start_end.append((count, end_i, end_j))
                    
                    print('end   : ( %3d, %3d )' % (tmp_i, tmp_j))
                    print('#############################')
                    count += 1
        
        if not os.path.exists(SAVE_PATH + '/' + self.type + '/' + self.img_name[:-4]):
            os.mkdir(SAVE_PATH + '/' + self.type +  '/' + self.img_name[:-4])
        np.savetxt(SAVE_PATH + '/' + self.type + '/' + self.img_name[:-4] + '/' +  'start_end.txt', self.start_end, fmt='%d', delimiter=',')
        np.savetxt(SAVE_PATH + '/' + self.type + '/' + self.img_name[:-4] + '/' + 'table.txt', self.table, fmt='%d', delimiter='')
        np.savetxt(SAVE_PATH + '/' + self.type + '/' + self.img_name[:-4] + '/' + 'skeleton.txt', self.skeleton, fmt='%d', delimiter='')
        self.stroke_len = count - 1
       
    def match_stroke(self):
        
        # load start_end.txt of SG stroke    
        start_end_a = np.loadtxt(SAVE_PATH + '/' +  TYPE_1 + '/' + self.img_name[:-4] + '/' + 'start_end.txt', delimiter=',', dtype=np.int16)
        len_stroke = int(start_end_a.shape[0] / 3)
        match_table = np.chararray((len_stroke, 2), itemsize=4, unicode=True)
        # print('start_end_a:')
        # print(start_end_a)
        for len_a in range(len_stroke):
            match_table[len_a, 0] = len_a + 1 
            _, start_x_a, start_y_a = start_end_a[len_a]
            _, mid_x_a, mid_y_a     = start_end_a[len_a + 1]
            _, end_x_a, end_y_a     = start_end_a[len_a + 2]

            #print('start_x_a, start_y_a:', start_x_a, start_y_a)
            #print('mid_x_a, mid_y_a:', mid_x_a, mid_y_a)
            #print('end_x_a, end_y_a', end_x_a, end_y_a)
        
            min_distance = 9999999
            print('################################################')
            for i in range(5):
                
                # load start_end.txt of BK stroke
                start_end_b = np.loadtxt(SAVE_PATH + '/' +  TYPE_2 + '/%04d' % i + '/' + 'start_end.txt', delimiter=',', dtype=np.int16)
                _, start_x_b, start_y_b = start_end_b[0]
                _, mid_x_b, mid_y_b     = start_end_b[1]
                _, end_x_b, end_y_b     = start_end_b[2]
                
                print('Comparing SG: %s_%d with BK: %04d' % (self.img_name[:-4], len_a+1, i))
                #print('start_x_b, start_y_b:', start_x_b, start_y_b)
                #print('mid_x_b, mid_y_b:', mid_x_b, mid_y_b)
                #print('end_x_b, end_y_b', end_x_b, end_y_b)

                # compute two direction inner stroke
                # direction 1:
                dis_start = (start_x_a - start_x_b)**2 + (start_y_a - start_y_b)**2 
                dis_mid   = (mid_x_a - mid_x_b)**2 + (mid_y_a - mid_y_b)**2
                dis_end   = (end_x_a - end_x_b)**2 + (end_y_a - end_y_b)**2

                dis_1 = dis_start + dis_mid + dis_end
                
                # direction 2:
                dis_start = (start_x_a - end_x_b)**2 + (start_y_a - end_y_b)**2 
                dis_mid   = (mid_x_a - mid_x_b)**2 + (mid_y_a - mid_y_b)**2
                dis_end   = (end_x_a - start_x_b)**2 + (end_y_a - start_y_b)**2

                dis_2 = dis_start + dis_mid + dis_end
                
                # find the actual distance in correct direction
                dis = min(dis_1, dis_2)

                print('min dis:', min_distance, 'dis:', dis)
                
                if dis < min_distance:
                    min_distance = dis
                    match_table[len_a, 1] = '%04d' % i
            print('SG: %s_%d is matched to BK: %4s' % (self.img_name[:-4], len_a+1, match_table[len_a, 1]))        
            print('################################################')
        np.savetxt(SAVE_PATH + '/' +  TYPE_1 + '/' + '%s_match.txt' % self.img_name[:-4], match_table, fmt='%5s', delimiter=',')
        
        # a BK stroke 
        
        
        # print('start_end_b:')
        # print(start_end_b)
        # print('hi')        


    
    def save_stroke(self):    
        # store 
        
        print('stroke len:', self.stroke_len)
        for num_stroke in range(self.stroke_len):
            stroke_img = np.zeros((256, 256))
            for i in range(256):
                for j in range(256):
                    if self.table[i, j] == (num_stroke + 1):
                        stroke_img[i, j] = 255
            
            if not os.path.exists(SAVE_PATH + '/' + self.type + '/' + self.img_name[:-4]):
                os.mkdir(SAVE_PATH + '/' + self.type + '/' + self.img_name[:-4])
            
            # cv2.imwrite(SAVE_PATH + '/' + self.type + '/' + self.img_name[:-4] + '/' + '%s_%d.jpg' % (self.img_name[:-4], (num_stroke + 1)), stroke_img)
            cv2.imencode('.jpg', stroke_img)[1].tofile(SAVE_PATH + '/' + self.type + '/' + self.img_name[:-4] + '/' + '%s_%d.jpg' % (self.img_name[:-4], (num_stroke + 1)))

        
            
            

        
        


if __name__ == '__main__':
    '''
    # getting SG stroke order
    for img_name in os.listdir(IMG_PATH + '/' + TYPE_1):
        print('img name:', img_name)
        
        data = Data(img_name, TYPE_1)
        data.load_data()
        data.thin()
        data.split()
        data.save_stroke()
    '''
    '''
    # getting BK stroke order
    for img_name in os.listdir(IMG_PATH + '/' + TYPE_1):
        print('img name:', img_name)
        
        data = Data(img_name, TYPE_1)
        data.load_data()
        data.thin()
        data.split()
        data.save_stroke()
    '''   
        
    # matching SG stroke order to BK stroke
    for img_name in os.listdir(IMG_PATH + '/' + TYPE_1):
        print('img name:', img_name)
        
        data = Data(img_name, TYPE_1)
        data.match_stroke()