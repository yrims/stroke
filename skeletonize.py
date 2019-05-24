import os
import cv2
import math
import shutil
from skimage.morphology import skeletonize
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

TYPE_1 = 'SG'
TYPE_2 = 'BK'
IMG_PATH = 'data'
SAVE_PATH = 'result'

class Data():
    
    def __init__(self, img_path, img_name, character, TYPE):
        self.img = None
        self.ori_img = None
        self.character = character
        self.type = TYPE
        self.img_path = img_path
        self.img_name = img_name
        self.skeleton = None
        self.table = None
        self.stroke_len = 0
        self.start_end = []


    def load_data(self):
        self.img = Image.open(self.img_path)
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
    
    def make_dir(self):
        if os.path.exists(SAVE_PATH + '/' + self.type + '/' + self.character):
            shutil.rmtree(SAVE_PATH + '/' + self.type + '/' + self.character)
            while not os.path.exists(SAVE_PATH + '/' + self.type + '/' + self.character):
                os.mkdir(SAVE_PATH + '/' + self.type + '/' + self.character)
        else:
            os.mkdir(SAVE_PATH + '/' + self.type + '/' + self.character)
    
    def split(self):

        # store each stroke
        self.table = np.zeros((256, 256), dtype=np.int8)
        # split stroke order
        count = 1 
        order = 1
        surround = []
        
        # print(self.skeleton)
        for i in range(256):
            # print('i:',i)
            for j in range(256):
                # print('j:',j)
                if self.skeleton[i, j] == 1 and self.table[i, j] == 0:
                    
                    print('#############################')
                    print('     stroke order:', count)

                    # print('########################')
                    
                    # compute length in a stroke
                    stroke_len = 1
                    
                    point_idxs = []

                    self.table[i, j] = count

                    tmp_i, tmp_j = i, j
                    
                    self.table[tmp_i, tmp_j] = count
                    
                    adjacent_idx = [(tmp_i + 1, tmp_j), (tmp_i + 1, tmp_j + 1), (tmp_i, tmp_j + 1), (tmp_i - 1, tmp_j + 1),
                                    (tmp_i - 1, tmp_j), (tmp_i - 1, tmp_j - 1), (tmp_i, tmp_j - 1), (tmp_i + 1, tmp_j - 1)]
                    
                    adjacent_pixel = [self.skeleton[i + 1, j], self.skeleton[i + 1, j + 1], self.skeleton[i, j + 1], self.skeleton[i - 1, j + 1],
                                      self.skeleton[i - 1, j], self.skeleton[i - 1, j - 1], self.skeleton[i, j - 1], self.skeleton[i + 1, j - 1]]

                    n_adjacent = np.count_nonzero(adjacent_pixel)
                    start_i = tmp_i
                    start_j = tmp_j
                    point_idxs.append((tmp_i, tmp_j))
                    self.start_end.append((order, start_i, start_j))
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
                                stroke_len += 1
                                
                                self.table[adj_i, adj_j] = count
                                tmp_i = adj_i
                                tmp_j = adj_j
                                point_idxs.append((tmp_i, tmp_j))

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
                                elif n_adjacent == 0:
                                
                                    continue
                        if FLAG:
                            break

                    if stroke_len < 10:
                        self.start_end.pop()
                    else:
                        # add mid pixel
                        mid_i, mid_j = point_idxs[int(stroke_len/2)]
                        self.start_end.append((order, mid_i, mid_j))
                    
                        # add end pixel
                        end_i = tmp_i
                        end_j = tmp_j
                        self.start_end.append((order, end_i, end_j))
                        
                        if self.type == 'BK': 
                            k = 1
                            while(1):
                                if not os.path.isfile(SAVE_PATH + '/' + self.type + '/' + self.character + '/' + '%s_%02d_start_end.txt' % (self.character, k)):
                                    np.savetxt(SAVE_PATH + '/' + self.type + '/' + self.character + '/' + '%s_%02d_start_end.txt' % (self.character, k), self.start_end[-3:], fmt='%d', delimiter=',')
                                    np.savetxt(SAVE_PATH + '/' + self.type + '/' + self.character + '/' + '%s_%02d_table.txt' % (self.character, k), self.table, fmt='%d', delimiter='')
                                    np.savetxt(SAVE_PATH + '/' + self.type + '/' + self.character + '/' + '%s_%02d_skeleton.txt' % (self.character, k), self.skeleton, fmt='%d', delimiter='')
                                    break
                                k += 1
                                
                        # the stroke order that stroke len > 10
                        order += 1
                    
                    print('end   : ( %3d, %3d )' % (tmp_i, tmp_j))
                    print('#############################')
                    
                    count += 1
                        
        if self.type == 'SG':
            np.savetxt(SAVE_PATH + '/' + self.type + '/' + self.character + '/' + '%s_start_end.txt' % self.character, self.start_end, fmt='%d', delimiter=',')
            np.savetxt(SAVE_PATH + '/' + self.type + '/' + self.character + '/' + '%s_table.txt' % self.character, self.table, fmt='%d', delimiter='')
            np.savetxt(SAVE_PATH + '/' + self.type + '/' + self.character + '/' + '%s_skeleton.txt' % self.character, self.skeleton, fmt='%d', delimiter='')
        self.stroke_len = count - 1
       
    def match_stroke(self):

        # load start_end.txt of SG stroke
        point_file_a = '%s_start_end.txt' % (self.character)    
        start_end_a = np.loadtxt(SAVE_PATH + '/' +  TYPE_1 + '/' + self.character + '/' + point_file_a, delimiter=',', dtype=np.int16)
        num_stroke_SK = int(start_end_a.shape[0] / 3)
        match_table = np.chararray((num_stroke_SK, 2), itemsize=4, unicode=True)
        dir_b = SAVE_PATH + '/' +  TYPE_2 + '/' + self.character
        num_stroke_BK = int(len(os.listdir(dir_b))/4)
        dis_table = np.zeros((num_stroke_SK, num_stroke_BK))
        dis_table[:] = 999999
        
        # print('start_end_a:')
        # print(start_end_a)
        for len_a in range(num_stroke_SK):
            match_table[len_a, 0] = len_a + 1 
            _, start_x_a, start_y_a = start_end_a[3 * len_a]
            _, mid_x_a, mid_y_a     = start_end_a[3 * len_a + 1]
            _, end_x_a, end_y_a     = start_end_a[3 * len_a + 2]

            #print('start_x_a, start_y_a:', start_x_a, start_y_a)
            #print('mid_x_a, mid_y_a:', mid_x_a, mid_y_a)
            #print('end_x_a, end_y_a', end_x_a, end_y_a)

            min_distance = 999999
            print('################################################')
            
            for i in range(1, num_stroke_BK+1):
            
                # load start_end.txt of BK stroke
                start_end_b = np.loadtxt(dir_b + '/%s_%02d_start_end.txt' % (self.character, i), delimiter=',', dtype=np.int16)
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
                dis_table[len_a, i-1] = dis
                print('min dis:', min_distance, 'dis:', dis)
                
                if dis < min_distance:
                    min_distance = dis
                    match_table[len_a, 1] = '%04d' % i
            print('SG: %s_%d is matched to BK: %4s' % (self.character, len_a+1, match_table[len_a, 1]))        
            print('################################################')
        print(dis_table)
        match_result = np.zeros((num_stroke_SK))
        match_result[:] = -1
        while -1 in match_result:
            min_dis = np.unravel_index(np.argmin(dis_table), dis_table.shape)
            
            # min_dis[0] : stroke order of SK
            # min_dis[1] : stroke order of BK
            # SK stroke is not matched
            if match_result[min_dis[0]] == -1:
                print('################################################')
                # BK stroke is not matched
                #if (min_dis[1] + 1) not in match_result:
                match_result[min_dis[0]] = min_dis[1] + 1
                print('matched.')
                
                print('min:', dis_table[min_dis])
                print('min idx: (%d, %d)' % (min_dis[0]+1, min_dis[1]+1))
                print(dis_table)
                dis_table[min_dis] = 999999
                print(match_result)
                print('################################################')
            # SK stroke is matched
            else:
                dis_table[min_dis] = 999999

            
        np.savetxt(SAVE_PATH + '/' +  TYPE_1 + '/' + '%s_match.txt' % self.character, match_result, fmt='%d', delimiter=',')
        
        # a BK stroke 
        
        
        # print('start_end_b:')
        # print(start_end_b)
        # print('hi')        
    
    def save_stroke(self):    
        # store 
        num_stroke = 0
        print('stroke len:', self.stroke_len)
        for n_stroke in range(self.stroke_len):
            
            if np.count_nonzero(self.table == (n_stroke + 1)) < 10:
                continue

            num_stroke += 1
            stroke_img = np.zeros((256, 256))
            for i in range(256):
                for j in range(256):
                    if self.table[i, j] == (n_stroke + 1):
                        stroke_img[i, j] = 255
            
            if self.type == 'BK':
                k = 1
                while(1):
                    save_name = '%s_%s_%02d.jpg' % (self.type, self.character, k)
                    if not os.path.isfile(SAVE_PATH + '/' + self.type + '/' + self.character + '/' + save_name):
                        break
                    k += 1

            elif self.type == 'SG':
                save_name = '%s_%s_%02d.jpg' % (self.type, self.character, num_stroke)
            
            # cv2.imwrite(SAVE_PATH + '/' + self.type + '/' + self.character + '/' + '%s_%d.jpg' % (self.img_name[:-4], num_stroke), stroke_img)
            cv2.imencode('.jpg', stroke_img)[1].tofile(SAVE_PATH + '/' + self.type + '/' + self.character + '/' + save_name)

if __name__ == '__main__':
    
    # TYPE_1 = 'SG'
    # TYPE_2 = 'BK'
    # IMG_PATH = 'data'
    # SAVE_PATH = 'result'
    '''
    # getting SG stroke order
    for root, dirs, fs in os.walk(IMG_PATH + '/' + TYPE_1):
        for f in fs:
            if len(f) != 0:
                img_path = root + '/' + f
                print('img path:', img_path)
            data = Data(img_path, f, root[-1], TYPE_1)
            data.make_dir()
            data.load_data()
            data.thin()
            data.split()
            data.save_stroke()
    
    # getting BK stroke order
    
    for root, dirs, fs in os.walk(IMG_PATH + '/' + TYPE_2):
        CLEAN = True
        for f in fs:
            if len(f) != 0:
                img_path = root + '/' + f
                print('img path:', img_path)
                print('character:', root[-1])
            data = Data(img_path, f, root[-1], TYPE_2)
            if CLEAN:
                data.make_dir()
                CLEAN = False
            data.load_data()
            data.thin()
            data.split()
            data.save_stroke()
    '''
    # matching SG stroke order to BK stroke
    for root, dirs, fs in os.walk(IMG_PATH + '/' + TYPE_1):
        for f in fs:
            if len(f) != 0:
                img_path = root + '/' + f
                print('img path:', img_path)
            data = Data(img_path, f, root[-1], TYPE_1)
            data.match_stroke()
    