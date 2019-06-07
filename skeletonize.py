import os
import cv2
import math
import shutil
from skimage.morphology import skeletonize
import matplotlib.pyplot as plt
from PIL import Image
from scipy.spatial.distance import cosine
import numpy as np

TYPE_1 = 'SG'
TYPE_2 = 'BK'
IMG_PATH = 'data'
SAVE_PATH = 'result'

font = {'family' : 'DFKai-SB',
'weight' : 'bold',
'size'  : '16'}
plt.rc('font', **font) # pass in the font dict as kwargs
plt.rc('axes',unicode_minus=False)

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
        self.binarization()

    def binarization(self):
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
    
    def cos(self, v_a, v_b):
        return cosine(v_a, v_b)

    def is_extreme(self, i, j):
        # start/end point or not
        # e.g.
        # 0  1  0
        # 0  1  0
        # 0  0  0
        if np.count_nonzero(self.update_point(i, j)) == 1:
            return True
        return False
    
    def is_connected(self, i, j):
        # connected point or not
        # e.g.
        #  0  1  0
        #  0  1  1
        #  0  1  0
        if np.count_nonzero(self.update_point(i, j)) > 2:
            return True
        return False

    def is_board(self, i, j):
        if i == 0 or j == 0 or i == 255 or j == 255 :
            return True
        return False
    
    def has_value(self, i, j):
        if self.skeleton[i, j] == 1:
            return True
        return False

    def not_find(self, i, j):
        if self.table[i, j] == 0:
            return True
        return False
    
    def compute_vector(self, p1x, p1y, p2x, p2y):
        return (p2x - p1x, p2y - p1y)

    def update_idx(self, i, j):
        """
        6  7  8
        5  0  1
        4  3  2
        """
        new_idx = [(i + 1, j    ), 
                   (i + 1, j + 1), 
                   (i    , j + 1), 
                   (i - 1, j + 1),
                   (i - 1, j    ), 
                   (i - 1, j - 1), 
                   (i    , j - 1), 
                   (i + 1, j - 1)]
        return new_idx
    
    def update_point(self, i, j):
        new_point = [self.skeleton[i + 1, j    ], 
                     self.skeleton[i + 1, j + 1], 
                     self.skeleton[i    , j + 1], 
                     self.skeleton[i - 1, j + 1],
                     self.skeleton[i - 1, j    ], 
                     self.skeleton[i - 1, j - 1], 
                     self.skeleton[i    , j - 1], 
                     self.skeleton[i + 1, j - 1]]
        return new_point

    def update_table(self, i, j):
        new_table = [self.table[i + 1, j    ], 
                     self.table[i + 1, j + 1], 
                     self.table[i    , j + 1], 
                     self.table[i - 1, j + 1],
                     self.table[i - 1, j    ], 
                     self.table[i - 1, j - 1], 
                     self.table[i    , j - 1], 
                     self.table[i + 1, j - 1]]
        return new_table

    def array_to_opencv(self, a):
        img = 255 * a.copy()
        img = Image.fromarray(img)
        # img = cv2.cvtColor(a, cv2.COLOR_GRAY2BGR)
        return img

    def make_dir(self):
        if os.path.exists(SAVE_PATH + '/' + self.type + '/' + self.character):
            shutil.rmtree(SAVE_PATH + '/' + self.type + '/' + self.character)
            while not os.path.exists(SAVE_PATH + '/' + self.type + '/' + self.character):
                os.mkdir(SAVE_PATH + '/' + self.type + '/' + self.character)
        else:
            os.mkdir(SAVE_PATH + '/' + self.type + '/' + self.character)
    
    def split(self):
        """
        search all points in self.skeleton,
        save the stroke number in self.table
        """
        # store each stroke
        self.table = np.zeros((256, 256), dtype=np.int8)
        # split stroke order
        count = 1 
        order = 1
        surround = []
        plt.figure()

        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter('result/videos/%s_%s.avi' % (self.character, self.type) ,fourcc, 20.0, (256, 256), isColor=False)
        
        for i in range(256):
            
            for j in range(256):
                
                if self.has_value(i, j) and self.not_find(i, j):
                    
                    # Compute length in a single stroke.
                    stroke_len = 1
                    
                    # Store single stroke coordinate.
                    point_idxs = []

                    self.table[i, j] = count

                    tmp_i, tmp_j = i, j

                    #if self.character == '永':
                    #    print('extreme point.', tmp_i, tmp_j)
                    
                    self.table[tmp_i, tmp_j] = count
                    
                    adjacent_idx = self.update_idx(tmp_i, tmp_j)
                    adjacent_point = self.update_point(tmp_i, tmp_j)
                    n_adjacent = np.count_nonzero(adjacent_point)

                    start_i = tmp_i
                    start_j = tmp_j
                    point_idxs.append((tmp_i, tmp_j))

                    self.start_end.append((order, start_i, start_j))
                    
                    # Search deep first.
                    while(1):
                        ENDPOINT = 1
                        n_adjacent = 0
                        cos_dis = 0
                        showImg = 1
                        
                        if showImg:
                            #plt.title('%s : %d (瘦金體)' % (self.character, count))
                            #plt.axis('off')   
                            #plt.imshow(self.table)
                            #plt.pause(0.000001)
                            #plt.clf()
                            frame = self.table.copy()
                            frame[np.where(frame > 0)] = 255
                            out.write(np.uint8(frame))
                            #cv2.imshow('frame', frame)
                            #if cv2.waitKey(1)& 0xFF == ord('q'):
                            #    break

                        # Search eight surrounded points
                        for idx in adjacent_idx:
                            adj_i, adj_j = idx
                            if self.is_board(adj_i, adj_j):
                                break
                            if self.has_value(adj_i, adj_j) and self.not_find(adj_i, adj_j):
                                ENDPOINT = 0
                                stroke_len += 1
                                
                                if len(point_idxs) > 1: 
                                    v1 = self.compute_vector(point_idxs[-2][0], point_idxs[-2][1], point_idxs[-1][0], point_idxs[-1][1])
                                    v2 = self.compute_vector(point_idxs[-1][0], point_idxs[-1][1], adj_i, adj_j)
                                    cos_dis = self.cos(v1, v2)
                                    
                                    """
                                    if cos_dis > 1:
                                        print("-----------------------------")
                                        print("v1:", v1, "v2:", v2)
                                        print("cosine dis(0~2):", cos_dis)
                                        print("-----------------------------")
                                    """
                                
                                

                                self.table[adj_i, adj_j] = count
                                tmp_i = adj_i
                                tmp_j = adj_j
                                point_idxs.append((tmp_i, tmp_j))

                                adjacent_idx = self.update_idx(tmp_i, tmp_j)
                                adjacent_point = self.update_point(tmp_i, tmp_j)
                                found_adjacent_point = self.update_table(tmp_i, tmp_j)

                                # How many adjacent points have not been labeled
                                n_adjacent = np.count_nonzero(adjacent_point) - np.count_nonzero(found_adjacent_point)

                                # Is connected point.
                                if n_adjacent > 1:
                                    self.table[tmp_i, tmp_j] = count
                                    next_min_point = 2
                                    point_idxs.append((tmp_i, tmp_j))
                                    """
                                    if self.character == '永':
                                        print('connected point.', tmp_i, tmp_j)
                                        print('%d %d %d\n%d %d %d\n%d %d %d' % (self.skeleton[tmp_i - 1, tmp_j - 1], 
                                                                                self.skeleton[tmp_i - 1, tmp_j    ],
                                                                                self.skeleton[tmp_i - 1, tmp_j + 1],
                                                                                self.skeleton[tmp_i    , tmp_j - 1],
                                                                                self.skeleton[tmp_i    , tmp_j    ],
                                                                                self.skeleton[tmp_i    , tmp_j + 1],
                                                                                self.skeleton[tmp_i + 1, tmp_j - 1],
                                                                                self.skeleton[tmp_i + 1, tmp_j    ],
                                                                                self.skeleton[tmp_i + 1, tmp_j + 1]))
                                    """
                                    # Find which connected component should follow.
                                    for n, idx in enumerate(adjacent_idx):
                                        try:
                                            if self.skeleton[idx[0], idx[1]] == 1:
                                                v1 = self.compute_vector(point_idxs[-2][0], point_idxs[-2][1], point_idxs[-1][0], point_idxs[-1][1])
                                                v2 = self.compute_vector(point_idxs[-1][0], point_idxs[-1][1], idx[0], idx[1])
                                            else:
                                                continue
                                        except IndexError:
                                            break
                                        
                                        cos_dis = self.cos(v1, v2)
                                        
                                        if cos_dis < next_min_point:
                                            next_min_point = cos_dis
                                            min_idx = idx
                                            
                                            if n == 7:
                                                adjacent_idx = self.update_idx(min_idx[0], min_idx[1])
                                                # print('followed point.', min_idx[0], min_idx[1])
                                                break
                                    # print('n_adj')
                                    
                                
                                # Is not connected point, foward the stroke direction,
                                # and split when angle > 90 (cosine dis > 1).
                                elif n_adjacent == 1:
                                    if cos_dis > 1.5:
                                        self.table[tmp_i, tmp_j] = count
                                        #print('cos dis > 1.5')
                                        break
                                    self.table[tmp_i, tmp_j] = count
                                    point_idxs.append((tmp_i, tmp_j))
                                    adjacent_idx = self.update_idx(tmp_i, tmp_j)
                                    
                                    #if self.character == '永':
                                    #    print('inner point.',tmp_i, tmp_j)
                                           
                                elif n_adjacent == 0:
                                    #if self.character == '永':
                                    #    print('extreme point.', tmp_i, tmp_j)
                                    ENDPOINT = 1
                                    break
                        
                        if cos_dis > 1.5:
                            break

                        if ENDPOINT:
                            break
                        
                        

                    if stroke_len < 15:
                        self.start_end.pop()
                    else:
                        # add mid pixel
                        mid_i, mid_j = point_idxs[int(stroke_len/2)]
                        self.start_end.append((order, mid_i, mid_j))
                    
                        # add end pixel
                        end_i, end_j = point_idxs[-1] 
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
                    
                    #print('end   : ( %3d, %3d )' % (tmp_i, tmp_j))
                    #print('#############################')
                    
                    count += 1
        # plt.close('all')
        out.release()
        cv2.destroyAllWindows()
        if self.type == 'SG':
            np.savetxt(SAVE_PATH + '/' + self.type + '/' + self.character + '/' + '%s_start_end.txt' % self.character, self.start_end, fmt='%d', delimiter=',')
            np.savetxt(SAVE_PATH + '/' + self.type + '/' + self.character + '/' + '%s_table.txt' % self.character, self.table, fmt='%d', delimiter='')
            np.savetxt(SAVE_PATH + '/' + self.type + '/' + self.character + '/' + '%s_skeleton.txt' % self.character, self.skeleton, fmt='%d', delimiter='')
        self.stroke_len = count - 1
       
    def match_stroke(self):
        
        dir_b = SAVE_PATH + '/' +  TYPE_2 + '/' + self.character
        if not os.path.exists(dir_b):
            print('SG: %s does not has coorespond BK.' % self.character)
            return
        # load start_end.txt of SG stroke
        point_file_a = '%s_start_end.txt' % (self.character)    
        start_end_a = np.loadtxt(SAVE_PATH + '/' +  TYPE_1 + '/' + self.character + '/' + point_file_a, delimiter=',', dtype=np.int16)
        num_stroke_SG = int(start_end_a.shape[0] / 3)
        num_stroke_BK = int(len(os.listdir(dir_b))/4)
        match_table = np.chararray((num_stroke_SG, 2), itemsize=4, unicode=True)
        dis_table = np.zeros((num_stroke_SG, num_stroke_BK))
        dis_table[:] = 999999
        
        # print('start_end_a:')
        # print(start_end_a)
        for len_a in range(num_stroke_SG):
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
        match_result = np.zeros((num_stroke_SG))
        match_result[:] = -1
        while -1 in match_result:
            min_dis = np.unravel_index(np.argmin(dis_table), dis_table.shape)
            
            # min_dis[0] : stroke order of SK
            # min_dis[1] : stroke order of BK
            # SG stroke is not matched
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
                
                SG_img = Image.open('result/SG/%s/SG_%s_%02d.jpg' % (self.character, self.character, min_dis[0] + 1))
                BK_img = Image.open('result/BK/%s/BK_%s_%02d.jpg' % (self.character, self.character, min_dis[1] + 1))
                
                plt.figure(figsize=(8,4))
                plt.subplot(1, 2, 1)
                plt.title('%s : %d (瘦金體)' % (self.character, min_dis[0] + 1))
                plt.axis('off')
                plt.imshow(SG_img)
                
                plt.subplot(1, 2, 2)
                plt.title('%s : %d (標楷體)' % (self.character, min_dis[1] + 1))
                plt.imshow(BK_img)
                plt.axis('off')
                
                plt.savefig('result/match_img/%s_%02d.jpg' % (self.character, min_dis[0] + 1))
                # plt.show()
                
            # SG stroke is matched
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
            
            if np.count_nonzero(self.table == (n_stroke + 1)) < 15:
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
            print('Start spliting strokes.')
            data.split()
            data.save_stroke()
    
    # getting BK stroke order
    
    for root, dirs, fs in os.walk(IMG_PATH + '/' + TYPE_2):
        CLEAN = True
        for f in fs:
            if len(f) != 0:
                img_path = root + '/' + f
                print('img path:', img_path)
                # print('character:', root[-1])
            
            data = Data(img_path, f, root[-1], TYPE_2)
            if CLEAN:
                data.make_dir()
                CLEAN = False
            data.load_data()
            data.thin()
            print('Start spliting strokes.')
            data.split()
            data.save_stroke()
    
    
    # matching SG stroke order to BK stroke
    for root, dirs, fs in os.walk(IMG_PATH + '/' + TYPE_1):
        for f in fs:
            if len(f) != 0:
                img_path = root + '/' + f
                print('img path:', img_path)
            
            data = Data(img_path, f, root[-1], TYPE_1)
            data.match_stroke()
    