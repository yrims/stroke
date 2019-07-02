import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.spatial.distance import cosine

SAVE_PATH = "result/SG"

def load_data(img_path):
    img = Image.open(img_path)
    img = img.convert('L')
    img = np.array(img)
    img = binarization(img)
    return img

def binarization(img):
    thres = 128
    for i in range(256):
        for j in range(256):
            if img[i, j] < thres:
                img[i, j] = 0
            else:
                img[i, j] = 1
    return img 

def cos(v_a, v_b):
    return cosine(v_a, v_b)

def is_extreme(i, j):
    # start/end point or not
    # e.g.
    # 0  1  0
    # 0  1  0
    # 0  0  0
    if np.count_nonzero(update_point(i, j)) == 1:
        return True
    return False

def is_connected(i, j):
    # connected point or not
    # e.g.
    #  0  1  0
    #  0  1  1
    #  0  1  0
    if np.count_nonzero(update_point(i, j)) > 2:
        return True
    return False

def is_board(i, j):
    if i == 0 or j == 0 or i == 255 or j == 255 :
        return True
    return False

def has_value(i, j, skeleton):
        if skeleton[i, j] == 1:
            return True
        return False


def not_find(i, j, table):
        if table[i, j] == 0:
            return True
        return False


def compute_vector(p1x, p1y, p2x, p2y):
        return (p2x - p1x, p2y - p1y)


def print_block(i, j, skeleton):
    print('    |  %3d %3d %3d' % (j, j + 1, j + 2))
    print('%3d | %3d %3d %3d\n%3d | %3d %3d %3d\n%3d | %3d %3d %3d' %
        (i,
        skeleton[i - 1, j - 1],
        skeleton[i - 1, j],
        skeleton[i - 1, j + 1],
        i + 1,
        skeleton[i, j - 1],
        skeleton[i, j],
        skeleton[i, j + 1],
        i + 2,
        skeleton[i + 1, j - 1],
        skeleton[i + 1, j],
        skeleton[i + 1, j + 1]))

def update_idx(i, j):
    """
    6  7  8
    5  x  1
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

def update_point(i, j, skeleton):
    new_point = [skeleton[i + 1, j    ], 
                skeleton[i + 1, j + 1], 
                skeleton[i    , j + 1], 
                skeleton[i - 1, j + 1],
                skeleton[i - 1, j    ], 
                skeleton[i - 1, j - 1], 
                skeleton[i    , j - 1], 
                skeleton[i + 1, j - 1]]
    return new_point

def update_table(i, j, table):
    new_table = [table[i + 1, j    ], 
                table[i + 1, j + 1], 
                table[i    , j + 1], 
                table[i - 1, j + 1],
                table[i - 1, j    ], 
                table[i - 1, j - 1], 
                table[i    , j - 1], 
                table[i + 1, j - 1]]
    return new_table

def split(path):
    # store each stroke
    table = np.zeros((256, 256), dtype=np.int16)
    # initialize
    all_ = np.zeros((256, 256), dtype=np.int16)
    count = 1 
    order = 1
    surround = []
    start_end = []
    name = path[-5]
    MAX_LENGTH = 15
    # thinned image
    skeleton = load_data(path)
    plt.figure()
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter('result/videos/SG_%s.avi' % name ,fourcc, 20.0, (256, 256), isColor=False)
    for i in range(256):
        for j in range(256):
            if has_value(i, j, skeleton) and not_find(i, j, table):
                if is_board(i, j):
                    continue
                # initialize
                stroke_len = 0  # Compute length in a single stroke.
                point_idxs = [] # Store single stroke coordinate.
                x, y = i, j # search by index (x,y)
                cos_dis = 0 # for direction vector
                p = np.zeros((1, 2)) # momemtum
                n_point = 0
                saveVdo = True # save video
                # table[i, j] = count
                
                # Search deep first.
                # Search eight surrounded points
                while True:
                    if n_point == MAX_LENGTH:
                        break
                    # update neighbor points.
                    adj_idx = update_idx(x, y)
                    adj_point = update_point(x, y, skeleton)
                    found_adj_point = update_table(x, y, table) 
                    # How many adjacent points have not been labeled
                    n_adjacent = np.count_nonzero(adj_point) - np.count_nonzero(found_adj_point)
                    point_idxs.append((x, y))
                    n_point = len(point_idxs)
                    
                    stroke_len += 1
                    # Is connected point.
                    if n_adjacent > 1:
                        if len(point_idxs) < 2:
                            break 
                        min_dis = 2
                        """    
                        if name == '永':
                            print('connected point:', x + 1, y + 1)
                            print_block(x, y, skeleton)       
                            print('point_idxs:',[(p[0] + 1, p[1] + 1) for p in point_idxs])
                        """           
                        # Find which connected component should follow.
                        min_idx = []
                        for n, idx in enumerate(adj_idx):
                            table[x, y] = -1
                            try:
                                if skeleton[idx[0], idx[1]] == 1 and not_find(idx[0], idx[1], table):
                                    v1 = compute_vector(point_idxs[-2][0], point_idxs[-2][1], x, y)
                                    v2 = compute_vector(x, y, idx[0], idx[1])
                                    v1 = np.reshape(np.array(v1), (1, 2))
                                    p = np.append(p, v1, axis=0)
                                    v1 = np.mean(p, axis=0)
                                else:
                                    continue
                            except IndexError:
                                print('IndexError.')
                                break 
                            cos_dis = cos(v1, v2)
                            """
                            if name == '永':
                                print('--------------------')
                                print('idx', idx[0] + 1, idx[1] + 1)
                                print('v1', v1)
                                print('v2', v2)
                                print('cos dis:', cos_dis)
                                print('--------------------')
                            """
                            if cos_dis < min_dis:
                                min_dis = cos_dis
                                min_idx = idx             
                        if n == 7:
                            x, y = min_idx[0], min_idx[1]
                            # adj_idx = update_idx(min_idx[0], min_idx[1])
                            # print('followed point.', x + 1, y + 1)
                            # print('point_idxs:', [(p[0] + 1, p[1] + 1) for p in point_idxs])        
                    # Is not connected point, foward the stroke direction,
                    # and split when angle > 90 (cosine dis > 1).
                    elif n_adjacent == 1:
                        table[x, y] = count
                        if saveVdo:
                            frame = table.copy()
                            frame[np.where(frame > 0)] = 255
                            out.write(np.uint8(frame))
                        for a in adj_idx:
                            if skeleton[a] == 1 and not_find(a[0], a[1], table):
                                break 
                        cos_dis = 0
                        if len(point_idxs) > 1: 
                            v1 = compute_vector(point_idxs[-2][0], point_idxs[-2][1], x, y)
                            v2 = compute_vector(x, y, a[0], a[1])
                            v1 = np.array(v1)
                            v1 = np.reshape(np.array(v1), (1, 2))
                            p = np.append(p, v1, axis=0)
                            v1 = np.mean(p, axis=0)
                            cos_dis = cos(v1, v2)
                        x, y = a[0], a[1]         
                        if cos_dis > 1.5:
                            break
                        # if name == '永':
                            # print('inner point.',x + 1, y + 1)               
                    elif n_adjacent == 0:
                        table[x, y] = count
                        if saveVdo:
                            frame = table.copy()
                            frame[np.where(frame > 0)] = 255
                            out.write(np.uint8(frame))
                        """
                        if name == '永':
                            print('extreme point.', x + 1, y + 1)
                            print('======================')
                        """
                        break
                if stroke_len >= 3:
                    # for xx, yy in point_idxs:
                        # table[xx, yy] = count
                    # add start pixel
                    start_i, start_j = point_idxs[0]
                    start_end.append((order, start_i, start_j))
                    # add mid pixel
                    mid_i, mid_j = point_idxs[int(stroke_len/2)]
                    start_end.append((order, mid_i, mid_j))
                    # add end pixel
                    end_i, end_j = point_idxs[-1] 
                    start_end.append((order, end_i, end_j))
                    stroke_img = np.zeros((256, 256))
                    
                    for ind in point_idxs:
                        stroke_img[ind] = 255 
                    save_name = 'SG_%s_%02d.jpg' % (name, order)
                    cv2.imencode('.jpg', stroke_img)[1].tofile(SAVE_PATH + '/' + name + '/' + save_name)
                    # the stroke order that stroke len > 10
                    order += 1
                    all_ = all_ + stroke_img
                    all_[np.where(stroke_img > 255)] = 255
                elif stroke_len < 3:
                    for xx, yy in point_idxs:
                        table[xx, yy] = 0
                count += 1        
    out.release()
    save_name = 'SG_%s_all.jpg' % (name)
    cv2.imencode('.jpg', all_)[1].tofile(SAVE_PATH + '/' + name + '/' + save_name)
    cv2.destroyAllWindows()
    np.savetxt(SAVE_PATH + '/' + name + '/' + '%s_start_end.txt' % name, start_end, fmt='%d', delimiter=',')
    np.savetxt(SAVE_PATH + '/' + name + '/' + '%s_table.txt' % name, table, fmt='%d', delimiter='')
    np.savetxt(SAVE_PATH + '/' + name + '/' + '%s_skeleton.txt' % name, skeleton, fmt='%d', delimiter='')
    stroke_len = count - 1     


if __name__ == '__main__':
    for root, dirs, fs in os.walk(SAVE_PATH):
        for f in fs:
            if len(f) == 8:
                p = os.path.join(root, f)
                # if p[-5] == '永':
                print(p)
                split(p)
            
