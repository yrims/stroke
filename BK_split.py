import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

IMG_PATH = "data/SG"
SAVE_PATH = "result"


def load_data(img_path):
    img = Image.open(img_path)
    img = np.array(img)
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

def update_point(i, j):
    new_point = [skeleton[i + 1, j    ], 
                skeleton[i + 1, j + 1], 
                skeleton[i    , j + 1], 
                skeleton[i - 1, j + 1],
                skeleton[i - 1, j    ], 
                skeleton[i - 1, j - 1], 
                skeleton[i    , j - 1], 
                skeleton[i + 1, j - 1]]
    return new_point

def update_table(i, j):
    new_table = [table[i + 1, j    ], 
                table[i + 1, j + 1], 
                table[i    , j + 1], 
                table[i - 1, j + 1],
                table[i - 1, j    ], 
                table[i - 1, j - 1], 
                table[i    , j - 1], 
                table[i + 1, j - 1]]
    return new_table

def split():
    # store each stroke
    table = np.zeros((256, 256), dtype=np.int8)
    # split stroke order
    count = 1 
    order = 1
    surround = []
    plt.figure()
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter('result/videos/%s_SG.avi' % character ,fourcc, 20.0, (256, 256), isColor=False)
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
                saveVdo = True # save video
                table[i, j] = count
                if name == '永':
                    print('======================')
                    print('extreme point.', x + 1, y + 1)
                if saveVdo:
                    frame = table.copy()
                    frame[np.where(frame > 0)] = 255
                    out.write(np.uint8(frame))
                # Search deep first.
                # Search eight surrounded points
                print('Hi')
                while True:
                    # update neighbor points.
                    adj_idx = update_idx(x, y)
                    adj_point = .update_point(x, y)
                    found_adj_point = .update_table(x, y) 
                    # How many adjacent points have not been labeled
                    n_adjacent = np.count_nonzero(adj_point) - np.count_nonzero(found_adj_point)
                    point_idxs.append((x, y))

                    table[x, y] = count
                    stroke_len += 1
                    # Is connected point.
                    if n_adjacent > 1:
                        if len(point_idxs) < 2:
                            break 
                        next_min_point = 2    
                        if name == '永':
                            print('connected point:', x + 1, y + 1)
                            print_block(x, y, skeleton)       
                            print('point_idxs:',[(p[0] + 1, p[1] + 1) for p in point_idxs])           
                        # Find which connected component should follow.
                        min_idx = []
                        for n, idx in enumerate(adj_idx):
                            print('n:',n)
                            try:
                                if skeleton[idx[0], idx[1]] == 1 and not_find(idx[0], idx[1]):
                                    v1 = .compute_vector(point_idxs[-2][0], point_idxs[-2][1], x, y)
                                    v2 = .compute_vector(x, y, idx[0], idx[1])
                                else:
                                    continue
                            except IndexError:
                                print('IndexError.')
                                break               
                            cos_dis = .cos(v1, v2)
                            if name == '永':
                                print('--------------------')
                                print('idx', idx[0] + 1, idx[1] + 1)
                                print('v1', v1)
                                print('v2', v2)
                                print('cos dis:', cos_dis)
                                print('--------------------')
                            if cos_dis < next_min_point:
                                next_min_point = cos_dis
                                min_idx = idx             
                        if n == 7:
                            x, y = min_idx[0], min_idx[1]
                            # adj_idx = update_idx(min_idx[0], min_idx[1])
                            print('followed point.', x + 1, y + 1)
                            print('point_idxs:', [(p[0] + 1, p[1] + 1) for p in point_idxs])        
                    # Is not connected point, foward the stroke direction,
                    # and split when angle > 90 (cosine dis > 1).
                    elif n_adjacent == 1: 
                        for a in adj_idx:
                            if skeleton[a] == 1 and not_find(a[0], a[1]):
                                break
                        adj_i, adj_j = a[0], a[1]
                        x, y = adj_i, adj_j
                        cos_dis = 0
                        if len(point_idxs) > 10: 
                            v1 = .compute_vector(point_idxs[-10][0], point_idxs[-10][1], point_idxs[-5][0], point_idxs[-5][1])
                            v2 = .compute_vector(point_idxs[-5][0], point_idxs[-5][1], adj_i, adj_j)
                            cos_dis = .cos(v1, v2)        
                        if cos_dis > 0.5:
                            break
                        # if name == '永':
                            # print('inner point.',x + 1, y + 1)               
                    elif n_adjacent == 0:
                        if name == '永':
                            print('extreme point.', x + 1, y + 1)
                            print('======================')
                        break
                if stroke_len > 10:
                    # table[point_idxs] = -1
                    # add start pixel
                    start_i, start_j = point_idxs[0]
                    start_end.append((order, start_i, start_j))
                    # add mid pixel
                    mid_i, mid_j = point_idxs[int(stroke_len/2)]
                    start_end.append((order, mid_i, mid_j))
                    # add end pixel
                    end_i, end_j = point_idxs[-1] 
                    start_end.append((order, end_i, end_j))
                    # the stroke order that stroke len > 10
                    order += 1
                count += 1        
                if .type == 'BK': 
                    k = 1
                    while(1):
                        if not os.path.isfile(SAVE_PATH + '/' + .type + '/' + name + '/' + '%s_%02d_start_end.txt' % (name, k)):
                            np.savetxt(SAVE_PATH + '/' + .type + '/' + name + '/' + '%s_%02d_start_end.txt' % (name, k), start_end[-3:], fmt='%d', delimiter=',')
                            np.savetxt(SAVE_PATH + '/' + .type + '/' + name + '/' + '%s_%02d_table.txt' % (name, k), table, fmt='%d', delimiter='')
                            np.savetxt(SAVE_PATH + '/' + .type + '/' + name + '/' + '%s_%02d_skeleton.txt' % (name, k), skeleton, fmt='%d', delimiter='')
                            break
                        k += 1
    out.release()
    cv2.destroyAllWindows()
    if .type == 'SG':
        np.savetxt(SAVE_PATH + '/' + .type + '/' + name + '/' + '%s_start_end.txt' % name, start_end, fmt='%d', delimiter=',')
        np.savetxt(SAVE_PATH + '/' + .type + '/' + name + '/' + '%s_table.txt' % name, table, fmt='%d', delimiter='')
        np.savetxt(SAVE_PATH + '/' + .type + '/' + name + '/' + '%s_skeleton.txt' % name, skeleton, fmt='%d', delimiter='')
    stroke_len = count - 1     


if __name__ == '__main__':
    for root, dirs, fs in os.walk(IMG_PATH):
        for f in fs:
            p = os.path.join(root, f)
            img = load_data(p)
            split()
            save_name = 'SG_%s.jpg' % root[-1]
            print(save_name)
            cv2.imencode('.jpg', img)[1].tofile('result/SG' + '/' + root[-1] + '/' + save_name)
