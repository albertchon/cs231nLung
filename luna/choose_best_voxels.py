import numpy as np
import os

LEADERBOARD_SIZE = 4

#ok if the new voxel is overlapping two old ones, its not getting counted.
#if its only overlapping one old one, then we'll count it if its good and remove the old one
def voxels_intersect(location1, location2):
    depth1, height1, width1 = location1
    depth2, height2, width2 = location2
    return abs(depth1 - depth2) <= 32 and abs(height1 - height2) <= 32 and abs(width1 - width2) <= 32


def voxel_smart_insert(voxel_leaderboard, test_voxel_score, test_voxel_location):
    num_overlapping_voxels = 0
    overlapping_voxel = None
    for leader in voxel_leaderboard:
        leader_location = (leader[1], leader[2], leader[3])
        if voxels_intersect(leader_location, test_voxel_location):
            num_overlapping_voxels += 1
            if num_overlapping_voxels > 1:
                return # too many overlapping, no insert needed.
            overlapping_voxel = leader
    new_voxel_entry = (test_voxel_score, test_voxel_location[0], test_voxel_location[1], test_voxel_location[2])
    if num_overlapping_voxels == 0: # case 1: no overlap, simply remove lowest score and add new voxel
        #print("voxel_smart_insert: no overlap...removing last place")
        voxel_leaderboard.remove(min(voxel_leaderboard, key = lambda t: t[0]))
        voxel_leaderboard.append(new_voxel_entry)
    if num_overlapping_voxels == 1: # case 2: one voxel is overlapping
        if overlapping_voxel[0] < test_voxel_score: # case 2a: our new score is better
            #print("voxel_smart_insert: one overlap..replacing")
            voxel_leaderboard.remove(overlapping_voxel)
            voxel_leaderboard.append(new_voxel_entry)

def initialize_leaderboard(ct_data):
    voxel_leaderboard = []
    corner1 = ct_data[:32, :32, :32]
    corner2 = ct_data[:32, 480:, :32]
    corner3 = ct_data[:32, 480:, 480:]
    corner4 = ct_data[:32, :32, 480:]
    voxel_leaderboard.append((np.sum(corner1), 0, 0, 0))
    voxel_leaderboard.append((np.sum(corner2), 0, 480, 0))
    voxel_leaderboard.append((np.sum(corner3), 0, 480, 480))
    voxel_leaderboard.append((np.sum(corner4), 0, 0, 480))
    return voxel_leaderboard


def find_best_voxels(ct_data): # 37 * 512 * 512
    #print(ct_data.shape)
    ct_data /= np.amax(ct_data) # fixes numerical instability
    depth = ct_data.shape[0]
    width = 256
    height = 256
    voxel_leaderboard = initialize_leaderboard(ct_data) # stores (score, voxel_data) tuple
    leaderboard_threshold = 0
    #print(np.amax(ct_data))
    for i in range(depth - 31): # 0,1,2,3,4
        #print('i=' + str(i))
        for j in range(0, height - 31, 4):
            #print('j=' + str(j))
            for k in range(0, width - 31, 4):
                test_voxel = ct_data[i:i + 32, j:j + 32, k:k + 32]
                test_voxel_score = np.sum(test_voxel)
                if(test_voxel_score > leaderboard_threshold):
                    voxel_smart_insert(voxel_leaderboard, test_voxel_score, (i, j, k))
                    leaderboard_threshold = min(voxel_leaderboard, key = lambda t: t[0])[0]
                    #print("score required to get on leaderboard is now: " + str(leaderboard_threshold))

    print(len(voxel_leaderboard))
    sorted_leaderboard = sorted(voxel_leaderboard, key=lambda tup: tup[0], reverse=True)
    for tup in sorted_leaderboard:
        print((tup[1], tup[2], tup[3]))

    return voxel_leaderboard

def process_patient(path, voxel, save_path):
    ct_data = np.load(path + voxel)
    depth = ct_data.shape[0]
    if(depth < 32):
        ct_data = np.pad(ct_data, ((0,32-depth),(0, 0), (0,0)),mode='constant', constant_values=0)
    print(ct_data.shape)
    voxel_leaderboard = find_best_voxels(ct_data)

    vox_data_list = []
    for i in range(LEADERBOARD_SIZE):
        vox = voxel_leaderboard[i]
        vox_data = ct_data[vox[1]:vox[1] + 32, vox[2]:vox[2] + 32, vox[3]:vox[3] + 32]
        vox_data_list.append(vox_data)
        print(np.sum(vox_data))

    combined_voxels_top = np.concatenate([vox_data_list[0], vox_data_list[1]], axis=2) # side by side
    combined_voxels_bottom = np.concatenate([vox_data_list[0], vox_data_list[1]], axis=2) # side by side
    combined_voxels = np.concatenate([combined_voxels_top, combined_voxels_bottom], axis=1) #top-bottom
    np.save(save_path + voxel, combined_voxels)
    print('saved at ' + save_path + voxel)

def main():
    save_path = 'voxel/compiledvoxels/'

    path = "masks/data/"
    voxels = os.listdir(path)
    for voxel in voxels:
        process_patient(path, voxel, save_path)


if __name__ == '__main__':
    main()