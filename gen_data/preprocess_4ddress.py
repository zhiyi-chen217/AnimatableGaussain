import glob
import os
import pickle
import shutil
import numpy as np


SUBJECT = 185
CAMERA = 76
TAKES = [i for i in range(8, 10)]
LAYER = "Inner"
def read_all_png_camera(data_dir, png_type="images"):
    png_list = []
    for take in TAKES:
        png_list += glob.glob(
            os.path.join(data_dir, '%05d' % SUBJECT, LAYER, f'Take{take}', "Capture", '%04d' % CAMERA, png_type, '*.png'),
            recursive=True)
    png_list.sort()
    return png_list
def read_all_pose(data_dir):
    pose_list = []
    for take in TAKES:
        pose_list += glob.glob(
            os.path.join(data_dir, '%05d' % SUBJECT, LAYER, f'Take{take}', "SMPLX", '*.pkl'),
            recursive=True)
    pose_list.sort()
    return pose_list
def copy_png_to_folder(new_data_dir, png_list, png_type="images"):
    if not os.path.exists(os.path.join(new_data_dir, '%05d' % SUBJECT, LAYER, '%04d' % CAMERA, png_type)):
        os.makedirs(os.path.join(new_data_dir, '%05d' % SUBJECT, LAYER, '%04d' % CAMERA, png_type))
    for ind in range(len(png_list)):
        new_path = os.path.join(new_data_dir, '%05d' % SUBJECT, LAYER, '%04d' % CAMERA, png_type, f'{ind:05d}.png')
        shutil.copy(png_list[ind], new_path)

def copy_pose_to_folder(new_data_dir, pose_list):
    if not os.path.exists(os.path.join(new_data_dir, '%05d' % SUBJECT, LAYER, "pose")):
        os.makedirs(os.path.join(new_data_dir, '%05d' % SUBJECT, LAYER, "pose"))
    for ind in range(len(pose_list)):
        new_path = os.path.join(new_data_dir, '%05d' % SUBJECT, LAYER, "pose", f'{ind:05d}.pkl')
        shutil.copy(pose_list[ind], new_path)
def combine_pose_to_npz(new_data_dir):
    regstr_list = []
    regstr_list += glob.glob(
        os.path.join(new_data_dir, '%05d' % SUBJECT, LAYER, "pose", '*.pkl'),
        recursive=True)
    regstr_list.sort()
    smpl_data = None
    for scan in regstr_list:
        try:
            with open(scan, 'rb') as f:
                regstr = pickle.load(f)
        except:
            print('corrupted npz')

        if smpl_data is None:
            del regstr['leye_pose']
            del regstr['reye_pose']
            smpl_data = regstr
            # expand dimension to concat
            for k in smpl_data.keys():
                smpl_data[k] = np.expand_dims(smpl_data[k], 0)
        else:
            for k in smpl_data.keys():
                if k == "betas":
                    continue
                smpl_data[k] = np.concatenate((smpl_data[k], np.expand_dims(regstr[k], 0)), axis=0)
    np.savez(os.path.join(new_data_dir, '%05d' % SUBJECT, LAYER, 'smpl_params.npz'), **smpl_data)


if __name__ == '__main__':
    data_dir = "/home/zhiychen/Desktop/snarf/data/4d_dress/"
    new_data_dir = "/home/zhiychen/Desktop/train_data/multiviewRGC/4d_dress/"
    image_list = read_all_png_camera(data_dir)
    mask_list = read_all_png_camera(data_dir, png_type="masks")
    pose_list = read_all_pose(data_dir)
    # copy_png_to_folder(new_data_dir, image_list)
    # copy_png_to_folder(new_data_dir, mask_list, png_type="masks")
    # copy_pose_to_folder(new_data_dir, pose_list)
    combine_pose_to_npz(new_data_dir)
    print("end")