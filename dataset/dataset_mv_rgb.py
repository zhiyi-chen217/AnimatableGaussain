import glob
import os
import pickle

import numpy as np
import cv2 as cv
import torch
from torch.utils.data import Dataset

import smplx
import config
import utils.nerf_util as nerf_util
import utils.visualize_util as visualize_util
import dataset.commons as commons
from utils.sh_utils import RGB2GRAY
from PIL import Image

class MvRgbDatasetBase(Dataset):
    @torch.no_grad()
    def __init__(
        self,
        data_dir,
        frame_range = None,
        used_cam_ids = None,
        training = True,
        subject_name = None,
        load_smpl_pos_map = False,
        load_smpl_nml_map = False,
        mode = '3dgs',
        layers = None
    ):
        super(MvRgbDatasetBase, self).__init__()

        self.data_dir = data_dir
        self.training = training
        self.subject_name = subject_name
        self.layers = layers
        if self.subject_name is None:
            self.subject_name = os.path.basename(self.data_dir)
        self.load_smpl_pos_map = load_smpl_pos_map
        self.load_smpl_nml_map = load_smpl_nml_map
        self.mode = mode  # '3dgs' or 'nerf'

        self.load_cam_data()
        self.load_smpl_data()
        if layers is None:
            self.smpl_pos_map = config.opt.get("smpl_pos_map", "smpl_pos_map")
        else:
            self.smpl_pos_map = {}
            for layer in layers:
                self.smpl_pos_map[layer] = config.opt.get("smpl_pos_map", "smpl_pos_map") + f"_{layer}"
        self.smpl_model = smplx.SMPLX(model_path = config.PROJ_DIR + '/smpl_files/smplx', gender = 'female', use_pca = True, num_pca_comps = 12, flat_hand_mean = True, batch_size = 1)

        pose_list = list(range(self.smpl_data['body_pose'].shape[0]))
        if frame_range is not None:
            if isinstance(frame_range, list):
                if len(frame_range) == 2:
                    print(f'# Selected frame indices: range({frame_range[0]}, {frame_range[1]})')
                    frame_range = range(frame_range[0], frame_range[1])
                elif len(frame_range) == 3:
                    print(f'# Selected frame indices: range({frame_range[0]}, {frame_range[1]}, {frame_range[2]})')
                    frame_range = range(frame_range[0], frame_range[1], frame_range[2])
            elif isinstance(frame_range, str):
                frame_range = np.loadtxt(self.data_dir + '/' + frame_range).astype(np.int).tolist()
                print(f'# Selected frame indices: {frame_range}')
            else:
                raise TypeError('Invalid frame_range!')
            self.pose_list = list(frame_range)
        else:
            self.pose_list = pose_list

        if self.training:
            if used_cam_ids is None:
                self.used_cam_ids = list(range(self.view_num))
            else:
                self.used_cam_ids = used_cam_ids
            print('# Used camera ids: ', self.used_cam_ids)
            self.data_list = []
            for pose_idx in self.pose_list:
                for view_idx in self.used_cam_ids:
                    self.data_list.append((pose_idx, view_idx))
            # filter missing files
            self.filter_missing_files()

        print('# Dataset contains %d items' % len(self))

        # SMPL related
        ret = self.smpl_model.forward(betas = self.smpl_data['betas'][0][None],
                                      global_orient = config.cano_smpl_global_orient[None],
                                      transl = config.cano_smpl_transl[None],
                                      body_pose = config.cano_smpl_body_pose[None])
        self.cano_smpl = {k: v[0] for k, v in ret.items() if isinstance(v, torch.Tensor)}
        self.inv_cano_jnt_mats = torch.linalg.inv(self.cano_smpl['A'])
        min_xyz = self.cano_smpl['vertices'].min(0)[0]
        max_xyz = self.cano_smpl['vertices'].max(0)[0]
        self.cano_smpl_center = 0.5 * (min_xyz + max_xyz)
        min_xyz[:2] -= 0.05
        max_xyz[:2] += 0.05
        min_xyz[2] -= 0.15
        max_xyz[2] += 0.15
        self.cano_bounds = torch.stack([min_xyz, max_xyz], 0).to(torch.float32).numpy()
        self.smpl_faces = self.smpl_model.faces.astype(np.int32)

        commons._initialize_hands(self)

    def __len__(self):
        if self.training:
            return len(self.data_list)
        else:
            return len(self.pose_list)

    def __getitem__(self, index):
        return self.getitem(index, self.training)

    def getitem(self, index, training = True, **kwargs):
        if training or kwargs.get('eval', False):  # training or evaluation
            pose_idx, view_idx = self.data_list[index]
            pose_idx = kwargs['pose_idx'] if 'pose_idx' in kwargs else pose_idx
            view_idx = kwargs['view_idx'] if 'view_idx' in kwargs else view_idx
            data_idx = (pose_idx, view_idx)
            if not training:
                print('data index: (%d, %d)' % (pose_idx, view_idx))
        else:  # testing
            pose_idx = self.pose_list[index]
            data_idx = pose_idx
            print('data index: %d' % pose_idx)

        # SMPL
        with torch.no_grad():
            live_smpl = self.smpl_model.forward(
                betas = self.smpl_data['betas'][0][None],
                global_orient = self.smpl_data['global_orient'][pose_idx][None],
                transl = self.smpl_data['transl'][pose_idx][None],
                body_pose = self.smpl_data['body_pose'][pose_idx][None],
                jaw_pose = self.smpl_data['jaw_pose'][pose_idx][None],
                expression = self.smpl_data['expression'][pose_idx][None],
                left_hand_pose = self.smpl_data['left_hand_pose'][pose_idx][None],
                right_hand_pose = self.smpl_data['right_hand_pose'][pose_idx][None]
            )
            cano_smpl = self.smpl_model.forward(
                betas = self.smpl_data['betas'][0][None],
                global_orient = config.cano_smpl_global_orient[None],
                transl = config.cano_smpl_transl[None],
                body_pose = config.cano_smpl_body_pose[None],
                jaw_pose = self.smpl_data['jaw_pose'][pose_idx][None],
                expression = self.smpl_data['expression'][pose_idx][None],
            )
            live_smpl_woRoot = self.smpl_model.forward(
                betas = self.smpl_data['betas'][0][None],
                body_pose = self.smpl_data['body_pose'][pose_idx][None],
                jaw_pose = self.smpl_data['jaw_pose'][pose_idx][None],
                expression = self.smpl_data['expression'][pose_idx][None],
                left_hand_pose=self.smpl_data['left_hand_pose'][pose_idx][None],
                right_hand_pose=self.smpl_data['right_hand_pose'][pose_idx][None]
            )

        data_item = dict()
        if self.load_smpl_pos_map:
            if self.layers is None:
                smpl_pos_map = cv.imread(self.data_dir + '/{}/%08d.exr'.format(self.smpl_pos_map) % pose_idx, cv.IMREAD_UNCHANGED)
                pos_map_size = smpl_pos_map.shape[1] // 2
                smpl_pos_map = np.concatenate([smpl_pos_map[:, :pos_map_size], smpl_pos_map[:, pos_map_size:]], 2)
                smpl_pos_map = smpl_pos_map.transpose((2, 0, 1))
                data_item['smpl_pos_map'] = smpl_pos_map
            else:
                smpl_pos_maps = {}
                for layer in self.layers:
                    smpl_pos_map = cv.imread(self.data_dir + '/{}/%08d.exr'.format(self.smpl_pos_map[layer]) % pose_idx,
                                             cv.IMREAD_UNCHANGED)
                    pos_map_size = smpl_pos_map.shape[1] // 2
                    smpl_pos_map = np.concatenate([smpl_pos_map[:, :pos_map_size], smpl_pos_map[:, pos_map_size:]], 2)
                    smpl_pos_map = smpl_pos_map.transpose((2, 0, 1))
                    smpl_pos_maps[layer] = smpl_pos_map
                data_item['smpl_pos_map'] = smpl_pos_maps


        if self.load_smpl_nml_map:
            smpl_nml_map = cv.imread(self.data_dir + '/smpl_nml_map/%08d.jpg' % pose_idx, cv.IMREAD_UNCHANGED)
            smpl_nml_map = (smpl_nml_map / 255.).astype(np.float32)
            nml_map_size = smpl_nml_map.shape[1] // 2
            smpl_nml_map = np.concatenate([smpl_nml_map[:, :nml_map_size], smpl_nml_map[:, nml_map_size:]], 2)
            smpl_nml_map = smpl_nml_map.transpose((2, 0, 1))
            data_item['smpl_nml_map'] = smpl_nml_map

        data_item['joints'] = live_smpl.joints[0, :22]
        data_item['kin_parent'] = self.smpl_model.parents[:22].to(torch.long)
        data_item['item_idx'] = index
        data_item['data_idx'] = data_idx
        data_item['time_stamp'] = np.array(pose_idx, np.float32)
        data_item['global_orient'] = self.smpl_data['global_orient'][pose_idx]
        data_item['transl'] = self.smpl_data['transl'][pose_idx]
        data_item['live_smpl_v'] = live_smpl.vertices[0]
        data_item['live_smpl_v_woRoot'] = live_smpl_woRoot.vertices[0]
        data_item['cano_smpl_v'] = cano_smpl.vertices[0]
        data_item['cano_jnts'] = cano_smpl.joints[0]
        data_item['cano2live_jnt_mats'] = torch.matmul(live_smpl.A[0], torch.linalg.inv(cano_smpl.A[0]))
        data_item['cano2live_jnt_mats_woRoot'] = torch.matmul(live_smpl_woRoot.A[0], torch.linalg.inv(cano_smpl.A[0]))
        data_item['cano_smpl_center'] = self.cano_smpl_center
        data_item['cano_bounds'] = self.cano_bounds
        data_item['smpl_faces'] = self.smpl_faces
        min_xyz = live_smpl.vertices[0].min(0)[0] - 0.15
        max_xyz = live_smpl.vertices[0].max(0)[0] + 0.15
        live_bounds = torch.stack([min_xyz, max_xyz], 0).to(torch.float32).numpy()
        data_item['live_bounds'] = live_bounds

        if training:
            color_img, mask_img = self.load_color_mask_images(pose_idx, view_idx)

            color_img = (color_img / 255.).astype(np.float32)

            boundary_mask_img, mask_img = self.get_boundary_mask(mask_img)

            if self.mode == '3dgs':
                data_item.update({
                    'img_h': color_img.shape[0],
                    'img_w': color_img.shape[1],
                    'extr': self.extr_mats[view_idx],
                    'intr': self.intr_mats[view_idx],
                    'color_img': color_img,
                    'mask_img': mask_img,
                    'boundary_mask_img': boundary_mask_img
                })
            elif self.mode == 'nerf':
                depth_img = np.zeros(color_img.shape[:2], np.float32)
                nerf_random = nerf_util.sample_randomly_for_nerf_rendering(
                    color_img, mask_img, depth_img,
                    self.extr_mats[view_idx], self.intr_mats[view_idx],
                    live_bounds,
                    unsample_region_mask = boundary_mask_img
                )
                data_item.update({
                    'nerf_random': nerf_random,
                    'extr': self.extr_mats[view_idx],
                    'intr': self.intr_mats[view_idx]
                })
            else:
                raise ValueError('Invalid dataset mode!')
        else:
            """ synthesis config """
            img_h = 512 if 'img_h' not in kwargs else kwargs['img_h']
            img_w = 512 if 'img_w' not in kwargs else kwargs['img_w']
            intr = np.array([[550, 0, 256], [0, 550, 256], [0, 0, 1]], np.float32) if 'intr' not in kwargs else kwargs['intr']
            if 'extr' not in kwargs:
                extr = visualize_util.calc_front_mv(live_bounds.mean(0), tar_pos = np.array([0, 0, 2.5]))
            else:
                extr = kwargs['extr']

            data_item.update({
                'img_h': img_h,
                'img_w': img_w,
                'extr': extr,
                'intr': intr
            })

        if self.mode == 'nerf' or self.mode == '3dgs' and not training:
            # mano
            data_item['left_cano_mano_v'], data_item['left_cano_mano_n'], data_item['right_cano_mano_v'], data_item['right_cano_mano_n'] \
                = commons.generate_two_manos(self, self.cano_smpl['vertices'])
            data_item['left_live_mano_v'], data_item['left_live_mano_n'], data_item['right_live_mano_v'], data_item['right_live_mano_n'] \
                = commons.generate_two_manos(self, live_smpl.vertices[0])

        return data_item

    def load_cam_data(self):
        """
        Initialize:
        self.cam_names, self.view_num, self.extr_mats, self.intr_mats,
        self.img_widths, self.img_heights
        """
        raise NotImplementedError

    def load_smpl_data(self):
        """
        Initialize:
        self.smpl_data, a dict including ['body_pose', 'global_orient', 'transl', 'betas', ...]
        """
        smpl_data = np.load(self.data_dir + '/smpl_params.npz', allow_pickle = True)
        smpl_data = dict(smpl_data)
        self.smpl_data = {k: torch.from_numpy(v.astype(np.float32)) for k, v in smpl_data.items()}

    def filter_missing_files(self):
        pass

    def load_color_mask_images(self, pose_idx, view_idx):
        raise NotImplementedError

    @staticmethod
    def get_boundary_mask(mask, kernel_size = 5):
        """
        :param mask: np.uint8
        :param kernel_size:
        :return:
        """
        mask_bk = mask.copy()
        thres = 128
        mask[mask < thres] = 0
        mask[mask > thres] = 1
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        mask_erode = cv.erode(mask.copy(), kernel)
        mask_dilate = cv.dilate(mask.copy(), kernel)
        boundary_mask = (mask_dilate - mask_erode) == 1
        boundary_mask = np.logical_or(boundary_mask,
                                      np.logical_and(mask_bk > 5, mask_bk < 250))

        # boundary_mask_resized = cv.resize(boundary_mask.astype(np.uint8), (0, 0), fx = 0.5, fy = 0.5)
        # cv.imshow('boundary_mask', boundary_mask_resized.astype(np.uint8) * 255)
        # cv.waitKey(0)

        return boundary_mask, mask == 1

    def compute_pca(self, n_components = 10):
        from sklearn.decomposition import PCA
        from tqdm import tqdm
        import joblib

        if not os.path.exists(self.data_dir + '/{}/pca_%d.ckpt'.format(self.smpl_pos_map) % n_components):
            pose_conds = []
            mask = None
            for pose_idx in tqdm(self.pose_list, desc = 'Loading position maps...'):
                pose_map = cv.imread(self.data_dir + '/{}/%08d.exr'.format(self.smpl_pos_map) % pose_idx, cv.IMREAD_UNCHANGED)
                pose_map = pose_map[:, :pose_map.shape[1] // 2]
                if mask is None:
                    mask = np.linalg.norm(pose_map, axis = -1) > 1e-6
                pose_conds.append(pose_map[mask])
            pose_conds = np.stack(pose_conds, 0)
            pose_conds = pose_conds.reshape(pose_conds.shape[0], -1)
            self.pca = PCA(n_components = n_components)
            self.pca.fit(pose_conds)
            joblib.dump(self.pca, self.data_dir + '/{}/pca_%d.ckpt'.format(self.smpl_pos_map) % n_components)
            self.pos_map_mask = mask
        else:
            self.pca = joblib.load(self.data_dir + '/{}/pca_%d.ckpt'.format(self.smpl_pos_map) % n_components)
            pose_map = cv.imread(sorted(glob.glob(self.data_dir + '/smpl_pos_map{}/0*.exr'.format(self.smpl_pos_map)))[0], cv.IMREAD_UNCHANGED)
            pose_map = pose_map[:, :pose_map.shape[1] // 2]
            self.pos_map_mask = np.linalg.norm(pose_map, axis = -1) > 1e-6

    def transform_pca(self, pose_conds, sigma_pca = 2.):
        pose_conds = pose_conds.reshape(1, -1)
        lowdim_pose_conds = self.pca.transform(pose_conds)
        std = np.sqrt(self.pca.explained_variance_)
        lowdim_pose_conds = np.maximum(lowdim_pose_conds, -sigma_pca * std)
        lowdim_pose_conds = np.minimum(lowdim_pose_conds, sigma_pca * std)
        new_pose_conds = self.pca.inverse_transform(lowdim_pose_conds)
        new_pose_conds = new_pose_conds.reshape(-1, 3)
        return new_pose_conds

class MvRgbDataset4DDress(MvRgbDatasetBase):
    def __init__(
        self,
        data_dir,
        frame_range = None,
        used_cam_ids = None,
        training = True,
        subject_name = None,
        load_smpl_pos_map = False,
        load_smpl_nml_map = False,
        mode = '3dgs',
        layers = None
    ):
        super(MvRgbDataset4DDress, self).__init__(
            data_dir,
            frame_range,
            used_cam_ids,
            training,
            subject_name,
            load_smpl_pos_map,
            load_smpl_nml_map,
            mode,
            layers
        )
        if subject_name is None:
            self.subject_name = os.path.basename(os.path.dirname(self.data_dir))
        if layers is not None:
            self.surface_label = ['lower']
            self.surface_label_color = RGB2GRAY([[128, 0, 255]])
            self.masklabel = dict(zip(self.surface_label, self.surface_label_color))

    def load_cam_data(self):
        import csv
        cam_names = []
        extr_mats = []
        intr_mats = []
        img_widths = []
        img_heights = []
        with open(os.path.join(self.data_dir, "cameras.pkl"), "rb") as fp:
            cameras = pickle.load(fp)
            for cam_name in cameras.keys():
                cam_names.append(cam_name)
                camera = cameras[cam_name]
                img_widths.append(940)
                img_heights.append(1280)

                extr_mat = np.identity(4, np.float32)
                extr_mat[:3, :] = camera["extrinsics"]
                extr_mats.append(extr_mat)

                intr_mats.append(camera["intrinsics"])

        self.cam_names, self.img_widths, self.img_heights, self.extr_mats, self.intr_mats \
            = cam_names, img_widths, img_heights, extr_mats, intr_mats
        self.view_num = len(self.cam_names)

    def load_color_mask_images(self, pose_idx, view_idx):
        cam_name = self.cam_names[view_idx]
        color_img = cv.imread(os.path.join(self.data_dir, cam_name,
                                           "images", '%05d.png' % pose_idx), cv.IMREAD_UNCHANGED)
        if color_img is None:
            from gen_data.preprocess_4ddress import read_all_png_camera, copy_png_to_folder
            data_dir = "/home/zhiychen/Desktop/snarf/data/4d_dress/"
            new_data_dir = "/home/zhiychen/Desktop/train_data/multiviewRGC/4d_dress/"
            takes = [i for i in range(11, 17)] + [19, 20]
            image_list = read_all_png_camera(data_dir, camera=int(cam_name), takes=takes)
            copy_png_to_folder(new_data_dir, image_list, camera=int(cam_name))
            color_img = cv.imread(os.path.join(self.data_dir, cam_name,
                                               "images", '%05d.png' % pose_idx), cv.IMREAD_UNCHANGED)
        mask_img = cv.imread(os.path.join(self.data_dir, cam_name, "masks", '%05d.png' % pose_idx), cv.IMREAD_UNCHANGED)

        # Use partial image
        # if self.layers is None:
        #     mask_img = cv.imread(os.path.join(self.data_dir, cam_name,
        #                                     "masks", '%05d.png' % pose_idx), cv.IMREAD_UNCHANGED)
        # else:
        #     selected_gray = []
        #     label_img = cv.imread(os.path.join(self.data_dir, cam_name,
        #                                     "labels", '%05d.png' % pose_idx), cv.IMREAD_GRAYSCALE)
        #     for layer in self.layers:
        #         selected_gray.append(self.masklabel[layer])
        #     mask_img = np.isin(label_img, selected_gray).astype(np.uint8) * 255
        return color_img, mask_img
