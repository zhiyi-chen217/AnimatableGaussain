import platform
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pytorch3d.ops
import pytorch3d.transforms
import cv2 as cv
from poetry.console.commands import self
from sympy import rotations

import config
from network.avatar import AvatarNet
from network.styleunet.dual_styleunet import DualStyleUNet
from gaussians.gaussian_model import GaussianModel
from gaussians.gaussian_renderer import render3

class MultiGaussianModel:
    def __init__(self, model_dict, layers):
        self.cano_gaussian_dict = {}
        self.layers = layers
        for layer in layers:
            self.cano_gaussian_dict[layer] = model_dict[layer].cano_gaussian_model
        

    @property
    def get_scaling(self):
        scaling_l = []
        for layer in self.layers:
            scaling_l.append(self.cano_gaussian_dict[layer].get_scaling)
        return torch.concat(scaling_l, dim=0)

    @property
    def get_rotation(self):
        rotation_l = []
        for layer in self.layers:
            rotation_l.append(self.cano_gaussian_dict[layer].get_rotation)
        return torch.concat(rotation_l, dim=0)

    @property
    def get_xyz(self):
        xyz_l = []
        for layer in self.layers:
            xyz_l.append(self.cano_gaussian_dict[layer].get_xyz)
        return torch.concat(xyz_l, dim=0)

    @property
    def get_opacity(self):
        opacity_l = []
        for layer in self.layers:
            opacity_l.append(self.cano_gaussian_dict[layer].get_opacity)
        return torch.concat(opacity_l, dim=0)


class MultiLAvatarNet(nn.Module):
    def __init__(self, opt, layers):
        super(MultiLAvatarNet, self).__init__()
        self.layers = layers
        self.layers_nn = nn.ModuleDict()
        self.init_points = []
        self.lbs = []
        self.smpl_pos_map = config.opt.get("smpl_pos_map", "smpl_pos_map") + "_smplx"
        for layer in layers:
            self.layers_nn[layer] = AvatarNet(opt, layer)
            self.init_points.append(self.layers_nn[layer].cano_smpl_map[self.layers_nn[layer].cano_smpl_mask])
            self.lbs.append(self.layers_nn[layer].lbs)
        self.init_points = torch.concat(self.init_points, dim=0)
        self.lbs = torch.concat(self.lbs, dim=0)
        self.max_sh_degree = 0
        self.cano_gaussian_model = MultiGaussianModel(self.layers_nn, self.layers)



    def transform_cano2live(self, gaussian_vals, lbs, items):
        pt_mats = torch.einsum('nj,jxy->nxy', lbs, items['cano2live_jnt_mats'])
        gaussian_vals['positions'] = torch.einsum('nxy,ny->nx', pt_mats[..., :3, :3], gaussian_vals['positions']) + pt_mats[..., :3, 3]
        rot_mats = pytorch3d.transforms.quaternion_to_matrix(gaussian_vals['rotations'])
        rot_mats = torch.einsum('nxy,nyz->nxz', pt_mats[..., :3, :3], rot_mats)
        gaussian_vals['rotations'] = pytorch3d.transforms.matrix_to_quaternion(rot_mats)
        return gaussian_vals

    def get_lbs(self, layers=None):
        if layers is None:
            return self.lbs
        layer_lbs = []
        for layer in layers:
            layer_lbs.append(self.layers_nn[layer].lbs)
        return torch.stack(layer_lbs, dim=0)

    def render_body(self, items, bg_color = (0., 0., 0.), use_pca = False, use_vae = False, with_body=False, layers=None):
        """
        Note that no batch index in items.
        """
        # body model
        self.cano_smpl_body_gaussian_model = GaussianModel(sh_degree=self.max_sh_degree)
        cano_smpl_body_map = cv.imread(config.opt['train']['data']['data_dir'] + '/{}/cano_smpl_pos_map.exr'
                                       .format(self.smpl_pos_map), cv.IMREAD_UNCHANGED)
        self.cano_smpl_body_map = torch.from_numpy(cano_smpl_body_map).to(torch.float32).to(config.device)
        self.cano_smpl_body_mask = torch.linalg.norm(self.cano_smpl_body_map, dim=-1) > 0.

        self.smpl_body_init_points = self.cano_smpl_body_map[self.cano_smpl_body_mask]
        self.smpl_body_lbs = torch.from_numpy(np.load(config.opt['train']['data']['data_dir'] + '/{}/init_pts_lbs.npy'
                                                      .format(self.smpl_pos_map))).to(torch.float32).to(config.device)
        self.cano_smpl_body_gaussian_model.create_from_pcd(self.smpl_body_init_points,
                                                           torch.rand_like(self.smpl_body_init_points),
                                                           spatial_lr_scale=2.5)
        bg_color = torch.from_numpy(np.asarray(bg_color)).to(torch.float32).to(config.device)

        gaussian_body_vals = {
            'positions': self.cano_smpl_body_gaussian_model.get_xyz,
            'opacity': self.cano_smpl_body_gaussian_model.get_opacity,
            'scales': self.cano_smpl_body_gaussian_model.get_scaling,
            'rotations': self.cano_smpl_body_gaussian_model.get_rotation,
            'colors': torch.ones(self.cano_smpl_body_gaussian_model.get_xyz.shape),
            'max_sh_degree': self.max_sh_degree
        }
        gaussian_body_vals = self.transform_cano2live(gaussian_body_vals, self.smpl_body_lbs, items)


        render_ret = render3(
            gaussian_body_vals,
            bg_color,
            items['extr'],
            items['intr'],
            items['img_w'],
            items['img_h']
        )
        rgb_map = render_ret['render'].permute(1, 2, 0)
        mask_map = render_ret['mask'].permute(1, 2, 0)

        ret = {
            'rgb_map': rgb_map,
            'mask_map': mask_map,
        }


        return ret

    def render(self, items, bg_color = (0., 0., 0.), use_pca = False, use_vae = False, with_body=False, layers=None):
        """
        Note that no batch index in items.
        """
        bg_color = torch.from_numpy(np.asarray(bg_color)).to(torch.float32).to(config.device)
        pose_map = {}
        for layer in self.layers:
            pose_map_layer = items['smpl_pos_map'][layer].squeeze(0)[:3]
            assert not (use_pca and use_vae), "Cannot use both PCA and VAE!"
            if use_pca:
                pose_map_layer = items[layer]['smpl_pos_map_pca'].squeeze(0)[:3]
            if use_vae:
                pose_map_layer = items[layer]['smpl_pos_map_vae'].squeeze(0)[:3]
            pose_map[layer] = pose_map_layer
        items['smpl_pos_map'] = pose_map


        cano_pts, pos_map = self.get_positions(pose_map, return_map = True, layers=layers)
        opacity, scales, rotations = self.get_others(pose_map, layers=layers)
        colors, color_map = self.get_colors(pose_map, items, layers=layers)

        gaussian_vals = {
            'positions': cano_pts,
            'opacity': opacity,
            'scales': scales,
            'rotations': rotations,
            'colors': colors,
            'max_sh_degree': self.max_sh_degree
        }

        nonrigid_offset = gaussian_vals['positions'] - self.init_points

        gaussian_vals = self.transform_cano2live(gaussian_vals, self.get_lbs(layers), items)

        render_ret = render3(
            gaussian_vals,
            bg_color,
            items['extr'],
            items['intr'],
            items['img_w'],
            items['img_h']
        )
        rgb_map = render_ret['render'].permute(1, 2, 0)
        mask_map = render_ret['mask'].permute(1, 2, 0)

        ret = {
            'rgb_map': rgb_map,
            'mask_map': mask_map,
            'offset': nonrigid_offset,
            'pos_map': pos_map
        }

        gaussian_offset_vals = {
            'positions': cano_pts,
            'opacity': opacity,
            'scales': scales,
            'rotations': rotations,
            'colors': torch.abs(nonrigid_offset) * 5,
            'max_sh_degree': self.max_sh_degree
        }
        gaussian_offset_vals = self.transform_cano2live(gaussian_offset_vals, items)

        render_offset_ret = render3(
            gaussian_offset_vals,
            bg_color,
            items['extr'],
            items['intr'],
            items['img_w'],
            items['img_h']
        )

        ret.update({
            "offset_map":  render_offset_ret['render'].permute(1, 2, 0)
        })

        if not self.training:
            ret.update({
                'cano_tex_map': color_map,
                'posed_gaussians': gaussian_vals
            })

        return ret
    def get_positions(self, pose_map, return_map = False, layers=None):
        if layers is None:
            layers = self.layers
        if return_map:
            positions_l = []
            position_map_l = []
            for layer in layers:
                pose_map_layer = pose_map[layer]
                positions, position_map = self.layers_nn[layer].get_positions(pose_map_layer, return_map)
                positions_l.append(positions)
                position_map_l.append(position_map)
            return torch.concat(positions_l, dim=0), torch.concat(position_map_l, dim=0)
        else:
            positions_l = []
            for layer in layers:
                pose_map_layer = pose_map[layer]
                positions = self.layers_nn[layer].get_positions(pose_map_layer, return_map)
                positions_l.append(positions)

            return torch.concat(positions_l, dim=0)

    def get_others(self, pose_map, layers=None):
        opacity_l = []
        scales_l = []
        rotations_l = []
        if layers is None:
            layers = self.layers
        for layer in layers:
            opacity, scales, rotations = self.layers_nn[layer].get_others(pose_map[layer])
            opacity_l.append(opacity)
            scales_l.append(scales)
            rotations_l.append(rotations)
        return torch.concat(opacity_l, dim=0), torch.concat(scales_l, dim=0), torch.concat(rotations_l, dim=0)

    def get_colors(self, pose_map, items, layers=None):
        colors_l = []
        color_map_l = []
        if layers is None:
            layers = self.layers
        for layer in layers:
            if self.layers_nn[layer].with_viewdirs:
                front_viewdirs, back_viewdirs = self.layers_nn[layer].get_viewdir_feat(items)
            else:
                front_viewdirs, back_viewdirs = None, None
            colors, color_map = self.layers_nn[layer].get_colors(pose_map[layer],
                                                                 front_viewdirs, back_viewdirs)
            colors_l.append(colors)
            color_map_l.append(color_map)
        return torch.concat(colors_l, dim=0), torch.concat(color_map_l, dim=0)


