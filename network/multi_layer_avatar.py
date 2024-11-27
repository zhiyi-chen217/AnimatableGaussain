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




class MultiLAvatarNet(nn.Module):
    def __init__(self, opt, layers):
        super(MultiLAvatarNet, self).__init__()
        self.layers = layers
        self.layers_nn = nn.ModuleDict()
        self.init_points = []
        for layer in layers:
            self.layers_nn[layer] = AvatarNet(opt, layer)
            self.init_points.append(self.layers_nn[layer].cano_smpl_map[self.layers_nn[layer].cano_smpl_mask])
        self.init_points = torch.concat(self.init_points, dim=0)
        self.cano_gaussian_model = MultiGaussianModel(self.layers_nn, self.layers)

    def render(self, items, bg_color = (0., 0., 0.), use_pca = False, use_vae = False):
        """
        Note that no batch index in items.
        """
        bg_color = torch.from_numpy(np.asarray(bg_color)).to(torch.float32).to(config.device)
        pose_map = {}
        for layer in self.layers:
            pose_map_layer = items['smpl_pos_map'][layer][:3]
            assert not (use_pca and use_vae), "Cannot use both PCA and VAE!"
            if use_pca:
                pose_map_layer = items[layer]['smpl_pos_map_pca'][:3]
            if use_vae:
                pose_map_layer = items[layer]['smpl_pos_map_vae'][:3]
            pose_map[layer] = pose_map_layer


        cano_pts, pos_map = self.get_positions(pose_map, return_map = True)
        opacity, scales, rotations = self.get_others(pose_map)
        colors, color_map = self.get_colors(pose_map, items)

        gaussian_vals = {
            'positions': cano_pts,
            'opacity': opacity,
            'scales': scales,
            'rotations': rotations,
            'colors': colors,
            'max_sh_degree': self.max_sh_degree
        }

        nonrigid_offset = gaussian_vals['positions'] - self.init_points

        gaussian_vals = self.transform_cano2live(gaussian_vals, items)

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
    def get_positions(self, pose_map, return_map = False):
        if return_map:
            positions_l = []
            position_map_l = []
            for layer in self.layers:
                pose_map_layer = pose_map[layer]
                positions, position_map = self.layers_nn[layer].get_positions(pose_map_layer, return_map)
                positions_l.append(positions)
                position_map_l.append(position_map)
            return torch.concat(positions_l, dim=0), torch.concat(position_map_l, dim=0)
        else:
            positions_l = []
            for layer in self.layers:
                pose_map_layer = pose_map[layer]
                positions = self.layers_nn[layer].get_positions(pose_map_layer, return_map)
                positions_l.append(positions)

            return torch.concat(positions_l, dim=0)

    def get_others(self, pose_map):
        opacity_l = []
        scales_l = []
        rotations_l = []
        for layer in self.layers:
            opacity, scales, rotations = self.layers_nn[layer].get_others(pose_map[layer])
            opacity_l.append(opacity)
            scales_l.append(scales)
            rotations_l.append(rotations)
        return torch.concat(opacity_l, dim=0), torch.concat(scales_l, dim=0), torch.concat(rotations_l, dim=0)

    def get_colors(self, pose_map, items):
        colors_l = []
        color_map_l = []
        for layer in self.layers:
            if self.with_viewdirs:
                front_viewdirs, back_viewdirs = self.layers_nn[layer].get_viewdir_feat(items)
            else:
                front_viewdirs, back_viewdirs = None, None
            colors, color_map = self.layers_nn[layer].get_colors(pose_map[layer],
                                                                 front_viewdirs, back_viewdirs)
            colors_l.append(colors)
            color_map_l.append(color_map)
        return torch.concat(colors_l, dim=0), torch.concat(color_map_l, dim=0)


