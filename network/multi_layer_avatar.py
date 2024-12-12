import platform
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pytorch3d.ops
import pytorch3d.transforms
import cv2 as cv
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
        self.upper_body_mask = self.layers_nn["body"].cano_smpl_mask & (~self.layers_nn["cloth"].cano_smpl_mask)
        self.selected_body_gaussian = self.upper_body_mask[self.layers_nn["body"].cano_smpl_mask]
        self.upper_cloth_mask = self.layers_nn["cloth"].cano_smpl_mask & self.layers_nn["body"].cano_smpl_mask
        self.selected_cloth_gaussian = self.upper_cloth_mask[self.layers_nn["cloth"].cano_smpl_mask]


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
        return torch.concat(layer_lbs, dim=0)
    
    def get_init_points(self, layers=None):
        if layers is None:
            return self.init_points
        layer_init_points = []
        for layer in layers:
            layer_init_points.append(self.layers_nn[layer].cano_smpl_map[self.layers_nn[layer].cano_smpl_mask])
        return torch.concat(layer_init_points, dim=0)
    
    def render(self, items, bg_color = (0., 0., 0.), use_pca = False, use_vae = False):
        """
        Note that no batch index in items.
        """
        bg_color = torch.from_numpy(np.asarray(bg_color)).to(torch.float32).to(config.device)
        for layer in self.layers:
            items["smpl_pos_map"][layer] = items["smpl_pos_map"][layer].squeeze(0)
    
        gaussian_body_vals = self.layers_nn["body"].render(items, only_gaussian=True)
        gaussian_cloth_vals = self.layers_nn["cloth"].render(items, only_gaussian=True)
        gaussian_vals = {}
        for key in gaussian_cloth_vals.keys():
            if key == "max_sh_degree":
                gaussian_vals[key] = gaussian_cloth_vals[key]
            elif key == "offset":
                gaussian_vals[key] = torch.concat([gaussian_body_vals[key], gaussian_cloth_vals[key]], dim=0)
            else:
                gaussian_vals[key] = torch.concat([gaussian_body_vals[key][self.selected_body_gaussian], gaussian_cloth_vals[key]], dim=0)

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
            'offset': gaussian_vals["offset"],
            "gaussian_cloth_pos": gaussian_cloth_vals["cano_positions"][self.selected_cloth_gaussian],
            "gaussian_body_pos": gaussian_body_vals["cano_positions"],
            "gaussian_body_norm": gaussian_body_vals["gaussian_norm"]
        }

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


