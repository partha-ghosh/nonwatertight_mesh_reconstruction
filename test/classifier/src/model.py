import torch
import numpy as np
import time
from src.utils import point_rasterize, grid_interp, mc_from_psr, \
calc_inters_points
from src.dpsr import DPSR
import torch.nn as nn
from src.network import encoder_dict, decoder_dict
from src.network.utils import map2local
import torchvision.transforms.functional as TF

class PSR2Mesh(torch.autograd.Function):

    @staticmethod
    def use_mask(mask):
        PSR2Mesh.mask = mask

    @staticmethod
    def forward(ctx, psr_grid):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        verts, faces, normals = mc_from_psr(psr_grid, pytorchify=True, mask=PSR2Mesh.mask)
        PSR2Mesh.mask=None
        verts = verts.unsqueeze(0)
        faces = faces.unsqueeze(0)
        normals = normals.unsqueeze(0)

        res = torch.tensor(psr_grid.detach().shape[2])
        ctx.save_for_backward(verts, normals, res)

        return verts, faces, normals

    @staticmethod
    def backward(ctx, dL_dVertex, dL_dFace, dL_dNormals):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        vert_pts, normals, res = ctx.saved_tensors
        res = (res.item(), res.item(), res.item())
        # matrix multiplication between dL/dV and dV/dPSR
        # dV/dPSR = - normals
        grad_vert = torch.matmul(dL_dVertex.permute(1, 0, 2), -normals.permute(1, 2, 0))
        grad_grid = point_rasterize(vert_pts, grad_vert.permute(1, 0, 2), res) # b x 1 x res x res x res
        
        return grad_grid

class PSR2SurfacePoints(torch.autograd.Function):
    @staticmethod
    def forward(ctx, psr_grid, poses, img_size, uv, psr_grad, mask_sample):
        verts, faces, normals = mc_from_psr(psr_grid, pytorchify=True)
        verts = verts * 2. - 1. # within the range of [-1, 1]

        
        p_all, n_all, mask_all = [], [], []

        for i in range(len(poses)):
            pose = poses[i]
            if mask_sample is not None:
                p_inters, mask, _, _ = calc_inters_points(verts, faces, pose, img_size, mask_gt=mask_sample[i])
            else:
                p_inters, mask, _, _ = calc_inters_points(verts, faces, pose, img_size)

            n_inters = grid_interp(psr_grad[None], (p_inters[None].detach() + 1) / 2).squeeze()
            p_all.append(p_inters)
            n_all.append(n_inters)
            mask_all.append(mask)
        p_inters_all = torch.cat(p_all, dim=0)
        n_inters_all = torch.cat(n_all, dim=0)
        mask_visible = torch.stack(mask_all, dim=0)


        res = torch.tensor(psr_grid.detach().shape[2])
        ctx.save_for_backward(p_inters_all, n_inters_all, res)

        return p_inters_all, mask_visible

    @staticmethod
    def backward(ctx, dL_dp, dL_dmask):
        pts, pts_n, res = ctx.saved_tensors
        res = (res.item(), res.item(), res.item())

        # grad from the p_inters via MLP renderer
        grad_pts = torch.matmul(dL_dp[:, None], -pts_n[..., None])
        grad_grid_pts = point_rasterize((pts[None]+1)/2, grad_pts.permute(1, 0, 2), res) # b x 1 x res x res x res
        
        return grad_grid_pts, None, None, None, None, None

class Encode2Points(nn.Module):
    def __init__(self):
        super().__init__()
        cfg = {'data': {'dataset': 'Shapes3D', 'path': 'data/demo/custom', 'class': [''], 'data_type': 'psr_full', 'input_type': 'pointcloud', 'dim': 3, 'num_points': 1000, 'num_gt_points': 10000, 'num_offset': 7, 'img_size': None, 'n_views_input': 20, 'n_views_per_iter': 2, 'pointcloud_noise': 0.005, 'pointcloud_file': 'pointcloud.npz', 'pointcloud_outlier_ratio': 0.0, 'fixed_scale': 0, 'train_split': 'train', 'val_split': 'val', 'test_split': 'test', 'points_file': None, 'points_iou_file': 'points.npz', 'points_unpackbits': True, 'padding': 0.1, 'multi_files': None, 'gt_mesh': None, 'zero_level': 0, 'only_single': False, 'sample_only_floor': False, 'pointcloud_n': 3000}, 'model': {'apply_sigmoid': True, 'grid_res': 128, 'psr_sigma': 2, 'psr_tanh': False, 'normal_normalize': False, 'raster': {}, 'renderer': {}, 'encoder': 'local_pool_pointnet', 'predict_normal': True, 'predict_offset': True, 's_offset': 0.001, 'local_coord': True, 'encoder_kwargs': {'hidden_dim': 32, 'plane_type': 'grid', 'grid_resolution': 32, 'unet3d': True, 'unet3d_kwargs': {'num_levels': 3, 'f_maps': 32, 'in_channels': 32, 'out_channels': 32}}, 'unet3d': False, 'unet3d_kwargs': {}, 'multi_gpu': False, 'rotate_matrix': False, 'c_dim': 32, 'sphere_radius': 0.2, 'decoder': 'simple_local', 'decoder_kwargs': {'sample_mode': 'bilinear', 'hidden_size': 32}}, 'train': {'lr': '1e-4', 'lr_pcl': '2e-2', 'input_mesh': '', 'out_dir': 'out/demo_shapenet_outlier', 'subsample_vertex': False, 'batch_size': 8, 'n_grow_points': 0, 'resample_every': 0, 'l_weight': {}, 'w_reg_point': 0, 'w_psr': 1, 'w_raw': 0, 'gauss_weight': 0, 'n_sup_point': 0, 'w_normals': 0, 'total_epochs': 400000, 'print_every': 10, 'visualize_every': 1000, 'save_every': 1000, 'vis_vert_color': True, 'o3d_show': False, 'o3d_vis_pcl': True, 'o3d_window_size': 540, 'vis_rendering': False, 'vis_psr': False, 'save_video': False, 'exp_mesh': True, 'exp_pcl': True, 'checkpoint_every': 200, 'validate_every': 500, 'backup_every': 10000, 'timestamp': False, 'model_selection_metric': 'psr_l2+dice_loss', 'model_selection_mode': 'minimize', 'n_workers': 8, 'n_workers_val': 0, 'dir_model': 'out/demo_shapenet_outlier/model', 'dir_mesh': 'out/demo_shapenet_outlier/vis/mesh', 'dir_pcl': 'out/demo_shapenet_outlier/vis/pointcloud'}, 'test': {'threshold': 0.5, 'eval_mesh': True, 'eval_pointcloud': False, 'model_file': 'https://s3.eu-central-1.amazonaws.com/avg-projects/shape_as_points/models/ours_outlier_7x.pt'}, 'generation': {'batch_size': 100000, 'exp_gt': False, 'exp_oracle': False, 'exp_input': True, 'vis_n_outputs': 10, 'generate_mesh': True, 'generate_pointcloud': True, 'generation_dir': 'generation', 'copy_input': True, 'use_sampling': False, 'psr_resolution': 128, 'psr_sigma': 2}, 'inherit_from': 'configs/learning_based/outlier/ours_7x.yaml'}
        self.cfg = cfg

        encoder = cfg['model']['encoder']
        decoder = cfg['model']['decoder']
        dim = cfg['data']['dim'] # input dim
        c_dim = cfg['model']['c_dim']
        encoder_kwargs = cfg['model']['encoder_kwargs']
        if encoder_kwargs == None:
            encoder_kwargs = {}
        decoder_kwargs = cfg['model']['decoder_kwargs']
        padding = cfg['data']['padding']
        self.predict_normal = cfg['model']['predict_normal']
        self.predict_offset = cfg['model']['predict_offset']

        out_dim = 3
        out_dim_offset = 3
        num_offset = cfg['data']['num_offset']
        # each point predict more than one offset to add output points
        if num_offset > 1:
            out_dim_offset = out_dim * num_offset
        self.num_offset = num_offset

        # local mapping
        self.map2local = None
        if cfg['model']['local_coord']:
            if 'unet' in encoder_kwargs.keys():
                unit_size = 1 / encoder_kwargs['plane_resolution']
            else:
                unit_size = 1 / encoder_kwargs['grid_resolution']
            
            local_mapping = map2local(unit_size)

        self.encoder = encoder_dict[encoder](
            dim=dim, c_dim=c_dim, map2local=local_mapping,
            **encoder_kwargs
        )

        if self.predict_normal:
            # decoder for normal prediction
            self.decoder_normal = decoder_dict[decoder](
                dim=dim, c_dim=c_dim, out_dim=out_dim,
                **decoder_kwargs)
        if self.predict_offset:
            # decoder for offset prediction
            self.decoder_offset = decoder_dict[decoder](
                dim=dim, c_dim=c_dim, out_dim=out_dim_offset,
                map2local=local_mapping,
                **decoder_kwargs)

            self.s_off = cfg['model']['s_offset']
        
        
    def forward(self, p):
        ''' Performs a forward pass through the network.

        Args:
            p (tensor): input unoriented points
        '''

        time_dict = {}
        mask = None
        
        batch_size = p.size(0)
        points = p.clone()

        # encode the input point cloud to a feature volume
        t0 = time.perf_counter()
        c = self.encoder(p)
        return c
        t1 = time.perf_counter()
        if self.predict_offset:
            offset = self.decoder_offset(p, c)
            # more than one offset is predicted per-point
            if self.num_offset > 1:
                points = points.repeat(1, 1, self.num_offset).reshape(batch_size, -1, 3)
            points = points + self.s_off * offset
        else:
            points = p

        if self.predict_normal:
            normals = self.decoder_normal(points, c)
        t2 = time.perf_counter()
        
        time_dict['encode'] = t1 - t0
        time_dict['predict'] = t2 - t1
        
        points = torch.clamp(points, 0.0, 0.99)
        if self.cfg['model']['normal_normalize']:
            normals = normals / (normals.norm(dim=-1, keepdim=True)+1e-8)
       
        return points, normals
    
class DoubleConv2d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv2d, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class UNET2D(nn.Module):
    def __init__(
            self, in_channels=3, out_channels=1, features=[64, 128, 256, 512],
    ):
        super(UNET2D, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down part of UNET
        for feature in features:
            self.downs.append(DoubleConv2d(in_channels, feature))
            in_channels = feature

        # Up part of UNET
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature*2, feature, kernel_size=2, stride=2,
                )
            )
            self.ups.append(DoubleConv2d(feature*2, feature))

        self.bottleneck = DoubleConv2d(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)

        return self.final_conv(x)

class DoubleConv3d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv3d, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class UNET3D(nn.Module):
    def __init__(
            self, in_channels=1, out_channels=1, features=[32, 64, 128, 256, 512],
    ):
        super(UNET3D, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

        # Down part of UNET
        for feature in features:
            self.downs.append(DoubleConv3d(in_channels, feature))
            in_channels = feature

        # Up part of UNET
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose3d(
                    feature*2, feature, kernel_size=2, stride=2,
                )
            )
            self.ups.append(DoubleConv3d(feature*2, feature))

        self.bottleneck = DoubleConv3d(features[-1], features[-1]*2)
        self.final_conv = nn.Conv3d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)

        return torch.sigmoid(torch.clip(self.final_conv(x),-5,5))

