import math

import torch
import torch.nn.functional as F
from einops import repeat, rearrange
from mmcv.runner import BaseModule, force_fp32
from torch import nn
from torch.cuda.amp import autocast

from .height_net import HeightNet
from ..builder import NECKS


@NECKS.register_module()
class HeightVT(BaseModule):

    def __init__(self,
                 input_size,
                 n_voxels,
                 voxel_size,
                 grid_config,
                 in_channels,
                 out_channels,
                 downsample,
                 use_height,
                 use_semantic,
                 height_threshold=1,
                 semantic_threshold=0.25,
                 loss_height_weight=3,
                 loss_semantic_weight=25,
                 backproject='inplace',
                 multi_scale_3d_scaler='upsample',
                 **kwargs):
        super(HeightVT, self).__init__(**kwargs)

        self.input_size = input_size
        self.n_voxels = n_voxels
        self.voxel_size = voxel_size
        self.grid_config = grid_config
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.downsample = downsample
        self.use_height = use_height
        self.use_semantic = use_semantic
        self.backproject = backproject
        self.multi_scale_3d_scaler = multi_scale_3d_scaler

        self.H = len(torch.arange(*self.grid_config['height']))  # 10
        self.height_net = HeightNet(self.in_channels,
                                    self.in_channels,
                                    self.out_channels,
                                    self.H + 2)
        self.height_threshold = height_threshold / self.H
        self.semantic_threshold = semantic_threshold
        self.loss_height_weight = loss_height_weight
        self.loss_semantic_weight = loss_semantic_weight

        # self.fp16_enabled = False

    @autocast(enabled=False)
    def _compute_projection(self, rots, trans, intrinsics, post_rots, post_trans, bda, stride):
        """
        计算前视相机ego坐标系到图像坐标系转换
        已经过验证。
        Args:
            rots (Tensor): (6, 3, 3)
            trans (Tensor): (6, 3)
            intrinsics (): (6, 3, 3)
            post_rots (Tensor): (6, 3, 3)
            post_trans (Tensor): (6, 3)
            bda (Tensor): (3, 3)
            stride (Tensor): 16
        Returns:
            (6, 3, 4)
        """
        N, *_ = rots.shape  # 6

        scale = torch.eye(3, dtype=rots.dtype)[None].repeat(N, 1, 1).to(rots.device)
        scale[:, :2] /= stride

        cam2keyego = rots.new_zeros(N, 4, 4)
        cam2keyego[:, :3, :3] = rots
        cam2keyego[:, :3, 3] = trans
        cam2keyego[:, 3, 3] = 1

        post_rt = post_rots.new_zeros(N, 3, 3)
        post_rt[:, :3, :3] = post_rots
        post_rt[:, :3, 2] += post_trans

        bda_mat = rots.new_zeros(N, 4, 4)
        bda_mat[:, :3, :3] = bda
        bda_mat[:, 3, 3] = 1

        cam2img = scale @ post_rt @ intrinsics
        ego2cam = cam2keyego.inverse() @ bda_mat.inverse()

        return cam2img, ego2cam[:, :3]

    def view_transform(self, mlvl_feat, height, mask, cam_params, img_metas):
        # Alias
        b, n, *_ = cam_params[0].shape  # b, 6, [3, 3]
        stride_i = math.ceil(self.input_size[-1] / mlvl_feat.shape[-1])  # 4
        n_voxels, voxel_size = self.n_voxels, self.voxel_size  # [128, 128, 10], [0.8, 0.8, 0.8]
        points = get_points(  # (3, z, y, x)
            n_voxels=torch.tensor(n_voxels),
            voxel_size=torch.tensor(voxel_size),
            origin=torch.tensor(img_metas[0]["origin"]),
        ).to(mlvl_feat.device)

        # Rearrange
        mlvl_feat = rearrange(mlvl_feat, '(b n) c h w -> b n c h w', b=b, n=n)  # (b, 6, 80, 64, 176)
        height = rearrange(height, '(b n) c h w -> b n c h w', b=b, n=n)  # (b, 6, 10, 64, 176)
        mask = rearrange(mask, '(b n) c h w -> b n c h w', b=b, n=n)  # (b, 6, 1, 64, 176)

        # VT
        vox_feats = []
        for batch_id, img_meta in enumerate(img_metas):
            feat_i = mlvl_feat[batch_id]  # (n, c, h, w)
            height_i = height[batch_id] if self.use_height else None
            mask_i = mask[batch_id] if self.use_semantic else None
            cam_param_i = [p[batch_id] for p in cam_params]

            # Ego to cameras & camera to image.
            cam2img, ego2cam = self._compute_projection(*cam_param_i, stride_i)

            volume = self.backproject_inplace(feat_i, height_i, mask_i, points, cam2img, ego2cam)  # [c, dz, dy, dx]
            vox_feats.append(volume)  # bs x (c, dz, dy, dx)
        vox_feats = torch.stack(vox_feats, dim=0)  # (b, c, z, y, x)

        # Height processing
        bev_feats = rearrange(vox_feats, 'b c z y x -> b (c z) y x')

        return bev_feats

    def forward(self, img_inputs, img_metas):

        (x, rots, trans, intrins, post_rots, post_trans, bda,
         mlp_input, paste_idx, bda_paste) = img_inputs[:10]
        cam_params = img_inputs[1:7]

        # Alias
        x = rearrange(x, 'b n c h w -> (b n) c h w')  # (b * 6, 128, 64, 176)

        # Height & Semantic predict
        x = self.height_net(x, mlp_input)  # (b * 6, 92, 64, 176)

        # Alias
        height_digit = x[:, :self.H, ...]  # (b * 6, 10, 64, 176)
        semantic_digit = x[:, self.H:self.H + 2]  # (b * 6, 2, 64, 176)
        tran_feat = x[:, self.H + 2:self.H + 2 + self.out_channels, ...]  # (b * n, 80, 64, 176)
        height = height_digit.softmax(dim=1)
        semantic = semantic_digit.softmax(dim=1)
        mask = (height >= self.height_threshold)

        # Use semantic
        if self.use_semantic:
            mask = mask * (semantic[:, 1:2] >= self.semantic_threshold)

        return self.view_transform(tran_feat, height, mask, cam_params, img_metas), \
            (height, semantic)

    def get_downsampled_gt_height_and_semantic(self, gt_heights, gt_semantics):
        # remove point not in height range
        gt_semantics[gt_heights < self.grid_config['height'][0]] = 0
        gt_semantics[gt_heights > self.grid_config['height'][1]] = 0
        gt_heights[gt_heights < self.grid_config['height'][0]] = 0
        gt_heights[gt_heights > self.grid_config['height'][1]] = 0
        gt_semantic_heights = gt_heights * gt_semantics

        # 语义下采样
        B, N, H, W = gt_semantics.shape
        gt_semantics = gt_semantics.view(
            B * N,
            H // self.downsample,
            self.downsample,
            W // self.downsample,
            self.downsample,
            1,
        )
        gt_semantics = gt_semantics.permute(0, 1, 3, 5, 2, 4).contiguous()
        gt_semantics = gt_semantics.view(
            -1, self.downsample * self.downsample)
        gt_semantics = torch.max(gt_semantics, dim=-1).values
        gt_semantics = gt_semantics.view(B * N, H // self.downsample,
                                         W // self.downsample)
        gt_semantics = F.one_hot(gt_semantics.long(),
                                 num_classes=2).view(-1, 2).float()

        # 高度下采样
        B, N, H, W = gt_heights.shape
        gt_heights = gt_heights.view(
            B * N,
            H // self.downsample,
            self.downsample,
            W // self.downsample,
            self.downsample,
            1,
        )
        gt_heights = gt_heights.permute(0, 1, 3, 5, 2, 4).contiguous()
        gt_heights = gt_heights.view(
            -1, self.downsample * self.downsample)
        gt_heights_tmp = torch.where(gt_heights == 0.0,
                                     1e5 * torch.ones_like(gt_heights),
                                     gt_heights)
        gt_heights = torch.min(gt_heights_tmp, dim=-1).values
        gt_heights = gt_heights.view(B * N, H // self.downsample,
                                     W // self.downsample)
        gt_heights = (gt_heights -
                      (self.grid_config['height'][0] - self.grid_config['height'][2])) / self.grid_config['height'][2]
        gt_heights = torch.where(
            (gt_heights < self.H + 1) & (gt_heights >= 0.0),
            gt_heights, torch.zeros_like(gt_heights))
        gt_heights = F.one_hot(gt_heights.long(),
                               num_classes=self.H + 1).view(
            -1, self.H + 1)[:, 1:].float()

        # 语义高度下采样
        gt_semantic_heights = gt_semantic_heights.view(
            B * N,
            H // self.downsample,
            self.downsample,
            W // self.downsample,
            self.downsample,
            1,
        )
        gt_semantic_heights = gt_semantic_heights.permute(0, 1, 3, 5, 2, 4).contiguous()
        gt_semantic_heights = gt_semantic_heights.view(
            -1, self.downsample * self.downsample)
        gt_semantic_heights = torch.where(gt_semantic_heights == 0.0,
                                          1e5 * torch.ones_like(gt_semantic_heights),
                                          gt_semantic_heights)
        gt_semantic_heights = (gt_semantic_heights - (self.grid_config['height'][0] -
                                                      self.grid_config['height'][2])) / self.grid_config['height'][2]
        gt_semantic_heights = torch.where(
            (gt_semantic_heights < self.H + 1) & (gt_semantic_heights >= 0.0),
            gt_semantic_heights, torch.zeros_like(gt_semantic_heights)).long()
        soft_height_mask = gt_semantics[:, 1] > 0
        gt_semantic_heights = gt_semantic_heights[soft_height_mask]
        gt_semantic_heights_cnt = gt_semantic_heights.new_zeros([gt_semantic_heights.shape[0], self.H + 1])
        for i in range(self.H + 1):
            gt_semantic_heights_cnt[:, i] = (gt_semantic_heights == i).sum(dim=-1)
        gt_semantic_heights = gt_semantic_heights_cnt[:, 1:] / gt_semantic_heights_cnt[:, 1:].sum(dim=-1, keepdim=True)
        gt_heights[soft_height_mask] = gt_semantic_heights

        return gt_heights, gt_semantics

    @force_fp32()
    def get_height_and_semantic_loss(self, height_labels, height_preds, semantic_labels, semantic_preds):
        height_preds = height_preds.permute(0, 2, 3, 1).contiguous().view(-1, self.H)
        semantic_preds = semantic_preds.permute(0, 2, 3, 1).contiguous().view(-1, 2)
        semantic_weight = torch.zeros_like(semantic_labels[:, 1:2])
        semantic_weight = torch.fill_(semantic_weight, 0.1)
        semantic_weight[semantic_labels[:, 1] > 0] = 0.9

        height_mask = torch.max(height_labels, dim=1).values > 0.0
        height_labels = height_labels[height_mask]
        height_preds = height_preds[height_mask]
        semantic_labels = semantic_labels[height_mask]
        semantic_preds = semantic_preds[height_mask]
        semantic_weight = semantic_weight[height_mask]

        with autocast(enabled=False):
            height_loss = (F.binary_cross_entropy(
                height_preds,
                height_labels,
                reduction='none',
            ) * semantic_weight).sum() / max(0.1, semantic_weight.sum())

            pred = semantic_preds
            target = semantic_labels
            alpha = 0.25
            gamma = 2
            pt = (1 - pred) * target + pred * (1 - target)
            focal_weight = (alpha * target + (1 - alpha) *
                            (1 - target)) * pt.pow(gamma)
            semantic_loss = F.binary_cross_entropy(pred, target, reduction='none') * focal_weight
            semantic_loss = semantic_loss.sum() / max(1, len(semantic_loss))
        return self.loss_height_weight * height_loss, self.loss_semantic_weight * semantic_loss

    def get_loss(self, img_preds, gt_height, gt_semantic):
        height, semantic = img_preds
        height_labels, semantic_labels = \
            self.get_downsampled_gt_height_and_semantic(gt_height, gt_semantic)
        loss_height, loss_semantic = \
            self.get_height_and_semantic_loss(height_labels, height, semantic_labels, semantic)
        return loss_height, loss_semantic

    def map2bin(self, height):
        """
        高度离散化
        """
        # [-5, 3, 0.1]
        # 将height离散化到 [-5, 3, 0.1] 的bin中
        height = (height - (self.grid_config['height'][0] - self.grid_config['height'][2])) / \
                 self.grid_config['height'][2]
        return height.long()

    def backproject_inplace(self, features, height, mask, ego_coords, cam2img, ego2cam):
        # Alias
        n, c, Himg, Wimg = features.shape
        n_z, n_y, n_x = ego_coords.shape[-3:]
        ego_coords = ego_coords.view(1, 3, -1).expand(n, 3, -1)
        ego_coords = torch.cat((ego_coords, torch.ones_like(ego_coords[:, :1])), dim=1)

        # Ego to cameras.
        cam_coords = torch.bmm(ego2cam, ego_coords)
        height_bin = self.map2bin(cam_coords[:, 1, :])

        # Camera to image.
        points_2d_3 = torch.bmm(cam2img, cam_coords)
        x = (points_2d_3[:, 0] / points_2d_3[:, 2]).round().long()  # [6, 480000]
        y = (points_2d_3[:, 1] / points_2d_3[:, 2]).round().long()  # [6, 480000]
        z = points_2d_3[:, 2]  # [6, 480000]
        valid = (x >= 0) & (y >= 0) & (x < Wimg) & (y < Himg) & (z > 0) & (height_bin > 0) & (
                height_bin < self.H + 1)  # [6, 480000]

        if self.use_height:
            features = features[:, :, None, :, :] * height[:, None, :, :, :]
        else:
            features = repeat(features, 'n c h w -> n c n_height h w', n_height=self.H)

        if self.use_semantic:
            assert self.use_height, '[use_height] must be true.'
            features = features * mask[:, None, :, :, :]

        # VT
        vox_feats = torch.zeros(
            (c, ego_coords.shape[-1]), device=features.device
        ).type_as(features)
        for i in range(n):
            vox_feats[:, valid[i]] = features[i, :, height_bin[i, valid[i]] - 1, y[i, valid[i]], x[i, valid[i]]]

        vox_feats = rearrange(vox_feats, 'c (z y x) -> c z y x', c=c, z=n_z, y=n_y, x=n_x)

        return vox_feats

    def get_mlp_input(self, rot, tran, intrin, post_rot, post_tran, bda):
        B, N, _, _ = rot.shape
        bda = bda.view(B, 1, 3, 3).repeat(1, N, 1, 1)
        mlp_input = torch.stack([
            intrin[:, :, 0, 0],
            intrin[:, :, 1, 1],
            intrin[:, :, 0, 2],
            intrin[:, :, 1, 2],
            post_rot[:, :, 0, 0],
            post_rot[:, :, 0, 1],
            post_tran[:, :, 0],
            post_rot[:, :, 1, 0],
            post_rot[:, :, 1, 1],
            post_tran[:, :, 1],
            bda[:, :, 0, 0],
            bda[:, :, 0, 1],
            bda[:, :, 1, 0],
            bda[:, :, 1, 1],
            bda[:, :, 2, 2],
        ],
            dim=-1)
        sensor2ego = torch.cat([rot, tran.reshape(B, N, 3, 1)],
                               dim=-1).reshape(B, N, -1)
        mlp_input = torch.cat([mlp_input, sensor2ego], dim=-1)
        return mlp_input


@torch.no_grad()
def get_points(n_voxels, voxel_size, origin):
    """
    构建ego坐标系采点
    Args:
        n_voxels (torch.Tensor):
        voxel_size ():
        origin:
    Returns:
        (3, dy, dx, dz)
    """
    dz, dy, dx = torch.meshgrid(
        [
            torch.arange(n_voxels[2]),
            torch.arange(n_voxels[1]),
            torch.arange(n_voxels[0]),
        ]
    )
    points = torch.stack((dx, dy, dz))
    new_origin = origin - n_voxels / 2.0 * voxel_size + voxel_size / 2.0
    points = points * voxel_size.view(3, 1, 1, 1) + new_origin.view(3, 1, 1, 1)
    return points


def backproject_vanilla(features, points, projection):
    '''
    function: 2d feature + predefined point cloud -> 3d volume
    input:
        features: [6, 64, 225, 400]
        points: [3, 200, 200, 12]
        projection: [6, 3, 4]
    output:
        volume: [6, 64, 200, 200, 12]
        valid: [6, 1, 200, 200, 12]
    '''
    n_images, n_channels, height, width = features.shape
    n_z_voxels, n_y_voxels, n_x_voxels = points.shape[-3:]
    # [3, 200, 200, 12] -> [1, 3, 480000] -> [6, 3, 480000]
    points = points.view(1, 3, -1).expand(n_images, 3, -1)
    # [6, 3, 480000] -> [6, 4, 480000]
    points = torch.cat((points, torch.ones_like(points[:, :1])), dim=1)
    # ego_to_cam
    # [6, 3, 4] * [6, 4, 480000] -> [6, 3, 480000]
    points_2d_3 = torch.bmm(projection, points)  # lidar2img
    x = (points_2d_3[:, 0] / points_2d_3[:, 2]).round().long()  # [6, 480000]
    y = (points_2d_3[:, 1] / points_2d_3[:, 2]).round().long()  # [6, 480000]
    z = points_2d_3[:, 2]  # [6, 480000]
    valid = (x >= 0) & (y >= 0) & (x < width) & (y < height) & (z > 0)  # [6, 480000]
    volume = torch.zeros(
        (n_images, n_channels, points.shape[-1]), device=features.device
    ).type_as(features)  # [6, 64, 480000]
    for i in range(n_images):
        volume[i, :, valid[i]] = features[i, :, y[i, valid[i]], x[i, valid[i]]]
    # [6, 64, 480000] -> [6, 64, 200, 200, 12]
    volume = volume.view(n_images, n_channels, n_z_voxels, n_y_voxels, n_x_voxels)
    # [6, 480000] -> [6, 1, 200, 200, 12]
    valid = valid.view(n_images, 1, n_z_voxels, n_y_voxels, n_x_voxels)
    return volume, valid


class SampledCoordSelector(nn.Module):
    """前视相机ego坐标系中采点
    """

    def __init__(self,
                 pc_range,
                 grid_config,
                 sampled_kwargs):
        super(SampledCoordSelector, self).__init__()

        self.pc_range = pc_range
        self.grid_config = grid_config

        # Init
        self._init_status(sampled_kwargs)
        self._init_buffer()

    def _init_status(self, sampled_kwargs):
        # Coarse
        self.n_coarse = sampled_kwargs['n_coarse']

        # Fine
        self.n_fine = sampled_kwargs['n_fine']

    def _init_buffer(self):
        X, Y, H = self.grid_config
        grid = torch.stack(  # (Y, X, 2)
            torch.meshgrid(
                torch.linspace(0, 1, X), torch.linspace(0, 1, Y), indexing='xy'
            ),
            dim=-1,
        )
        self.register_buffer('grid_buffer', grid)

    def get_coarse_coords(self, bt, device):
        X, Y, H = self.grid_config
        sb = self.pc_range
        n_coarse = self.n_coarse
        grid = self.grid_buffer

        pillars = repeat(grid, 'x y c -> b (x y) c', b=bt)
        rnd = torch.randperm(X * Y)[:n_coarse].to(device)
        pillars = torch.index_select(pillars, dim=1, index=rnd)

        # lift
        n_xy = pillars.size(1)
        pillars = repeat(pillars, 'bt xy c -> bt c (xy h)', h=H)

        # -> Regular Z points
        pillar_heights = torch.linspace(0, 1, H, device=device)
        pillar_heights = repeat(pillar_heights, "h -> bt 1 (xy h)", bt=bt, xy=n_xy)

        # Pillar pts: [0,1]
        pillar_pts = torch.cat([pillars, pillar_heights], dim=1)

        # Voxel coordinates: [-BoundMin, BoundMax]
        scale = torch.tensor(
            [sb[1] - sb[0], sb[3] - sb[2], sb[5] - sb[4]], device=device
        ).view(1, 3, 1)
        dist = torch.tensor([abs(sb[0]), abs(sb[2]), abs(sb[4])], device=device).view(
            1, 3, 1
        )
        vox_coords = pillar_pts * scale - dist

        # Voxel indices: [0,X-1]
        xyz = torch.tensor([X - 1, Y - 1, H - 1], device=device).view(1, 3, 1)
        vox_idx = (pillar_pts * xyz).round().to(torch.int32)

        return vox_coords, vox_idx

    def get_fine_coords(self, out, masks):
        pass
