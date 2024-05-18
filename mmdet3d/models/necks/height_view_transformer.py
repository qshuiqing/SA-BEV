import torch
import torch.nn.functional as F
from einops import rearrange
from mmcv.runner import BaseModule, force_fp32
from torch.cuda.amp import autocast

from .height_net import HeightNet
from ..builder import NECKS
from ..heightbev.projector import CamProjector
from ..heightbev.sampled import SampledCoordSelector


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
                 height_threshold=1,
                 semantic_threshold=0.25,
                 loss_height_weight=3,
                 loss_semantic_weight=25,
                 backproject='inplace',
                 multi_scale_3d_scaler='upsample',
                 sampled_kwargs=None,
                 **kwargs):
        super(HeightVT, self).__init__(**kwargs)

        self.input_size = input_size
        self.grid_config = grid_config
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.downsample = downsample
        self.backproject = backproject
        self.multi_scale_3d_scaler = multi_scale_3d_scaler

        self.H = len(torch.arange(*self.grid_config['z_bounds']))  # 10
        self.height_net = HeightNet(self.in_channels,
                                    self.in_channels,
                                    self.out_channels,
                                    self.H + 2)
        self.height_threshold = height_threshold / self.H
        self.semantic_threshold = semantic_threshold
        self.loss_height_weight = loss_height_weight
        self.loss_semantic_weight = loss_semantic_weight

        # Random sample points
        self.sampled_kwargs = sampled_kwargs

        self.coord_selector = SampledCoordSelector(sampled_kwargs)
        self.projector = CamProjector(sampled_kwargs)

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

    def forward_height(self, vox_feats):

        vox_feats = rearrange(vox_feats, 'b c z y x -> b (c z) y x')

        return vox_feats

    def view_transform(self, mlvl_feat, cam_params, img_metas):
        """
        2D-to-3D视图转换
        Args:
            mlvl_feat:
            cam_params:
            img_metas:
        Returns:
            bev_feat:
        """

        # Prepare
        dict_shape = {
            'Himg': self.input_size[0],
            'Wimg': self.input_size[1],
            'Hfeat': mlvl_feat.shape[-2],
            'Wfeat': mlvl_feat.shape[-1],
        }

        vox_feats = []
        for batch_id, feat_i in enumerate(mlvl_feat):
            cam_param = [p[batch_id] for p in cam_params]

            # Random sample points
            voxel_coords, voxel_mask = self.coord_selector.get_coarse_coords(dict_shape)  # (3, 10, 256, 256)

            # Project from ego to img
            voxel_cam_coords, voxel_cam_heights, voxel_cam_valid = self.projector(voxel_coords,
                                                                                  cam_param,
                                                                                  dict_shape)

            # VT
            voxel_feat_i = self.bp_projection(feat_i, voxel_cam_coords,
                                              voxel_cam_heights, voxel_cam_valid, dict_shape)  # [c, dz, dy, dx]
            vox_feats.append(voxel_feat_i)  # bs x (c, dz, dy, dx)
        vox_feats = torch.stack(vox_feats, dim=0)  # (b, c, z, y, x)

        # Height processing
        bev_feats = self.forward_height(vox_feats)  # (b, c*z, y, x)

        return bev_feats

    def forward(self, img_inputs, img_metas):

        (x, rots, trans, intrins, post_rots, post_trans, bda,
         mlp_input, paste_idx, bda_paste) = img_inputs[:10]
        cam_params = img_inputs[1:7]

        # 最后一层高度预测
        x = x[0]  # (bs, n, c, 32, 88)
        B, N, C, H, W = x.shape
        x = x.view(B * N, C, H, W)
        x = self.height_net(x, mlp_input)

        height_digit = x[:, :self.H, ...]  # (bs*n, 10, 32, 88)
        semantic_digit = x[:, self.H:self.H + 2]  # (bs*n, 2, 32, 88)
        tran_feat = x[:, self.H + 2:self.H + 2 + self.out_channels, ...]  # (bs*n, 256, 32, 88)

        height = height_digit.softmax(dim=1)
        semantic = semantic_digit.softmax(dim=1)
        kept = (height >= self.height_threshold) * (semantic[:, 1:2] >= self.semantic_threshold)  # (24, 10, 32, 88)

        # (bs*n, 256, 10, 32, 88)
        tran_height_feat = tran_feat[:, :, None, :, :] * height[:, None, :, :, :] * kept[:, None, :, :, :]
        tran_height_feat = tran_height_feat.view([B, N] + list(tran_height_feat.shape[1:]))

        return self.view_transform(tran_height_feat, cam_params, img_metas), \
            (height, semantic)

    def get_downsampled_gt_height_and_semantic(self, gt_heights, gt_semantics):
        # remove point not in height range
        gt_semantics[gt_heights < self.grid_config['z_bounds'][0]] = 0
        gt_semantics[gt_heights > self.grid_config['z_bounds'][1]] = 0
        gt_heights[gt_heights < self.grid_config['z_bounds'][0]] = 0
        gt_heights[gt_heights > self.grid_config['z_bounds'][1]] = 0
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
                      (self.grid_config['z_bounds'][0] - self.grid_config['z_bounds'][2])) / \
                     self.grid_config['z_bounds'][2]
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
        gt_semantic_heights = (gt_semantic_heights - (self.grid_config['z_bounds'][0] -
                                                      self.grid_config['z_bounds'][2])) / self.grid_config['z_bounds'][
                                  2]
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
        height = (height - (self.grid_config['z_bounds'][0] - self.grid_config['z_bounds'][2])) / \
                 self.grid_config['z_bounds'][2]
        return height.long()

    def bp_projection(self, features, voxel_cam_coords,
                      voxel_cam_heights, voxel_cam_valid, dict_shape):
        # Alias
        n, c, *_ = features.shape  # 6, 80, [10, 64, 176]
        Z, Y, X = dict_shape['Z'], dict_shape['Y'], dict_shape['X']  # 10, 256, 256

        x = voxel_cam_coords[:, 0].round().long()  # [6, 480000]
        y = voxel_cam_coords[:, 1].round().long()  # [6, 480000]

        # VT
        voxel_feats = torch.zeros(  # (80, Npts)
            (c, Z * Y * X), device=features.device
        ).type_as(features)
        for i in range(n):
            voxel_feats[:, voxel_cam_valid[i]] = features[i, :, voxel_cam_heights[i, voxel_cam_valid[i]] - 1,
                                                 y[i, voxel_cam_valid[i]], x[i, voxel_cam_valid[i]]]

        voxel_feats = rearrange(voxel_feats, 'c (z y x) -> c z y x', c=c, z=Z, y=Y, x=X)

        return voxel_feats  # (80, 10, 256, 256)

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
