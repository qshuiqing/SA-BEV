import math

import torch
import torch.nn.functional as F
from mmcv.runner import BaseModule
from mmseg.ops import resize

from ..builder import NECKS


@NECKS.register_module()
class HeightVT(BaseModule):

    def __init__(self,
                 input_size,
                 n_voxels,
                 voxel_size,
                 backproject='inplace',
                 multi_scale_3d_scaler='upsample',
                 **kwargs):
        super(HeightVT, self).__init__(**kwargs)

        self.input_size = input_size
        self.n_voxels = n_voxels
        self.voxel_size = voxel_size
        self.backproject = backproject
        self.multi_scale_3d_scaler = multi_scale_3d_scaler

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

        cam2img = rots.new_zeros(N, 4, 4)
        cam2img[:, :3, :3] = intrinsics
        cam2img[:, 3, 3] = 1

        post_rt = post_rots.new_zeros(N, 4, 4)
        post_rt[:, :3, :3] = post_rots
        post_rt[:, :3, 2] += post_trans
        post_rt[:, 3, 3] = 1

        bda_mat = rots.new_zeros(N, 4, 4)
        bda_mat[:, :3, :3] = bda
        bda_mat[:, 3, 3] = 1

        keyego2img = post_rt @ cam2img @ cam2keyego.inverse() @ bda_mat.inverse()

        return scale @ keyego2img[:, :3]

    def view_transform(self, mlvl_feats, cam_params, img_metas):
        """
        2D-to-3D视图转换
        Args:
            mlvl_feats:
            cam_params:
            img_metas:
        Returns:
            bev_feat:
        """

        #
        mlvl_volumes = []
        for lvl, mlvl_feat in enumerate(mlvl_feats):  # 3
            stride_i = math.ceil(self.input_size[-1] / mlvl_feat.shape[-1])  # 16

            volumes = []
            for batch_id, img_meta in enumerate(img_metas):
                feat_i = mlvl_feat[batch_id]  # (n, c, h, w)
                cam_param_i = [p[batch_id] for p in cam_params]

                # 计算前视相机ego到图像投影矩阵
                projection = self._compute_projection(*cam_param_i, stride_i)

                # 计算前视相机ego下采点
                n_voxels, voxel_size = self.n_voxels[lvl], self.voxel_size[lvl]
                points = get_points(  # [3, dz, dy, dx]
                    n_voxels=torch.tensor(n_voxels),
                    voxel_size=torch.tensor(voxel_size),
                    origin=torch.tensor(img_meta["origin"]),
                ).to(feat_i.device)

                if self.backproject == 'inplace':
                    volume = backproject_inplace(
                        feat_i[:, :, :, :], points, projection)  # [c, dz, dy, dx]
                else:
                    volume, valid = backproject_vanilla(
                        feat_i[:, :, :, :], points, projection)
                    volume = volume.sum(dim=0)
                    valid = valid.sum(dim=0)
                    volume = volume / valid
                    valid = valid > 0
                    volume[:, ~valid[0]] = 0.0
                volumes.append(volume)  # bs x (c, dz, dy, dx)
            mlvl_volumes.append(torch.stack(volumes, dim=0))  # lvl x (bs, c, dz, dy, dx)

        # multi-voxels -> multi-bev
        for i in range(len(mlvl_volumes)):  # 3
            mlvl_volume = mlvl_volumes[i]
            bs, c, dz, dy, dx = mlvl_volume.shape
            # (bs, c, dz, dy, dx)->(bs, c*dz, dy, dx)
            mlvl_volume = mlvl_volume.view(bs, c * dz, dy, dx)

            # different x/y, [bs, seq*c*vz, vx, vy] -> [bs, seq*c*vz, vx', vy']
            if self.multi_scale_3d_scaler == 'pool' and i != (len(mlvl_volumes) - 1):
                # pooling to bottom level
                mlvl_volume = F.adaptive_avg_pool2d(mlvl_volume, mlvl_volumes[-1].size()[2:4])
            elif self.multi_scale_3d_scaler == 'upsample' and i != 0:
                # upsampling to top level
                mlvl_volume = resize(
                    mlvl_volume,
                    mlvl_volumes[0].size()[2:4],
                    mode='bilinear',
                    align_corners=False)
            else:
                # same x/y
                pass

            # [bs, seq*c*vz, vx', vy'] -> [bs, seq*c*vz, vx, vy, 1]
            # mlvl_volume = mlvl_volume.unsqueeze(-1)
            mlvl_volumes[i] = mlvl_volume
        mlvl_volumes = torch.cat(mlvl_volumes, dim=1)  # [bs, z1*c1+z2*c2+..., vx, vy, 1]

        # bev_feat = self.neck_3d(mlvl_volumes)

        return mlvl_volumes

    def forward(self, img_inputs, img_metas):
        mlvl_feats = img_inputs[0][0]
        cam_params = img_inputs[1:7]

        return self.view_transform([mlvl_feats], cam_params, img_metas), (None, None)


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


def backproject_inplace(features, points, projection):
    '''
    function: 2d feature + predefined point cloud -> 3d volume
    input:
        features: [6, 64, 225, 400]
        points: [3, 200, 200, 12]
        projection: [6, 3, 4]
    output:
        volume: [64, 200, 200, 12]
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

    # method2：特征填充，只填充有效特征，重复特征直接覆盖
    volume = torch.zeros(
        (n_channels, points.shape[-1]), device=features.device
    ).type_as(features)
    for i in range(n_images):
        volume[:, valid[i]] = features[i, :, y[i, valid[i]], x[i, valid[i]]]

    volume = volume.view(n_channels, n_z_voxels, n_y_voxels, n_x_voxels)
    return volume
