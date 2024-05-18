import torch
from einops import rearrange, repeat
from torch import nn


class CamProjector(nn.Module):

    def __init__(self,
                 sampled_kwargs):
        super(CamProjector, self).__init__()

        self._init_status(sampled_kwargs)

    def _init_status(self, sampled_kwargs):
        # Height
        self.z_bounds = sampled_kwargs['z_bounds']

        self.H = len(torch.arange(*self.z_bounds))

    def map2bin(self, vox_cam_coords):
        """
        高度离散化
        """
        # [-5, 3, 0.1]
        # 将height离散化到 [-5, 3, 0.1] 的bin中
        height = (vox_cam_coords - (self.z_bounds[0] - self.z_bounds[2])) / \
                 self.z_bounds[2]
        return height.long()

    def from_voxel_to_cams(self, vox_coords, rots, trans, bda):
        # Alias
        n, *_ = rots.shape  # 6, [3, 3]

        homog_mat = rots.new_zeros(n, 4, 4)
        homog_mat[:, :3, :3] = rots
        homog_mat[:, :3, 3] = trans
        homog_mat[:, 3, 3] = 1

        bda_mat = rots.new_zeros(n, 4, 4)
        bda_mat[:, :3, :3] = bda
        bda_mat[:, 3, 3] = 1

        ego2cam = homog_mat.inverse() @ bda_mat.inverse()

        vox_coords = rearrange(vox_coords, 'i z y x -> i (z y x)', i=3)  # (3, Npts)
        vox_coords = repeat(vox_coords, 'i Npts -> n i Npts', n=n)  # (6, 3, Npts)
        vox_cam_coords = torch.cat([vox_coords, torch.ones_like(vox_coords[:, :1])], dim=1)  # (4, Npts)
        vox_cam_coords = torch.bmm(ego2cam[:, :3], vox_cam_coords)  # (6, 4, Npts)

        # Calculate heights
        vox_cam_heights = self.map2bin(vox_cam_coords[:, 1, :])

        return vox_cam_coords, vox_cam_heights

    def valid_points_in_cam(self, vox_cam_coords, vox_cam_heights):
        return (vox_cam_coords[:, 2] > 0.0), (vox_cam_heights > 0) & (vox_cam_heights < self.H + 1)

    def from_cameras_to_pixels(self, vox_cam_coords, intrins, post_rots, post_trans, rescale):
        """Transform points from camera reference frame to image reference frame."""
        # Alias
        n, i, j = intrins.shape  # 6, 3, 3

        # Rescale
        rescale_mat = torch.eye(3, device=post_rots.device).repeat(n, 1, 1)
        rescale_mat[:, :2] /= rescale

        # Image aug
        homog_mat = torch.eye(3, device=post_rots.device).repeat(n, 1, 1)
        homog_mat[:, :3, :3] = post_rots
        homog_mat[:, :3, 2] += post_trans

        # Cameras to pixels.
        intrins = torch.matmul(rescale_mat, torch.matmul(homog_mat, intrins))
        vox_cam_coords = torch.bmm(intrins, vox_cam_coords)

        return vox_cam_coords

    def normalize_z_cam(self, vox_cam_coords):
        vox_cam_coords[:, :2] /= vox_cam_coords[:, 2:]
        return vox_cam_coords

    def valid_points_in_pixels(self, vox_cam_coords, feat_shape):
        x = vox_cam_coords[:, 0].round().long()
        y = vox_cam_coords[:, 1].round().long()
        return (x >= 0) & (x < feat_shape[1]), (y >= 0) & (y < feat_shape[0])

    def forward(self, vox_coords, cam_params, dict_shape):
        # Unpack
        rots, trans, intrins, post_rots, post_trans, bda = cam_params

        # Ego to cams.
        vox_cam_coords, vox_cam_heights = self.from_voxel_to_cams(  # (6, 3, Npts), (6, Npts)
            vox_coords,
            rots,
            trans,
            bda
        )
        z_valid, h_valid = self.valid_points_in_cam(vox_cam_coords, vox_cam_heights)  # (6, Npts)

        # Cams to pixels.
        vox_cam_coords = self.from_cameras_to_pixels(
            vox_cam_coords,  # (6, 3, Npts)
            intrins,
            post_rots,
            post_trans,
            dict_shape['Himg'] / dict_shape['Hfeat']  # 4.0
        )
        vox_cam_coords = self.normalize_z_cam(vox_cam_coords)
        x_valid, y_valid = self.valid_points_in_pixels(vox_cam_coords, (dict_shape["Hfeat"], dict_shape["Wfeat"]))

        # Filter valid points.
        vox_valid = (x_valid & y_valid & z_valid & h_valid)

        return vox_cam_coords, vox_cam_heights, vox_valid
