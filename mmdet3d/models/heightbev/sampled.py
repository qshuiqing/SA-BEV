import torch
from einops import rearrange
from torch import nn


class SampledCoordSelector(nn.Module):

    def __init__(self,
                 sampled_kwargs):
        super(SampledCoordSelector, self).__init__()

        # Init
        self._init_status(sampled_kwargs)
        self._init_buffer()

    def _init_status(self, sampled_kwargs):
        # Coords
        self.x_bounds = sampled_kwargs['x_bounds']
        self.y_bounds = sampled_kwargs['y_bounds']
        self.z_bounds = sampled_kwargs['z_bounds']

        # self.n_voxels = sampled_kwargs['n_voxels']
        # self.voxel_size = sampled_kwargs['voxel_size']

        # Coarse
        self.n_coarse = sampled_kwargs['n_coarse']

        # Fine
        self.n_fine = sampled_kwargs['n_fine']

    def _init_buffer(self):
        # Init points, (z, y, x)
        dz, dy, dx = torch.meshgrid(
            [
                torch.arange(*self.z_bounds),
                torch.arange(*self.y_bounds),
                torch.arange(*self.x_bounds)
            ]
        )

        voxel_coords = torch.stack((dx, dy, dz))  # (3, 10, 256, 256)

        add = rearrange(
            torch.tensor([bounds[-1] / 2 for bounds in [self.x_bounds, self.y_bounds, self.z_bounds]]),
            'i -> i 1 1 1'
        )

        # Move 1/2 voxel size
        voxel_coords += add

        self.register_buffer('voxel_coords', voxel_coords)

    def get_coarse_coords(self, dict_shape):
        # Alias
        _, Z, Y, X = self.voxel_coords.shape

        # Prepare
        dict_shape.update(dict(Z=Z, Y=Y, X=X))

        return self.voxel_coords, None


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
