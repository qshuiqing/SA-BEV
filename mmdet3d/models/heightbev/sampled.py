import torch
from einops import repeat
from torch import nn


class SampledCoordSelector(nn.Module):

    def __init__(self,
                 spatial_kwargs):
        super(SampledCoordSelector, self).__init__()

        self.spatial_bounds = spatial_kwargs["spatial_bounds"]
        self.spatial_range = spatial_kwargs["projector"]

    def get_coarse_coords(self, bt, device):
        X, Y, H = self.spatial_range  # 256, 256, 10
        sb = self.spatial_bounds  # [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
        n_coarse = self.n_coarse  # 2500
        grid = self.grid_buffer

        # Random sample
        pillars = repeat(grid, 'x y c -> b (x y) c', b=bt)
        rnd = torch.randperm(X * Y)[:n_coarse].to(device)
        pillars = torch.index_select(pillars, dim=1, index=rnd)

        # Lift
        n_xy = pillars.size(1)
        pillars = repeat(pillars, 'bt xy c -> bt c (xy h)', h=H)

        # -> Regular H points
        pillar_heights = torch.linspace(0, 1, H, device=device)
        pillar_heights = repeat(pillar_heights, 'h -> bt 1 (xy h)', bt=bt, xy=n_xy)

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
        xyh = torch.tensor([X - 1, Y - 1, H - 1], device=device).view(1, 3, 1)
        vox_idx = (pillar_pts * xyh).round().to(torch.int32)

        return vox_coords, vox_idx

    def get_fine_coords(self):
        pass
