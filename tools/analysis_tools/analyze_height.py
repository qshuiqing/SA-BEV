# Copyright (c) OpenMMLab. All rights reserved.
import argparse

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint
from nuscenes import NuScenes
from tqdm import tqdm

from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_detector

nusc = NuScenes(version='v1.0-trainval', dataroot='data/nuscenes', verbose=True)


def parse_args():
    parser = argparse.ArgumentParser(description='MMDet benchmark a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--samples', default=1000, help='samples to benchmark')
    parser.add_argument(
        '--log-interval', default=50, help='interval of logging')
    parser.add_argument(
        '--mem-only',
        action='store_true',
        help='Conduct the memory analysis only')
    parser.add_argument(
        '--no-acceleration',
        action='store_true',
        help='Omit the pre-computation acceleration')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    # build the dataloader
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False)

    # build the model and load checkpoint
    # if not args.no_acceleration:
    #     cfg.model.img_view_transformer.accelerate = True
    cfg.model.train_cfg = None
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    model = MMDataParallel(model, device_ids=[0])

    model.eval()

    semantic_threshold = model.module.img_view_transformer.semantic_threshold  # 0.25
    for i, data in enumerate(tqdm(data_loader)):

        if i % 10 != 0: continue

        with torch.no_grad():
            img_inputs = [it.cuda() for it in data['img_inputs'][0]]
            img_metas = data['img_metas'][0].data[0]
            # (6, 10, 64, 176), (6, 2, 64, 176)
            _, (height_digit, semantic) = model.module.extract_img_feat(img_inputs, img_metas)

        # Alias
        height = [h.cpu().numpy() for h in height_digit.softmax(1).argmax(1)]
        kept = (semantic[:, 1:2] >= semantic_threshold)  # (6, 1, 64, 176)
        canvas = [cvs[0].numpy() for cvs in data['canvas'][0]]
        height_dist = height_dist_plot(height_digit, kept)

        # Up-sampling (64, 176) -> (256, 704)
        kept = F.interpolate(kept.float(), size=(256, 704), mode='nearest').bool()
        kept = kept.permute(0, 2, 3, 1).cpu().numpy()

        fig, axs = plt.subplots(6, 4, figsize=(15, 15))
        for cam_id in range(6):
            # 高度
            axs[cam_id, 0].imshow(height[cam_id], cmap='hot', interpolation='nearest')
            axs[cam_id, 0].axis('off')

            # 随机点高度分布
            axs[cam_id, 1].bar(range(10), height_dist[cam_id])
            axs[cam_id, 1].axis('off')

            # 原图
            axs[cam_id, 2].imshow(canvas[cam_id] / 255.0)
            axs[cam_id, 2].axis('off')

            # 语义掩码
            axs[cam_id, 3].imshow(canvas[cam_id] * kept[cam_id] / 255.0)
            axs[cam_id, 3].axis('off')

        plt.tight_layout()
        plt.savefig('vis/height/{}.png'.format(img_metas[0]['sample_idx']))
        plt.close(fig)

        nusc.render_sample(img_metas[0]['sample_idx'],
                           out_path='vis/height/{}.png'.format(img_metas[0]['sample_idx'] + '_3d'))


def height_dist_plot(height_digit, kept):
    height = height_digit.softmax(dim=1)
    height = height * kept

    height_dist = []
    for i in range(height.shape[0]):
        valid_idx = height[i].nonzero()
        if len(valid_idx) > 0:
            _, h, w = height[i].nonzero()[0]
        else:
            h, w = 0, 0
        height_dist.append(height[i, :, h, w].cpu().numpy())

    return height_dist


if __name__ == '__main__':
    main()
