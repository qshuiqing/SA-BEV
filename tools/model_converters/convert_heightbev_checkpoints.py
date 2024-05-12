# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import tempfile

import torch
from mmcv import Config


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet3D upgrade model version(before v0.6.0) of H3DNet')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--out', help='path of the output checkpoint file')
    args = parser.parse_args()
    return args


def parse_config(config_strings):
    """Parse config from strings.

    Args:
        config_strings (string): strings of model config.

    Returns:
        Config: model config
    """
    temp_file = tempfile.NamedTemporaryFile()
    config_path = f'{temp_file.name}.py'
    with open(config_path, 'w') as f:
        f.write(config_strings)

    config = Config.fromfile(config_path)

    # Update backbone config
    if 'pool_mod' in config.model.backbone.backbones:
        config.model.backbone.backbones.pop('pool_mod')

    if 'sa_cfg' not in config.model.backbone:
        config.model.backbone['sa_cfg'] = dict(
            type='PointSAModule',
            pool_mod='max',
            use_xyz=True,
            normalize_xyz=True)

    if 'type' not in config.model.rpn_head.vote_aggregation_cfg:
        config.model.rpn_head.vote_aggregation_cfg['type'] = 'PointSAModule'

    # Update rpn_head config
    if 'pred_layer_cfg' not in config.model.rpn_head:
        config.model.rpn_head['pred_layer_cfg'] = dict(
            in_channels=128, shared_conv_channels=(128, 128), bias=True)

    if 'feat_channels' in config.model.rpn_head:
        config.model.rpn_head.pop('feat_channels')

    if 'vote_moudule_cfg' in config.model.rpn_head:
        config.model.rpn_head['vote_module_cfg'] = config.model.rpn_head.pop(
            'vote_moudule_cfg')

    if config.model.rpn_head.vote_aggregation_cfg.use_xyz:
        config.model.rpn_head.vote_aggregation_cfg.mlp_channels[0] -= 3

    for cfg in config.model.roi_head.primitive_list:
        cfg['vote_module_cfg'] = cfg.pop('vote_moudule_cfg')
        cfg.vote_aggregation_cfg.mlp_channels[0] -= 3
        if 'type' not in cfg.vote_aggregation_cfg:
            cfg.vote_aggregation_cfg['type'] = 'PointSAModule'

    if 'type' not in config.model.roi_head.bbox_head.suface_matching_cfg:
        config.model.roi_head.bbox_head.suface_matching_cfg[
            'type'] = 'PointSAModule'

    if config.model.roi_head.bbox_head.suface_matching_cfg.use_xyz:
        config.model.roi_head.bbox_head.suface_matching_cfg.mlp_channels[
            0] -= 3

    if 'type' not in config.model.roi_head.bbox_head.line_matching_cfg:
        config.model.roi_head.bbox_head.line_matching_cfg[
            'type'] = 'PointSAModule'

    if config.model.roi_head.bbox_head.line_matching_cfg.use_xyz:
        config.model.roi_head.bbox_head.line_matching_cfg.mlp_channels[0] -= 3

    if 'proposal_module_cfg' in config.model.roi_head.bbox_head:
        config.model.roi_head.bbox_head.pop('proposal_module_cfg')

    temp_file.close()

    return config


def main():
    """Convert keys in checkpoints for VoteNet.

    There can be some breaking changes during the development of mmdetection3d,
    and this tool is used for upgrading checkpoints trained with old versions
    (before v0.6.0) to the latest one.
    """
    args = parse_args()
    checkpoint = torch.load(args.checkpoint)

    # 获取你的模型的状态字典
    model_dict = checkpoint['state_dict']

    # 创建一个新的状态字典，将预训练模型的键修改为你的模型的键
    new_state_dict = {}
    for k, v in model_dict.items():
        # 将'backbone'修改为'img_backbone'，将'neck'修改为'img_neck'
        if 'backbone' in k:
            k = k.replace('backbone', 'img_backbone')
        if 'neck' in k:
            k = k.replace('neck', 'img_neck')
        new_state_dict[k] = v

    # 更新你的模型的状态字典
    checkpoint['state_dict'] = new_state_dict
    # torch.save(checkpoint, args.out)


if __name__ == '__main__':
    main()
