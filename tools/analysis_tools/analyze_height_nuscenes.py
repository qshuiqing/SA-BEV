import csv
import os

import numpy as np
import pandas as pd
from nuscenes import NuScenes
from tqdm import tqdm


def generate():
    dataroot = r'data/nuscenes'
    nusc = NuScenes(version='v1.0-trainval', dataroot=dataroot, verbose=False)

    def update_dict(d, key):
        if key in d:
            d[key] += 1
        else:
            if key < 0:
                d[key - 1] = 1
            else:
                d[key] = 1

    height = {}
    for i, sample in tqdm(enumerate(nusc.sample)):

        camera_data = {}
        for channel, token in sample['data'].items():
            sd_record = nusc.get('sample_data', token)
            sensor_modality = sd_record['sensor_modality']

            if sensor_modality == 'camera':
                camera_data[channel] = token

        for channel, sd_token in camera_data.items():
            sd_record = nusc.get('sample_data', sd_token)
            sensor_modality = sd_record['sensor_modality']
            assert sensor_modality == 'camera'

            _, boxes, _ = nusc.get_sample_data(sd_token)

            for box in boxes:
                update_dict(height, box.center[1].astype(np.int64))

    # 保存 height 分布
    with open('height_dis.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        for key, value in height.items():
            writer.writerow([key, value])


def generate_height_from_lidar():
    def update_dict(d, k):
        if k in d:
            d[k] += 1
        else:
            if k < 0:
                d[k - 1] = 1
            else:
                d[k] = 1

    height = {}
    for root, dirs, files in os.walk('data/nuscenes/samples_point_label'):
        for file in tqdm(files):
            if file.endswith('.npy'):
                data = np.load(os.path.join(root, file)).astype(np.float64)[:5].T
                # 过滤超过图片的点
                coords = data[:, :2].astype(np.int16)
                valid_mask = ((coords[:, 1] < 900)
                              & (coords[:, 0] < 1600)
                              & (coords[:, 1] >= 0)
                              & (coords[:, 0] >= 0))
                height_data = data[valid_mask, 4]
                for _h in height_data:
                    update_dict(height, _h.astype(np.int64))

    # 保存 height 分布
    with open('height_dis_lidar.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        for key, value in height.items():
            writer.writerow([key, value])


def dis_heatmap():
    import matplotlib.pyplot as plt

    # 假设这是你的字典
    df = pd.read_csv('height_dis_lidar.csv', header=None, names=['key', 'value'])
    df = df.sort_values(by='key', ascending=False)

    # 将字典转换为列表，以便于绘图
    keys = list(df['key'].tolist())
    values = list(df['value'].tolist())

    # 创建柱状图
    plt.bar(keys, values)

    # 添加标题和标签
    plt.title('Key-Value Bar Chart')
    plt.xlabel('Keys')
    plt.ylabel('Values')

    # 显示柱状图
    plt.show()


if __name__ == '__main__':
    dis_heatmap()
