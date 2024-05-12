from nuscenes import NuScenes
from tqdm import tqdm

NameMapping = {
    'movable_object.barrier': 'barrier',
    'vehicle.bicycle': 'bicycle',
    'vehicle.bus.bendy': 'bus',
    'vehicle.bus.rigid': 'bus',
    'vehicle.car': 'car',
    'vehicle.construction': 'construction_vehicle',
    'vehicle.motorcycle': 'motorcycle',
    'human.pedestrian.adult': 'pedestrian',
    'human.pedestrian.child': 'pedestrian',
    'human.pedestrian.construction_worker': 'pedestrian',
    'human.pedestrian.police_officer': 'pedestrian',
    'movable_object.trafficcone': 'traffic_cone',
    'vehicle.trailer': 'trailer',
    'vehicle.truck': 'truck'
}
filter_lidarseg_classes = tuple(NameMapping.keys())

class_names = (
    'car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle',
    'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'
)
class_idx = dict()
for i, name in enumerate(class_names):
    class_idx[name] = i

camera_names = (
    'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT',
    'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_FRONT_LEFT'
)


def generate():
    dataroot = r'data/nuscenes'
    nusc = NuScenes(version='v1.0-trainval', dataroot=dataroot, verbose=False)
    filter_lidarseg_labels = []
    for class_name in filter_lidarseg_classes:
        filter_lidarseg_labels.append(nusc.lidarseg_name2idx_mapping[class_name])

    height = []
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
                height.append(box.center[1])


if __name__ == '__main__':
    generate()
