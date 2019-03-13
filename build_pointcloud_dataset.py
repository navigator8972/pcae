from plyfile import PlyData, PlyElement
import numpy as np

import os

datafiles_folder = '../data/shape_net_core_uniform_samples_2048'

snc_synth_id_to_category = {
    '02691156': 'airplane',  '02773838': 'bag',        '02801938': 'basket',
    '02808440': 'bathtub',   '02818832': 'bed',        '02828884': 'bench',
    '02834778': 'bicycle',   '02843684': 'birdhouse',  '02871439': 'bookshelf',
    '02876657': 'bottle',    '02880940': 'bowl',       '02924116': 'bus',
    '02933112': 'cabinet',   '02747177': 'can',        '02942699': 'camera',
    '02954340': 'cap',       '02958343': 'car',        '03001627': 'chair',
    '03046257': 'clock',     '03207941': 'dishwasher', '03211117': 'monitor',
    '04379243': 'table',     '04401088': 'telephone',  '02946921': 'tin_can',
    '04460130': 'tower',     '04468005': 'train',      '03085013': 'keyboard',
    '03261776': 'earphone',  '03325088': 'faucet',     '03337140': 'file',
    '03467517': 'guitar',    '03513137': 'helmet',     '03593526': 'jar',
    '03624134': 'knife',     '03636649': 'lamp',       '03642806': 'laptop',
    '03691459': 'speaker',   '03710193': 'mailbox',    '03759954': 'microphone',
    '03761084': 'microwave', '03790512': 'motorcycle', '03797390': 'mug',
    '03928116': 'piano',     '03938244': 'pillow',     '03948459': 'pistol',
    '03991062': 'pot',       '04004475': 'printer',    '04074963': 'remote_control',
    '04090263': 'rifle',     '04099429': 'rocket',     '04225987': 'skateboard',
    '04256520': 'sofa',      '04330267': 'stove',      '04530566': 'vessel',
    '04554684': 'washer',    '02858304': 'boat',       '02992529': 'cellphone'
}

def build_dataset_from_plyfiles(folder, categories=None):
    data = {}
    for class_folder in os.listdir(datafiles_folder):
        category = snc_synth_id_to_category[class_folder]
        if categories is not None:
            if not category in categories:
                continue
        full_class_folder = os.path.join(datafiles_folder, class_folder)
        def extract_vertex_from_plyfile(f):
            pc_data = PlyData.read(f)
            return np.array([pc_data['vertex']['x'], pc_data['vertex']['y'], pc_data['vertex']['z']])
        class_data = [extract_vertex_from_plyfile(os.path.join(full_class_folder, f)) for f in os.listdir(full_class_folder) if f.endswith('.ply')]
        print('Processing Class {}'.format(category))
        data[category] = np.array(class_data)

    return data

if __name__ == '__main__':
    categories = ['chair', 'mug', 'table']
    data = build_dataset_from_plyfiles(datafiles_folder, categories)
    print(data['chair'].shape)
    np.savez('3DShapeNet_PointCloud2048', data=data)

    # loaded_data = np.load('3DShapeNet_PointCloud2048.npz')
    # print(loaded_data['data'].item()['chair'].shape)
    