import os
import pandas as pd
import pickle


def get_global_part_labels_description(dictionaries_dir='dictionaries', return_full=False):
    csv_name = 'part_id_to_parts_description.csv'
    if return_full:
        csv_name = 'FULL_' + csv_name

    path = os.path.join(dictionaries_dir, csv_name)
    global_labels = pd.read_csv(path, index_col=0, dtype=str)

    global_labels['part_id'] = global_labels.part_id.astype(int)
    global_labels['set_id'] = global_labels.set_id.astype(int)

    return global_labels


def get_partnet_to_shapenet_transforms(dictionaries_dir='dictionaries'):
    path = os.path.join(dictionaries_dir, 'partnet_to_shapenet_transforms.pkl')
    with open(path, 'rb') as f:
        transforms = pickle.load(f)

    return transforms
