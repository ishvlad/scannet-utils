import os
import pandas as pd
import pickle


class Partnet:
    def __init__(self, data_dir: str = None, dictionaries_dir: str = None):
        self.data_dir = data_dir
        self.dictionaries_dir = dictionaries_dir

    def load_part_description(self, return_full=False):
        csv_name = 'part_id_to_parts_description.csv'
        if return_full:
            csv_name = 'FULL_' + csv_name

        path = os.path.join(self.dictionaries_dir, csv_name)
        global_labels = pd.read_csv(path, index_col=0, dtype=str)

        global_labels['part_id'] = global_labels.part_id.astype(int)
        global_labels['set_id'] = global_labels.set_id.astype(int)

        return global_labels

    def load_to_shapenet_transforms(self):
        pkl_name = 'partnet_to_shapenet_transforms.pkl'
        path = os.path.join(self.dictionaries_dir, pkl_name)

        with open(path, 'rb') as f:
            transforms = pickle.load(f)
        return transforms
