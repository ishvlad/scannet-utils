##############################################
#  Map Partnet labels to Scannet pointclouds
##############################################
import sys
sys.path.append('/home/ishvlad/workspace/scannet-utils')

import json
import numpy as np
import os
import pickle
import tqdm

from scipy.spatial import cKDTree

from src import dictionaries

from src.get_scannet import get_scannet
from src.transformations import apply_transform


SCAN2CAD_DIR = 'datasets/scan2cad'
# SCANNET_DIR = 'datasets/scannet'
# PARTNET_DIR = 'datasets/partnet'

SCANNET_DIR = '/home/ishvlad/datasets/scannet/scans'
PARTNET_DIR = '/home/ishvlad/datasets/PartNet'


# load global part description
global_labels = dictionaries.get_global_part_labels_description()
partnet_to_shapenet_transforms = dictionaries.get_partnet_to_shapenet_transforms()


# load SCAN2CAD annotations
with open(os.path.join(SCAN2CAD_DIR, 'full_annotations.json'), 'rb') as f:
    scan2cad_anno = json.load(f)

# for each scene at Scan2CAD
for anno_item in tqdm.tqdm(scan2cad_anno):
    # get Scan2CAD info
    scan_id = anno_item['id_scan']
    scan_transform = anno_item["trs"]
    aligned_models = anno_item['aligned_models']

    # load scannet point cloud (mesh vertices)
    scan_points_origin = get_scannet(scan_id, SCANNET_DIR, output_type='array')

    # transform scan to Scan2CAD coordinate system
    scan_points = apply_transform(scan_points_origin, scan_transform)

    # init_scan_mask
    semantic_mask = -np.ones(len(scan_points))
    instance_mask = -np.ones(len(scan_points))

    # for each aligned shape
    object_id = 0
    for anno_shape in tqdm.tqdm(aligned_models):
        # get Scan2CAD info about shape
        category_id = anno_shape['catid_cad']
        shape_id = anno_shape['id_cad']
        shape_transform = anno_shape["trs"]

        # mapping PART_ID -> GLOBAL_ID
        df_parts = global_labels[
            (global_labels['category_id'] == category_id) & (global_labels['object_id'] == shape_id)
        ]
        from_part_id_2_global_id = dict(df_parts.reset_index()[['part_id', 'global_id']].values)

        if len(df_parts) == 0:
            continue

        # load shape pointcloud
        partnet_id = df_parts.object_partnet_id.values[0]
        partnet_path = os.path.join(PARTNET_DIR, partnet_id, 'point_sample')

        # LOAD and MAP: PARTNET -> SHAPENET -> Scan2CAD coordinate system
        shape_ply = np.loadtxt(os.path.join(partnet_path, 'pts-10000.pts'), delimiter=' ')[:, :3]
        partnet_transform = partnet_to_shapenet_transforms[partnet_id]
        shape_ply = apply_transform(shape_ply, partnet_transform, shape_transform)

        # load shape part labels
        shape_label = np.loadtxt(os.path.join(partnet_path, 'label-10000.txt'), delimiter='\n')
        shape_label = np.array([from_part_id_2_global_id[p_id] for p_id in shape_label])

        # calculate distance
        tree = cKDTree(shape_ply)
        min_dist, min_idx = tree.query(scan_points)

        # Color
        for is_near, i_nn, i_point in zip(min_dist <= 0.07, min_idx, range(len(scan_points))):
            if is_near:
                semantic_mask[i_point] = shape_label[i_nn]
                instance_mask[i_point] = object_id

        object_id += 1

    result = {
        'num_objects': object_id,
        'vertices': scan_points_origin,
        'semantic_labels': semantic_mask,
        'instance_labels': instance_mask
    }

    output_path = os.path.join(SCANNET_DIR, scan_id, scan_id + '.pc.colored.pkl')
    with open(output_path, 'wb+') as f:
        pickle.dump(result, f)
