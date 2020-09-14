########################################
# Prerequisites (in TMP_DIR):
#   - move partnet_transformations_3.p and icp_partnet_transformations_2.p to TMP_DIR,
#           obtained by combinatorial optimization and ICP procedure
########################################

import os
import pickle

TMP_DIR = 'tmp'
OUTPUT_DIR = 'dictionaries'


#####################################
# PARTNET -> SHAPENET transformations
#####################################
output_name = os.path.join(OUTPUT_DIR, 'partnet_to_shapenet_transforms.pkl')
if not os.path.exists(output_name):
    with open(os.path.join(TMP_DIR, 'partnet_transformations_3.p'), 'rb') as f:
        first_matrix = pickle.load(f)
    with open(os.path.join(TMP_DIR, 'icp_partnet_transformations_2.p'), 'rb') as f:
        second_matrix = pickle.load(f)

    matrices = {}
    for k in first_matrix:
        matrices[k] = first_matrix[k]['matrix'] @ second_matrix[k]['matrix']

    with open(output_name, 'wb+') as f:
        pickle.dump(matrices, f)


#################################################################
# Great part description table w/ using PARTNET and SHAPENET info
#################################################################
# def dfs(note, res):
#     """
#     Helper recursive function for DFS through Partnet trees
#
#     :param note: tree root
#     :param res: dictionary with result
#     :return: dictionary obj_id:(text, id, name)
#     """
#     if 'children' in note:
#         for c in note['children']:
#             res = dfs(c, res)
#         return res
#     else:
#         for o in note['objs']:
#             res[o] = {
#                 'text': note['text'],
#                 'id': note['id'],
#                 'name': note['name'],
#             }
#         return res
#
#
# # mapping SHAPE_ID -> CATEGORY_ID
# shapenet_objects = glob.glob(SHAPENET_DIR + '/*/*')
# from_object_id_to_category = dict([x.split('/')[-2:][::-1] for x in shapenet_objects])
#
# # mapping CATEGORY_ID -> CATEGORY_NAME
# with open(os.path.join(SHAPENET_DIR, 'taxonomy.json'), 'rb') as f:
#     shapenet_taxonomy = json.load(f)
# category_description = {
#     x['synsetId']: x['name'] for x in shapenet_taxonomy
# }
#
# # loop over PARTNET objects
# partnet_objects = sorted(glob.glob(PARTNET_DIR + '/*'))
# part_description = []
# for path in tqdm.tqdm(partnet_objects):
#     partnet_id = path.split('/')[-1]
