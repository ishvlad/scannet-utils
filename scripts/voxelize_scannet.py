import os
import argparse
import subprocess

from tqdm import tqdm
from multiprocessing import Pool

from config import SCANNET_DIR, SCANNET_VOXELIZED_DIR
from config import SDF_VOXELIZER_PATH


def parse_args():
    parser = argparse.ArgumentParser(description='Obtain voxelized scanned with given resolusion (usually 2cm or '
                                                 '5cm) from execution file of SDF voxelizer: PLY + SENS = VOX')
    parser.add_argument('--res', type=float, default=0.05, help='voxel resolution im meters, default=0.05 (5cm)')
    parser.add_argument('--n_jobs', type=int, default=10, help='number of processes')

    return parser.parse_args()


def ply_to_vox(args):
    scan_id, work_dir, resolution = args

    # deal with i\o paths
    common_input_path = os.path.join(SCANNET_DIR, work_dir, scan_id, scan_id)
    path_sens = common_input_path + '.sens'
    path_ply = common_input_path + '_vh_clean_2.ply'
    path_vox = os.path.join(SCANNET_VOXELIZED_DIR, work_dir, scan_id, scan_id) + '-res_' + str(resolution) + '.vox'

    # create output directory if needed
    os.makedirs('/'.join(path_vox.split('/')[:-1]), exist_ok=True)

    # exit if voxel already exists
    if os.path.exists(path_vox):
        return

    try:
        subprocess.check_call([
            SDF_VOXELIZER_PATH,
            '--rot=0', '--res=' + str(resolution),
                       '--vox=' + path_vox,
                       '--ply=' + path_ply,
                       '--sens=' + path_sens
        ], stdout=subprocess.DEVNULL)
    except subprocess.CalledProcessError as e:
        print(f'ERROR processing scene {common_input_path}.\n' + str(e), flush=True)


def main():
    args = parse_args()

    for work_dir in ['scans', 'scans_test']:
        # get full path
        print(work_dir)

        # get into dataset
        pool_args = [(scan_id, work_dir, args.res) for scan_id in os.listdir(os.path.join(SCANNET_DIR, work_dir))]

        with Pool(args.n_jobs) as pool:
            _ = list(tqdm(pool.imap(ply_to_vox, pool_args), total=len(pool_args)))


if __name__ == '__main__':
    main()
