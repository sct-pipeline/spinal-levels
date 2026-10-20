"""
Compute the distance along the spinal cord centerline between the pontomedullary junction (PMJ) and the last
(most inferior) axial slice of the spinal cord segmentation.

The script does the following:
    - generate the centerline extrapolated up to the PMJ (sct_process_segmentation -pmj, -v 2)
    - compute the cumulative arc length along the centerline, from the PMJ (= 0 mm) downward
    - find the most inferior axial slice containing the spinal cord segmentation
    - report the distance (mm) between the PMJ and that slice, and save it in a CSV file

The script requires the SCT conda environment to be activated:
    source ${SCT_DIR}/python/etc/profile.d/conda.sh
    conda activate venv_sct

Examples:
    python pmj_to_seg_bottom_distance.py -seg sub-001_T2w_label-SC_seg.nii.gz -pmj sub-001_T2w_label-pmj.nii.gz
    python pmj_to_seg_bottom_distance.py -seg sub-001_T2w_label-SC_seg.nii.gz -pmj sub-001_T2w_label-pmj.nii.gz \
        -o results/pmj_to_seg_bottom.csv -append

NOTE: Inspired by rootlets_to_spinal_levels.py (sct-pipeline/spinal-levels).
"""

import os
import argparse
import subprocess
import tempfile
import numpy as np
import pandas as pd

from argparse import RawTextHelpFormatter
from spinalcordtoolbox.image import Image


def get_parser():
    """
    parser function
    """
    parser = argparse.ArgumentParser(
        description='Compute the distance along the spinal cord centerline between the pontomedullary junction (PMJ) '
                    'and the last (most inferior) axial slice of the spinal cord segmentation.',
        formatter_class=RawTextHelpFormatter,
        prog=os.path.basename(__file__)
    )
    parser.add_argument(
        '-seg',
        required=True,
        help='Path to the spinal cord segmentation (binary).'
    )
    parser.add_argument(
        '-pmj',
        required=True,
        help='Path to the pontomedullary junction (PMJ) label.'
    )
    parser.add_argument(
        '-o',
        required=False,
        help='Output CSV file. Default: <seg>_pmj_to_seg_bottom.csv next to the segmentation.'
    )
    parser.add_argument(
        '-append',
        action='store_true',
        help='Append the result to the output CSV if it already exists (useful to gather all subjects in one file).'
    )
    return parser


def get_centerline_from_pmj(fname_seg, fname_pmj):
    """
    Generate extrapolated centerline from pontomedullary junction (PMJ).
    sct_process_segmentation is run in a temporary folder so that its side outputs (csa.csv, png files) do not
    clutter (or delete files from) the current working directory.
    :param fname_seg: spinal cord segmentation
    :param fname_pmj: PMJ label
    :return: 3xN array (x, y, z) of the extrapolated centerline, in voxel coordinates (RPI)
    """
    # Inspiration: https://github.com/sct-pipeline/pmj-based-csa/blob/e362536a7bef17c0e151830cf777fb51fd09cb87/process_data.sh#L157-L160
    # Note: -v 2 is used to generate the extrapolated centerline CSV
    fname_seg = os.path.abspath(fname_seg)
    fname_pmj = os.path.abspath(fname_pmj)
    with tempfile.TemporaryDirectory() as tmpdir:
        subprocess.run(['sct_process_segmentation', '-i', fname_seg, '-pmj', fname_pmj, '-pmj-distance', '50',
                        '-o', os.path.join(tmpdir, 'csa.csv'), '-v', '2'], check=True, cwd=tmpdir)

        # The extrapolated centerline is written next to the segmentation (or in the working directory,
        # depending on the SCT version) -> check both locations
        basename = os.path.basename(fname_seg).replace('.nii.gz', '_centerline_extrapolated.csv')
        candidates = [fname_seg.replace('.nii.gz', '_centerline_extrapolated.csv'),
                      os.path.join(tmpdir, basename)]
        for fname_centerline in candidates:
            if os.path.isfile(fname_centerline):
                return np.genfromtxt(fname_centerline, delimiter=',')

    raise FileNotFoundError(f'Extrapolated centerline not found. Looked for: {candidates}')


def get_distance_from_pmj(centerline_points, z_index, px, py, pz):
    """
    Compute distance from projected pontomedullary junction (PMJ) on centerline and cord centerline.
    Inspiration: https://github.com/sct-pipeline/pmj-based-csa/blob/419ece49c81782f23405d89c7b4b15d8e03ed4bd/get_distance_pmj_disc.py#L40-L60
    :param centerline_points: 3xn array: Centerline in continuous coordinate (float) for each slice in RPI orientation.
    :param z_index: z index PMJ on the centerline.
    :param px: x pixel size.
    :param py: y pixel size.
    :param pz: z pixel size.
    :return: nd-array: row 0 = distance from PMJ (mm), row 1 = z slice index
    """
    length = 0
    arr_length = [0]
    for i in range(z_index, 0, -1):
        distance = np.sqrt(((centerline_points[0, i] - centerline_points[0, i - 1]) * px) ** 2 +
                           ((centerline_points[1, i] - centerline_points[1, i - 1]) * py) ** 2 +
                           ((centerline_points[2, i] - centerline_points[2, i - 1]) * pz) ** 2)
        length += distance
        arr_length.append(length)
    arr_length = arr_length[::-1]
    arr_length = np.stack((arr_length, centerline_points[2][:z_index + 1]), axis=0)

    return arr_length


def main():
    # Parse the command line arguments
    parser = get_parser()
    args = parser.parse_args()

    fname_seg = args.seg
    fname_pmj = args.pmj
    fname_out = args.o or fname_seg.replace('.nii.gz', '_pmj_to_seg_bottom.csv')

    # Load input images using the SCT Image class
    im_seg = Image(fname_seg).change_orientation('RPI')
    im_pmj = Image(fname_pmj).change_orientation('RPI')

    # Sanity checks
    if len(np.unique(im_seg.data)) != 2:
        raise ValueError('The spinal cord segmentation should be binary.')
    if len(np.unique(im_pmj.data)) == 1:
        raise ValueError('The PMJ label file is empty.')

    # Most inferior axial slice of the SC segmentation (RPI: z increases superiorly -> min z)
    z_last = int(np.where(im_seg.data > 0)[2].min())
    # PMJ slice (for reporting)
    z_pmj = int(np.where(im_pmj.data > 0)[2].max())

    # Generate extrapolated centerline from PMJ and compute the distance along it
    centerline = get_centerline_from_pmj(fname_seg, fname_pmj)
    px, py, pz = im_seg.dim[4], im_seg.dim[5], im_seg.dim[6]
    arr_distance = get_distance_from_pmj(centerline, int(centerline[2].argmax()), px, py, pz)

    # Find the centerline point at the last slice of the segmentation
    z_centerline = np.rint(arr_distance[1]).astype(int)
    if z_last < z_centerline.min():
        print(f'WARNING: last seg slice ({z_last}) is below the centerline extent ({z_centerline.min()}); '
              f'using the lowest centerline slice instead.')
        z_last = int(z_centerline.min())
    idx = int(np.argmin(np.abs(z_centerline - z_last)))
    dist_pmj_to_last = float(arr_distance[0, idx])

    print(f'PMJ slice: {z_pmj}, last SC seg slice: {z_last}')
    print(f'Distance PMJ -> last SC seg slice along the centerline: {dist_pmj_to_last:.2f} mm')

    df = pd.DataFrame([{'fname_seg': fname_seg,
                        'fname_pmj': fname_pmj,
                        'slice_pmj_I-S': z_pmj,
                        'slice_seg_last_I-S': z_last,
                        'distance_pmj_to_seg_last_mm': dist_pmj_to_last}])

    # Save (or append to) the CSV file
    os.makedirs(os.path.dirname(os.path.abspath(fname_out)), exist_ok=True)
    if args.append and os.path.isfile(fname_out):
        df.to_csv(fname_out, mode='a', header=False, index=False)
    else:
        df.to_csv(fname_out, index=False)
    print(f'CSV file saved in {fname_out}.')


if __name__ == '__main__':
    main()