"""
The script does the following:
    - obtain spinal levels, either
        rootlets (default): project the nerve rootlets on the spinal cord segmentation. This is done by dilating the
            spinal cord segmentation by selected number of voxels (input argument) and then finding the intersection
            between the dilated spinal cord segmentation and the rootlets segmentation. The spinal levels are then
            defined based on the top and bottom slice of the intersection.
        PAM50: register the image to the PAM50 template using the rootlets (sct_register_to_template -lrootlet),
            warp the template to the subject space (sct_warp_template) and use the warped PAM50 spinal levels.
    - compute the distance (along the cord centerline) of the start, end and midpoint of each spinal level from a
    reference point, chosen with -ref:
        pmj : pontomedullary junction (PMJ label required, -pmj)
        top : most superior slice of the SC segmentation
        c2  : top (most superior slice) of the C2 spinal level (label 2)

The script outputs .nii.gz file with spinal levels and saves the results in CSV files.

The script requires the SCT conda environment to be activated:
    source ${SCT_DIR}/python/etc/profile.d/conda.sh
    conda activate venv_sct

Examples:
    python rootlets_to_spinal_levels.py -i sub-001_T2w_label-rootlets_dseg.nii.gz -s sub-001_T2w_label-SC_seg.nii.gz -pmj sub-001_T2w_label-pmj.nii.gz
    python rootlets_to_spinal_levels.py -i sub-001_T2w_label-rootlets_dseg.nii.gz -s sub-001_T2w_label-SC_seg.nii.gz -ref top
    python rootlets_to_spinal_levels.py -i sub-001_T2w_label-rootlets_dseg.nii.gz -s sub-001_T2w_label-SC_seg.nii.gz -ref c2
    python rootlets_to_spinal_levels.py -i sub-001_T2w_label-rootlets_dseg.nii.gz -s sub-001_T2w_label-SC_seg.nii.gz -ref c2 \
        -method PAM50 -img sub-001_T2w.nii.gz -qc ./qc -qc-subject sub-001

OR, the script can be run using the wrapper script 01_run_batch_cervical_rootlets_spinal_levels.sh

NOTE: Modified from https://github.com/ivadomed/model-spinal-rootlets/blob/main/inter-rater_variability/02a_rootlets_to_spinal_levels.py
(branch kk/spinal-levels-length, commit 613674e, which added the distance_from_pmj_midpoint column).
"""

import os
import argparse
import subprocess
import numpy as np
import pandas as pd

from argparse import RawTextHelpFormatter
from spinalcordtoolbox.image import Image, zeros_like
from spinalcordtoolbox.centerline.core import ParamCenterline, get_centerline

# Label value of the C2 spinal level in the rootlets segmentation (and in the PAM50 spinal levels)
C2_LEVEL = 2

# Output folder of the PAM50 registration and path of the warped PAM50 spinal levels inside it.
# NOTE: check the file name against the output of sct_warp_template for your SCT version.
PAM50_OFOLDER = 'reg_rootlets'
PAM50_SPINAL_LEVELS = os.path.join('template', 'PAM50_spinal_levels.nii.gz')


def get_parser():
    """
    parser function
    """

    parser = argparse.ArgumentParser(
        description='The script does the following:'
                    '\n\t- obtain spinal levels from the rootlets (projection on the SC segmentation) or from the '
                    'PAM50 template registered using the rootlets'
                    '\n\t- compute the distance between a reference point (PMJ, top of the SC segmentation, or top '
                    'of the C2 spinal level) and the start, end and midpoint of each spinal level',
        formatter_class=RawTextHelpFormatter,
        prog=os.path.basename(__file__)
    )
    parser.add_argument(
        '-rootlets',
        required=True,
        help='Path to the spinal nerve rootlet segmentation.'
    )
    parser.add_argument(
        '-seg',
        required=True,
        help='Path to the spinal cord segmentation.'
    )
    parser.add_argument(
        '-pmj',
        required=False,
        help='Path to the pontomedullary junction (PMJ) label. Required when -ref pmj.'
    )
    parser.add_argument(
        '-ref',
        required=False,
        choices=['pmj', 'top', 'c2'],
        default=None,
        help='Reference point for the distances:'
             '\n\tpmj : pontomedullary junction (requires -pmj)'
             '\n\ttop : most superior slice of the SC segmentation'
             '\n\tc2  : top of the C2 spinal level (label ' + str(C2_LEVEL) + ')'
             '\nDefault: "pmj" if -pmj is provided, otherwise "top".'
    )
    parser.add_argument(
        '-dilate',
        required=False,
        type=int,
        help='Size of spinal cord segmentation dilation in pixels. Large number leads to "longer" spinal levels. '
             'Typical values: 1, 2 or 3. Default: 3. Used only with -method rootlets.',
        default=3,
    )
    parser.add_argument(
        '-method',
        required=False,
        choices=['rootlets', 'PAM50'],
        default='rootlets',
        help='How to obtain the spinal levels:'
             '\n\trootlets : projection of the rootlets on the SC segmentation (default)'
             '\n\tPAM50    : PAM50 spinal levels warped to the subject space (registration using the rootlets; '
             'requires -mri)'
    )
    parser.add_argument(
        '-mri',
        required=False,
        help='Path to the anatomical image (e.g. T2w). Required when -method PAM50.'
    )
    parser.add_argument(
        '-qc',
        required=False,
        help='Path to the QC folder (used by sct_register_to_template).'
    )
    parser.add_argument(
        '-qc-subject',
        dest='qc_subject',
        required=False,
        help='Subject ID for the QC report.'
    )

    return parser


def get_centerline_from_pmj(fname_seg, fname_pmj):
    """
    Generate extrapolated centerline from pontomedullary junction (PMJ).
    :param fname_seg: spinal cord segmentation
    :param fname_pmj: PMJ label
    :return: fname_centerline: path to the CSV file with the extrapolated centerline
    """
    # Inspiration: https://github.com/sct-pipeline/pmj-based-csa/blob/e362536a7bef17c0e151830cf777fb51fd09cb87/process_data.sh#L157-L160
    # Note: -v 2 is used to generate the extrapolated centerline CSV
    subprocess.run(['sct_process_segmentation', '-i', fname_seg, '-pmj', fname_pmj, '-pmj-distance', '50',
                    '-v', '2'], check=True)
    # Remove unnecessary png files and csa.csv
    os.system('rm -f *.png')
    os.system('rm -f csa.csv')

    fname_centerline = fname_seg.replace('.nii.gz', '_centerline_extrapolated.csv')

    return fname_centerline


def get_centerline_from_seg(im_seg):
    """
    Fit the centerline on the SC segmentation (RPI).
    :return: 3xN array (x, y, z) in voxel coordinates, sorted by ascending z (inferior -> superior)
    """
    param = ParamCenterline(algo_fitting='bspline', smooth=30, minmax=True)
    _, arr_ctl, _, _ = get_centerline(im_seg, param, verbose=0)
    return arr_ctl


def get_distance_along_centerline(centerline_points, px, py, pz):
    """
    Cumulative arc length (mm) along the centerline, measured from the most superior slice (= 0 mm) downward.
    Same output format as get_distance_from_pmj(): row 0 = distance, row 1 = z slice index.
    """
    z_index = centerline_points.shape[1] - 1   # top slice (centerline is sorted ascending in z)
    return get_distance_from_pmj(centerline_points, z_index, px, py, pz)


def intersect_seg_and_rootlets(im_rootlets, fname_seg, fname_rootlets, dilate_size):
    """
    Intersect the spinal cord segmentation and the spinal nerve rootlet segmentation.
    :param im_rootlets: Image object of the spinal nerve rootlet segmentation
    :param fname_seg: path to the spinal cord segmentation
    :param fname_rootlets: path to the spinal nerve rootlet segmentation
    :param dilate_size: size of spinal cord segmentation dilation in pixels
    :return: fname_intersect: path to the intersection between the spinal cord segmentation and the spinal nerve
    rootlet segmentation
    """

    # Dilate the SC segmentation using sct_maths
    fname_seg_dil = fname_seg.replace('.nii.gz', '_dil.nii.gz')
    subprocess.run(['sct_maths', '-i', fname_seg, '-o', fname_seg_dil, '-dilate', str(dilate_size)], check=True)

    # Load the dilated SC segmentation
    im_seg_dil = Image(fname_seg_dil).change_orientation('RPI')
    im_seg_dil_data = im_seg_dil.data

    # Intersect the rootlets and the dilated SC segmentation
    intersect_data = im_rootlets.data * im_seg_dil_data

    # Save the intersection using the Image class
    im_intersect = zeros_like(im_rootlets)
    im_intersect.data = intersect_data
    fname_intersect = fname_rootlets.replace('.nii.gz', '_intersect.nii.gz')
    im_intersect.save(fname_intersect)
    print(f'Intersection saved in {fname_intersect}.')

    return fname_intersect


def project_rootlets_to_segmentation(im_rootlets, im_seg, im_intersect, rootlets_levels, fname_rootlets):
    """"
    Project the nerve rootlets intersection on the spinal cord segmentation
    :param im_rootlets: Image object of the spinal nerve rootlet segmentation
    :param im_seg: Image object of the spinal cord segmentation
    :param im_intersect: Image object of the intersection between the spinal cord segmentation and the spinal nerve
    rootlet segmentation
    :param rootlets_levels: list of the spinal nerve rootlets levels
    :param fname_rootlets: path to the spinal nerve rootlet segmentation
    :return: fname_spinal_levels: path to the spinal levels segmentation
    :return: start_end_slices: dict of the spinal levels start (inferior) and end (superior) slices
    """
    im_spinal_levels_data = np.copy(im_seg.data)

    start_end_slices = dict()

    # Loop across the rootlets levels
    for level in rootlets_levels:
        # Get the list of slices where the level is present
        slices_list = np.unique(np.where(im_intersect.data == level)[2])
        # Skip the level if it is not present in the intersection
        if len(slices_list) != 0:
            min_slice = min(slices_list)
            max_slice = max(slices_list)
            start_end_slices[level] = {'start': min_slice, 'end': max_slice}
            # Color the SC segmentation with the level
            im_spinal_levels_data[:, :, min_slice:max_slice+1][im_seg.data[:, :, min_slice:max_slice+1] == 1] = level

    # Set zero to the slices with no intersection
    im_spinal_levels_data[im_spinal_levels_data == 1] = 0

    # Save the projection using the Image class
    im_spinal_levels = zeros_like(im_rootlets)
    im_spinal_levels.data = im_spinal_levels_data
    fname_spinal_levels = fname_rootlets.replace('.nii.gz', '_spinal_levels.nii.gz')
    im_spinal_levels.save(fname_spinal_levels)
    print(f'Spinal levels file saved in {fname_spinal_levels}.')

    return fname_spinal_levels, start_end_slices


def register_to_pam50(fname_img, fname_seg, fname_rootlets, ofolder, qc=None, qc_subject=None):
    """
    Register the image to the PAM50 template using the rootlets and warp the template to the subject space.
    The registration is skipped if the warping field already exists (e.g. the script is called once per reference).
    :return: path to the warped PAM50 spinal levels
    """
    fname_warp = os.path.join(ofolder, 'warp_template2anat.nii.gz')

    if os.path.isfile(fname_warp):
        print(f'Warping field {fname_warp} already exists, skipping sct_register_to_template.')
    else:
        cmd = ['sct_register_to_template', '-i', fname_img, '-s', fname_seg, '-lrootlet', fname_rootlets, '-ofolder',
               ofolder, '-qc', qc, '-qc-subject', qc_subject]
        subprocess.run(cmd, check=True)

    fname_spinal_levels = os.path.join(ofolder, PAM50_SPINAL_LEVELS)
    if not os.path.isfile(fname_spinal_levels):
        subprocess.run(['sct_warp_template', '-d', fname_img, '-w', fname_warp, '-a', '0', '-ofolder', ofolder], check=True)

    if not os.path.isfile(fname_spinal_levels):
        raise FileNotFoundError(f'{fname_spinal_levels} not found after sct_warp_template. Check the content of '
                                f'{os.path.join(ofolder, "template")} and update PAM50_SPINAL_LEVELS.')

    return fname_spinal_levels


def get_start_end_slices_from_levels(im_levels):
    """
    Get the start (inferior) and end (superior) slice of each spinal level from a spinal levels image (RPI).
    :param im_levels: Image object with integer labels (one value per spinal level)
    :return: start_end_slices: dict of the spinal levels start and end slices
    :return: levels: array of the spinal levels present in the image
    """
    data = np.rint(im_levels.data).astype(int)   # warped labels can be non-integer after interpolation
    levels = np.unique(data[data > 0])
    start_end_slices = dict()
    for level in levels:
        slices_list = np.unique(np.where(data == level)[2])
        start_end_slices[level] = {'start': slices_list.min(), 'end': slices_list.max()}
    return start_end_slices, levels


def get_distance_from_pmj(centerline_points, z_index, px, py, pz):
    """
    Compute distance from projected pontomedullary junction (PMJ) on centerline and cord centerline.
    Inspiration: https://github.com/sct-pipeline/pmj-based-csa/blob/419ece49c81782f23405d89c7b4b15d8e03ed4bd/get_distance_pmj_disc.py#L40-L60
    :param centerline_points: 3xn array: Centerline in continuous coordinate (float) for each slice in RPI orientation.
    :param z_index: z index PMJ on the centerline.
    :param px: x pixel size.
    :param py: y pixel size.
    :param pz: z pixel size.
    :return: nd-array: distance from PMJ and corresponding indexes.
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


def pmj_or_c2_top_dist(centerline_dist, start, end):
    """
    Compute the distance between the reference point (PMJ or C2 top) and the start and end of the spinal level
    :param centerline_dist: distance between the reference point and the centerline
    :param start: start slice of the spinal level
    :param end: end slice of the spinal level
    :return: dist_start: distance between the reference point and the start of the spinal level
    :return: dist_end: distance between the reference point and the end of the spinal level
    """
    if not np.isnan(start):
        dist_start = float(centerline_dist[0, centerline_dist[1] == start][0])
    else:
        dist_start = np.nan
    if not np.isnan(end):
        dist_end = float(centerline_dist[0, centerline_dist[1] == end][0])
    else:
        dist_end = np.nan
    return dist_start, dist_end


def build_distance_table(arr_distance, start_end_slices, rootlets_levels, fname_rootlets, col_prefix, clip):
    """
    Build a DataFrame with the distance of the start, end and midpoint of each spinal level from the reference point.
    :param arr_distance: 2xN array, row 0 = distance from the reference (mm), row 1 = z slice index
    :param col_prefix: column prefix, e.g. 'distance_from_pmj'
    :param clip: clip the level slices to the centerline extent (the dilated seg can reach past the seg ends)
    """
    z_min, z_max = arr_distance[1].min(), arr_distance[1].max()
    output_data = list()
    for level in rootlets_levels:
        print(f'Processing level {level}...')
        if level not in start_end_slices:
            print(f'WARNING: No intersection found for {level}.')
            continue
        start, end = start_end_slices[level]['start'], start_end_slices[level]['end']
        if clip:
            start, end = int(np.clip(start, z_min, z_max)), int(np.clip(end, z_min, z_max))

        # Slices are I-S (start = inferior), distances are S-I -> the inferior slice is the end of the level
        dist_end, dist_start = pmj_or_c2_top_dist(arr_distance, start, end)
        output_data.append({'spinal_level': level,
                            'fname': fname_rootlets,
                            'slice_start_I-S': start_end_slices[level]['start'],
                            'slice_end_I-S': start_end_slices[level]['end'],
                            f'{col_prefix}_start_S-I': dist_start,
                            f'{col_prefix}_end_S-I': dist_end,
                            f'{col_prefix}_midpoint': (dist_start + dist_end) / 2,
                            'height': dist_end - dist_start})
    return pd.DataFrame(output_data)


def main():
    # Parse the command line arguments
    parser = get_parser()
    args = parser.parse_args()

    fname_rootlets = args.rootlets
    fname_seg = args.seg
    fname_mri_img = args.mri
    dilate_size = args.dilate
    method = args.method

    # Resolve the reference point
    ref = args.ref if args.ref is not None else ('pmj' if args.pmj else 'top')
    if ref == 'pmj' and not args.pmj:
        parser.error('-ref pmj requires the -pmj label.')
    if method == 'PAM50' and not fname_mri_img:
        parser.error('-method PAM50 requires the anatomical image (-img).')

    # Load input images using the SCT Image class
    im_rootlets = Image(fname_rootlets).change_orientation('RPI')
    im_seg = Image(fname_seg).change_orientation('RPI')

    # Check if the SC seg is binary
    if len(np.unique(im_seg.data)) != 2:
        raise ValueError('The spinal cord segmentation should be binary.')

    if method == 'rootlets':
        # Intersect the rootlets and the SC segmentation
        fname_intersect = intersect_seg_and_rootlets(im_rootlets, fname_seg, fname_rootlets, dilate_size)

        # Load the intersection
        im_intersect = Image(fname_intersect).change_orientation('RPI')

        # Get unique values in the rootlets segmentation larger than 0
        levels = np.unique(im_rootlets.data[np.where(im_rootlets.data > 0)])

        # Project the nerve rootlets intersection on the spinal cord segmentation to obtain spinal levels
        fname_spinal_levels, start_end_slices = project_rootlets_to_segmentation(im_rootlets, im_seg, im_intersect,
                                                                                 levels, fname_rootlets)
        csv_suffix = '_rootlets-only'

    elif method == 'PAM50':
        # Register to PAM50 using the rootlets and warp the PAM50 spinal levels to the subject space
        fname_spinal_levels = register_to_pam50(fname_mri_img, fname_seg, fname_rootlets, PAM50_OFOLDER,
                                                qc=args.qc, qc_subject=args.qc_subject)
        im_levels = Image(fname_spinal_levels).change_orientation('RPI')
        start_end_slices, levels = get_start_end_slices_from_levels(im_levels)
        print(f'PAM50 spinal levels (subject space): {fname_spinal_levels}')
        csv_suffix = '_PAM50'

    if ref == 'pmj':
        im_pmj = Image(args.pmj).change_orientation('RPI')

        # Check if the PMJ label file is not empty
        if len(np.unique(im_pmj.data)) == 1:
            raise ValueError('The PMJ label file is empty.')

        # Generate extrapolated centerline from PMJ
        fname_centerline = get_centerline_from_pmj(fname_seg, args.pmj)

        # Load CSV file with centerline coordinates generated by the previous command as an array
        centerline = np.genfromtxt(fname_centerline, delimiter=',')
        # Compute distance from PMJ of the centerline
        arr_distance = get_distance_from_pmj(centerline, centerline[2].argmax(), im_pmj.dim[4], im_pmj.dim[5],
                                             im_pmj.dim[6])

        df = build_distance_table(arr_distance, start_end_slices, levels, fname_rootlets,
                                  col_prefix='distance_from_pmj', clip=False)
        fname_out = fname_rootlets.replace('.nii.gz', f'_pmj_distance{csv_suffix}.csv')

    elif ref == 'c2':
        # Distance along the centerline fitted on the SC segmentation, 0 mm = top of the segmentation
        centerline = get_centerline_from_seg(im_seg)
        arr_distance = get_distance_along_centerline(centerline, im_seg.dim[4], im_seg.dim[5], im_seg.dim[6])
        if C2_LEVEL not in start_end_slices:
            raise ValueError(f'C2 spinal level (label {C2_LEVEL}) not found in the spinal levels.')
        # Top of C2 = most superior slice of the C2 level ('end', since RPI z increases superiorly)
        z_min, z_max = arr_distance[1].min(), arr_distance[1].max()
        c2_top_slice = int(np.clip(start_end_slices[C2_LEVEL]['end'], z_min, z_max))
        c2_top_dist = float(arr_distance[0, arr_distance[1] == c2_top_slice][0])
        print(f'Top of C2 spinal level: slice {c2_top_slice}, {c2_top_dist:.2f} mm below the SC seg top.')

        # Shift so that the top of C2 = 0 mm (levels above C2, e.g. C1, get negative values)
        arr_distance = arr_distance.copy()
        arr_distance[0] -= c2_top_dist

        df = build_distance_table(arr_distance, start_end_slices, levels, fname_rootlets,
                                  col_prefix='distance_from_c2_top', clip=True)
        fname_out = fname_rootlets.replace('.nii.gz', f'_c2_distance{csv_suffix}.csv')

    # Save the DataFrame as a CSV file
    df.to_csv(fname_out, index=False)
    print(f'CSV file saved in {fname_out}.')


if __name__ == '__main__':
    main()