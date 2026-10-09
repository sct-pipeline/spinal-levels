#!/bin/bash
#
# This script performs:
# - segmentation of spinal cord from T2w data (seg_sc_contrast_agnostic)
# - detection of PMJ from T2w data (sct_detect_pmj) -- only when the PMJ reference is requested
# - finding the rootlets segmentation (if it exists)
# - computing the spinal levels of the rootlets and distances of the spinal levels from the chosen reference:
#     pmj : pontomedullary junction
#     top : top of the spinal cord segmentation (centerline top)
#     c2  : top of the C2 spinal level (rootlets label 2)
#     all : all three of the above
#
# Expected file naming:
#   image:        sub-XXX_<contrast>.nii.gz
#   SC seg:       sub-XXX_<contrast>_label-SC_seg.nii.gz
#   PMJ:          sub-XXX_<contrast>_label-pmj.nii.gz
#   rootlets:     sub-XXX_<contrast>_label-rootlets_dseg.nii.gz

# NOTE: This script is inspired by the script 'inter-rater_variability/02_run_batch_inter_rater_variability.sh'
# https://github.com/ivadomed/model-spinal-rootlets/blob/main/inter-rater_variability/02_run_batch_inter_rater_variability.sh

# This script used the script '02a_rootlets_to_spinal_levels.py' (to get spinal levels) available at:
# https://github.com/ivadomed/model-spinal-rootlets/blob/main/inter-rater_variability/02a_rootlets_to_spinal_levels.py

# Usage:
## sct_run_batch -script analysis_preprocess_pipeline.sh
##                     -script-args "<REF>"            # pmj | top | c2 | all  (default: all)
##                     -path-data <DATA>
##                     -path-output <DATA>_202X-XX-XX
##                     -jobs 5
##
## Examples:
##   sct_run_batch -script analysis_preprocess_pipeline.sh -script-args "c2"  -path-data <DATA> -path-output <OUT>
##   sct_run_batch -script analysis_preprocess_pipeline.sh -script-args "all" -path-data <DATA> -path-output <OUT>

# Authors: Katerina Krejci


# Uncomment for full verbose
set -x

# Immediately exit if error
set -e -o pipefail

# Exit if user presses CTRL+C (Linux) or CMD+C (OSX)
trap "echo Caught Keyboard Interrupt within script. Exiting now.; exit" INT

# Retrieve input params
SUBJECT=${1%%/*}
# Reference for the distances: 2nd argument (sct_run_batch -script-args), else REF env variable, else "all"
REF=${2:-${REF:-all}}
REF=$(echo "$REF" | tr '[:upper:]' '[:lower:]')   # accept C2, PMJ, TOP, ...

case $REF in
  pmj|top|c2) REF_LIST="$REF" ;;
  all)        REF_LIST="pmj top c2" ;;
  *)          echo "ERROR: unknown reference '$REF'. Use one of: pmj, top, c2, all."; exit 1 ;;
esac
echo "Reference(s) for spinal level distances: ${REF_LIST}"

# Get starting time:
start=`date +%s`


# FUNCTIONS
# ==============================================================================
# Uses global variables:
#   base     = sub-XXX_<contrast>          (e.g. sub-amuAL_T2w)
#   file     = sub-XXX_<contrast>          (image, without .nii.gz)
#   contrast = T1w | T2w

# Segment spinal cord if it does not exist in the derivatives folder
segment_sc_if_does_not_exist(){
  FILESEG="${base}_label-SC_seg"
  FILESEGMANUAL="${PATH_DATA}/derivatives/labels/${SUBJECT}/anat/${FILESEG}.nii.gz"
  echo
  echo "Looking for manual segmentation: $FILESEGMANUAL"
  if [[ -e $FILESEGMANUAL ]]; then
    echo "Found! Using manual segmentation."
    rsync -avzh $FILESEGMANUAL ${FILESEG}.nii.gz
    sct_image -i ${FILESEG}.nii.gz -setorient RPI -o ${FILESEG}.nii.gz
    sct_qc -i ${file}.nii.gz -s ${FILESEG}.nii.gz -p sct_deepseg_sc -qc ${PATH_QC} -qc-subject ${SUBJECT}
  else
    echo "Not found. Proceeding with automatic segmentation."
    CUDA_VISIBLE_DEVICES=0 SCT_USE_GPU=1  sct_deepseg -task seg_sc_contrast_agnostic -i ${file}.nii.gz -qc ${PATH_QC} -qc-subject ${SUBJECT} -o ${FILESEG}.nii.gz
  fi
}

# Detect PMJ if it does not exist in the derivatives folder
detect_pmj_if_does_not_exist(){
  FILEPMJ="${base}_label-pmj"
  FILEPMJMANUAL="${PATH_DATA}/derivatives/labels/${SUBJECT}/anat/${FILEPMJ}.nii.gz"
  echo
  echo "Looking for manual PMJ detection: $FILEPMJMANUAL"
  if [[ -e $FILEPMJMANUAL ]]; then
    echo "Found! Using manual PMJ detection."
    rsync -avzh $FILEPMJMANUAL ${FILEPMJ}.nii.gz
    sct_image -i ${FILEPMJ}.nii.gz -setorient RPI -o ${FILEPMJ}.nii.gz
    sct_qc -i ${file}.nii.gz -s ${FILEPMJ}.nii.gz -p sct_detect_pmj -qc ${PATH_QC} -qc-subject ${SUBJECT}
  else
    echo "Not found. Proceeding with automatic PMJ detection."

    # if the contrast is T2w, use t2 for PMJ detection; otherwise use t1
    if [[ $contrast == "T2w" ]]; then
      sct_detect_pmj -i ${file}.nii.gz -s ${FILESEG}.nii.gz -c t2 -o ${FILEPMJ}.nii.gz -qc ${PATH_QC} -qc-subject ${SUBJECT}
    else
      sct_detect_pmj -i ${file}.nii.gz -s ${FILESEG}.nii.gz -c t1 -o ${FILEPMJ}.nii.gz -qc ${PATH_QC} -qc-subject ${SUBJECT}
    fi
  fi
}

# Copy rootlets segmentation if it exists in the derivatives folder
copy_rootlets_if_exist(){
  FILESEGROOTLETS="${base}_label-rootlets_dseg"
  FILESEGROOTLETSMANUAL="${PATH_DATA}/derivatives/labels/${SUBJECT}/anat/${FILESEGROOTLETS}.nii.gz"
  echo
  echo "Looking for manual rootlets segmentation: $FILESEGROOTLETSMANUAL"
  if [[ -e $FILESEGROOTLETSMANUAL ]]; then
    echo "Found! Using manual segmentation."
    rsync -avzh $FILESEGROOTLETSMANUAL ${FILESEGROOTLETS}.nii.gz
    sct_qc -i ${file}.nii.gz -s ${FILESEG}.nii.gz -d ${FILESEGROOTLETS}.nii.gz -p sct_deepseg_lesion -qc ${PATH_QC} -qc-subject ${SUBJECT} -plane axial
  else
    echo "Not found. Creating automatic rootlets segmentation."
    CUDA_VISIBLE_DEVICES=0 SCT_USE_GPU=1 sct_deepseg rootlets -i ${file}.nii.gz -qc ${PATH_QC} -qc-subject ${SUBJECT} -o ${FILESEGROOTLETS}.nii.gz
  fi
}

# Get rootlets spinal levels and distances from the given reference (pmj | top | c2),
# then copy the resulting CSV to the results folder
# Note: we use SCT python because the `02a_rootlets_to_spinal_levels.py` script imports some SCT classes
run_spinal_levels(){
  local ref=$1
  local ref_name pmj_arg csv_suffix
  case $ref in
    pmj) ref_name="PMJ";            pmj_arg="-pmj ${FILEPMJ}.nii.gz"; csv_suffix="pmj_distance" ;;
    top) ref_name="centerline top"; pmj_arg="";                       csv_suffix="centerline_distance" ;;
    c2)  ref_name="C2 top";         pmj_arg="";                       csv_suffix="c2_distance" ;;
  esac

  echo "👉 Getting spinal levels and distances from the ${ref_name}..."
  $SCT_DIR/python/envs/venv_sct/bin/python ~/PycharmProjects/spinal-levels/02a_rootlets_to_spinal_levels.py \
    -i ${FILESEGROOTLETS}.nii.gz -s ${FILESEG}.nii.gz ${pmj_arg} -dilate 3 -ref ${ref}

  # Copy the CSV file with the spinal levels distances to the results folder (used by
  # 02_compute_cervical_midpoints_distance.py), so that only the results folder needs to be copied from the server
  rsync -avzh ${FILESEGROOTLETS}_${csv_suffix}.csv ${PATH_RESULTS}/
}

# SCRIPT STARTS HERE
# ==============================================================================
# Display useful info for the log, such as SCT version, RAM and CPU cores available
sct_check_dependencies -short

# Go to folder where data will be copied and processed
cd $PATH_DATA_PROCESSED

# Copy source images
rsync -Ravzh ${PATH_DATA}/./${SUBJECT}/anat/${SUBJECT//[\/]/_}_*.* .

# Go to anat folder where all data are located
cd ${SUBJECT}/anat

echo "SUBJECT=${SUBJECT}"
echo "PWD=$(pwd)"
ls

# Can be extended to T1w usage (i.e. for contrast in T1w T2w; do)
for contrast in T2w; do
    base="${SUBJECT//[\/]/_}_${contrast}"      # e.g. sub-amuAL_T2w
    file="${base}"

    if [[ ! -f ${file}.nii.gz ]]; then
        echo "WARNING: ${file}.nii.gz not found in $(pwd), skipping."
        continue
    fi

    # Segment spinal cord (only if it does not exist)
    segment_sc_if_does_not_exist

    # Detect PMJ (only if it does not exist and only if the PMJ reference is requested)
    if [[ " ${REF_LIST} " == *" pmj "* ]]; then
        detect_pmj_if_does_not_exist
    fi

    # Copy the rootlets segmentation if it exists
    copy_rootlets_if_exist

    # Spinal levels + distances for each requested reference
    for ref in ${REF_LIST}; do
        run_spinal_levels ${ref}
    done

done

# Display useful info for the log
end=`date +%s`
runtime=$((end-start))
echo
echo "~~~"
echo "SCT version: `sct_version`"
echo "Reference(s): ${REF_LIST}"
echo "Ran on:      `uname -nsr`"
echo "Duration:    $(($runtime / 3600))hrs $((($runtime / 60) % 60))min $(($runtime % 60))sec"
echo "~~~"
