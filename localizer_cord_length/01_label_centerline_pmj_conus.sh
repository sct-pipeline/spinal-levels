#!/bin/bash
#
# This script creates the manual labels used to compute the spinal cord length (PMJ -> conus) on the whole-spine
# sagittal localizer (*_acq-localizerSag_T2w). For each subject:
# - manual centerline (sct_get_centerline -method viewer): click the cord every 10 mm
# - PMJ (label 50) and conus (label 60) (sct_label_utils -create-viewer)
#   If the conus world Z is known from the CISS image (CONUS_CSV), only the PMJ is clicked and the conus is placed
#   automatically on the centerline at that level (using 01a_place_conus_from_ciss.py).
# - sagittal QC (sct_qc) of the centerline, the PMJ and the conus: one entry each, plus one entry with all three
#   (using 01b_merge_labels_for_qc.py)
#
# The labels are saved to derivatives/labels/sub-*/ses-*/anat/ of the input dataset. Existing labels are kept (delete
# the file to redo it). The cord length is then computed by 02_compute_cord_length.py.
#
# The script needs a display (interactive viewers) and the SCT conda environment:
#   source ${SCT_DIR}/python/etc/profile.d/conda.sh
#   conda activate venv_sct
#
# Usage:
#   bash 01_label_centerline_pmj_conus.sh <PATH_DATA> [sub-ltr01 sub-ltr02 ...]
#   CONUS_CSV=tip_conus_CISS_worldZ.csv bash 01_label_centerline_pmj_conus.sh <PATH_DATA>
#
#   PATH_DATA: BIDS dataset (all subjects are processed if no subject is given)
#   CONUS_CSV (optional): CSV file with the conus world Z from the CISS image, with columns 'subject' (subject number,
#             e.g., 3 for sub-ltr03) and 'world_z_mm_ciss_tip' (in mm)
#   PATH_QC (optional): folder for the QC report (default: qc_localizer_cord_length next to PATH_DATA). Each run adds
#             new QC entries; delete the folder to start a clean report.
#
# Context: https://github.com/sct-pipeline/spinal-levels/issues/4
#
# Authors: Elia Rochiccioli, Jan Valosek

# Immediately exit if error
set -e -o pipefail

PATH_DATA=$(cd "$1" && pwd)
shift
PATH_SCRIPTS=$(cd "$(dirname "$0")" && pwd)
PATH_LABELS="${PATH_DATA}/derivatives/labels"
PATH_QC=${PATH_QC:-"$(dirname "${PATH_DATA}")/qc_localizer_cord_length"}
PATH_TMP=$(mktemp -d)  # files used only for the QC
# Text labels for the QC entry with the centerline, the PMJ and the conus (values from 01b_merge_labels_for_qc.py)
echo '{"1": "centerline", "2": "PMJ", "3": "conus"}' > "${PATH_TMP}/qc_labels.json"
GAP=10  # mm between manually clicked centerline points

cd "${PATH_DATA}"
if [ $# -gt 0 ]; then SUBJECTS="$@"; else SUBJECTS=$(ls -d sub-*); fi

for SUBJECT in $SUBJECTS; do
  for file_path in ${SUBJECT}/ses-*/anat/${SUBJECT}_ses-*_acq-localizerSag_T2w.nii.gz; do
    [ -f "$file_path" ] || { echo "No localizer for ${SUBJECT}"; continue; }
    file=$(basename "$file_path" .nii.gz)
    PATH_OUT="${PATH_LABELS}/$(dirname "$file_path")"
    mkdir -p "${PATH_OUT}"
    FILECENTERLINE="${PATH_OUT}/${file}_label-centerline"
    FILELABELS="${PATH_OUT}/${file}_label-PMJtip_dlabel.nii.gz"
    echo "=== ${file} ==="

    # Manual centerline (the .csv file with the continuous centerline is used to compute the cord length)
    if [ ! -f "${FILECENTERLINE}.csv" ]; then
      sct_get_centerline -i "$file_path" -method viewer -gap $GAP -space phys -o "${FILECENTERLINE}.nii.gz"
    fi

    # PMJ and conus labels
    if [ ! -f "${FILELABELS}" ]; then
      # Conus world Z from the CISS image (empty if CONUS_CSV is not set or the subject is not listed)
      CONUS_Z=""
      if [ -n "${CONUS_CSV}" ]; then
        CONUS_Z=$(awk -F, -v num="${SUBJECT#sub-ltr}" 'NR>1 && $1+0 == num+0 {print $2}' "${CONUS_CSV}")
      fi
      # Note: the SCT viewer can exit with a non-zero code when closed, hence "|| true"
      if [ -n "${CONUS_Z}" ]; then
        sct_label_utils -i "$file_path" -create-viewer 50 \
          -msg "Click 50 = PMJ (pontomedullary junction). The conus (60) is placed automatically from the CISS." \
          -o "${FILELABELS}" || true
        python "${PATH_SCRIPTS}/01a_place_conus_from_ciss.py" -labels "${FILELABELS}" \
          -centerline "${FILECENTERLINE}.csv" -z "${CONUS_Z}"
      else
        sct_label_utils -i "$file_path" -create-viewer 50,60 \
          -msg "Click 50 = PMJ (pontomedullary junction), then 60 = tip of the spinal cord (conus)" \
          -o "${FILELABELS}" || true
      fi
    fi

    # QC: sagittal view following the centerline, and the PMJ and the conus labels (each on its own sagittal slice)
    if [ -f "${FILECENTERLINE}.nii.gz" ] && [ -f "${FILELABELS}" ]; then
      QC_ARGS=(-qc "${PATH_QC}" -qc-dataset "$(basename "${PATH_DATA}")" -qc-subject "${SUBJECT}"
               -qc-contrast localizerSag_T2w)
      # -text-labels 0: do not draw vertebral level names (C1, ...) for the centerline value
      sct_qc -i "$file_path" -s "${FILECENTERLINE}.nii.gz" -p sct_label_vertebrae -text-labels 0 "${QC_ARGS[@]}"
      sct_label_utils -i "${FILELABELS}" -keep 50 -o "${PATH_TMP}/${file}_pmj.nii.gz"
      sct_label_utils -i "${FILELABELS}" -keep 60 -o "${PATH_TMP}/${file}_conus.nii.gz"
      sct_qc -i "$file_path" -s "${PATH_TMP}/${file}_pmj.nii.gz" -p sct_label_utils "${QC_ARGS[@]}"
      sct_qc -i "$file_path" -s "${PATH_TMP}/${file}_conus.nii.gz" -p sct_label_utils "${QC_ARGS[@]}"
      # All three in a single entry
      python "${PATH_SCRIPTS}/01b_merge_labels_for_qc.py" -centerline "${FILECENTERLINE}.nii.gz" \
        -labels "${FILELABELS}" -o "${PATH_TMP}/${file}_merged.nii.gz"
      sct_qc -i "$file_path" -s "${PATH_TMP}/${file}_merged.nii.gz" -p sct_label_vertebrae \
        -custom-labels "${PATH_TMP}/qc_labels.json" "${QC_ARGS[@]}"
    fi
  done
done

rm -rf "${PATH_TMP}"
echo "QC report: ${PATH_QC}/index.html"
