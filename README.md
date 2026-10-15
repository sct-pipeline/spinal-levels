# Spinal levels

In-vivo analysis of spinal levels from MRI images.

Measure the distance between spinal level midpoints along the spinal cord:

- **Cervical: C2–C8**, based on the spinal rootlets segmentation
- **Lumbar: L1–S1**, based on manually labeled spinal levels

All scripts require SCT (they use the SCT Python environment).

## Scripts

| Script | What it does |
|---|---|
| `01_run_batch_rootlets_spinal_levels.sh` | Cervical, per subject (run with `sct_run_batch`): spinal cord segmentation, PMJ detection, rootlets segmentation, and spinal levels with their distances from the PMJ. Manual labels in `derivatives/labels` are used when available. |
| `02_compute_cervical_midpoints_distance.py` | Cervical: aggregates the results across subjects, computes the C2–C8 midpoints distance and plots it. |
| `config_01_run_batch_rootlets_spinal_levels.json` | `sct_run_batch` config listing the spine-generic subjects with manual rootlets and spinal cord segmentations. |

Spinal levels for the cervical analysis are computed by
[`02a_rootlets_to_spinal_levels.py`](02a_rootlets_to_spinal_levels.py),
called from the batch script.

## Cervical: C2–C8

Spinal levels are obtained from the intersection of the rootlets and the spinal cord segmentation. The midpoint of
each level is the average of the distances of its start and end from the pontomedullary junction (PMJ). The C2–C8
distance is then `distance_from_pmj_midpoint` of C8 minus `distance_from_pmj_midpoint` of C2.

1. Download the data (the images are stored with git-annex):

```bash
cd ~/data/data.neuro.polymtl.ca/data-multi-subject
git annex get sub-*/anat/*_T2w.nii.gz derivatives/labels/sub-*/anat/*_T2w_label-*
```

2. Run the per-subject processing (edit `path_data` and `path_output` in the config if needed):

```bash
sct_run_batch -config config_01_run_batch_rootlets_spinal_levels.json
```

3. Aggregate the results:

```bash
python 02_compute_cervical_midpoints_distance.py -i <path_output>/data_processed
```

Outputs per subject (in `data_processed/<subject>/anat/`):

- `*_label-rootlets_dseg_spinal_levels.nii.gz`: spinal levels projected on the spinal cord
- `*_label-rootlets_dseg_pmj_distance.csv`: per-level start, end and midpoint distances from the PMJ

Outputs across subjects (in the folder passed to `-i`):

- `spinal_levels_spine-generic.csv`: per-level distances for all subjects
- `spinal_levels_midpoints_distance_2-8_spine-generic.csv`: C2–C8 midpoints distance (mm) per subject
- `figure_spinal_levels_midpoints_distance_2-8.png`: C2–C8 midpoints distance per subject

## Lumbar Balgrist dataset

Scripts for the lumbar Balgrist dataset are in [`lumbar-balgrist/`](lumbar-balgrist). Input: manually labeled spinal
cord segmentations (`sub-*/*_spinallevels_dseg.nii.gz`) with one value per spinal level: 1=L1, 2=L2, 3=L3, 4=L4, 5=L5,
6=S1, 7=S2.

| Script | What it does |
|---|---|
| `lumbar-balgrist/lumbar_length_count_slices.py` | Computes the lumbar spinal cord length by counting axial slices (`-metric top-bottom`: top of L1 to bottom of L5, default; `-metric midpoints`: L1–S1 midpoints distance) and plots it per subject. |
| `lumbar-balgrist/lumbar_length_along_centerline.py` | Same lengths, but measured along the spinal cord centerline (SCT `get_centerline`, `ParamCenterline`: bspline, smooth=30, fitted to the L1–S2 segmentation) to account for the cord curvature; also saves the fitted centerlines for QC. |
| `lumbar-balgrist/correlation_length_height.py` | Plots the correlation between the length and subject height. |

```bash
python lumbar-balgrist/lumbar_length_count_slices.py -i <folder with sub-*/*_spinallevels_dseg.nii.gz> -o <output folder>
python lumbar-balgrist/correlation_length_height.py -i <output folder>/lumbar_length_L1-L5.csv
```

Outputs (in the folder passed to `-o`, default `~/data/spinal-levels/lumbar_rootlets_Balgrist/results_and_figures`):
`lumbar_<metric>.csv` (length per subject, and the min slice, max slice and
midpoint slice of each spinal level) and `figure_lumbar_<metric>.png`.

## Lumbar EPFL dataset

Scripts for the lumbar EPFL dataset (`~/data/spinal-levels/lumbar_mri_Sergio/spinalcord_masks`) are in
[`lumbar-epfl/`](lumbar-epfl). Input: spinal cord masks with the personalized spinal levels saved one level per file
(`sub-*/ses-*/anat/*_label-SC_desc-<level>_mask.nii.gz`, `desc-L1` ... `desc-S2`), and `participants.tsv` (sex and
height) one level above the masks folder.

| Script | What it does |
|---|---|
| `lumbar-epfl/lumbar_length_count_slices.py` | Computes the lumbar spinal cord length (`-metric top-bottom`: top of L1 to bottom of L5, default; `-metric midpoints`: L1–S1 midpoints distance), plots it per subject and its correlation with height. |
| `lumbar-epfl/lumbar_length_along_centerline.py` | Same lengths, but measured along the spinal cord centerline (SCT `get_centerline`, `ParamCenterline`: bspline, smooth=30) to account for the cord curvature; also saves the fitted centerlines for QC. |

```bash
python lumbar-epfl/lumbar_length_count_slices.py -i ~/data/spinal-levels/lumbar_mri_Sergio/spinalcord_masks
```

Outputs (in the current working directory): `lumbar_<metric>_masks.csv`, `figure_lumbar_<metric>_masks.png` and
`figure_correlation_lumbar_<metric>_masks_height.png`.

## Lumbar Balgrist + EPFL + Fudan combined

[`lumbar-combined/combine_balgrist_epfl.py`](lumbar-combined/combine_balgrist_epfl.py) combines the L1–L5 length along
the centerline of the Balgrist and EPFL datasets (CSV files from the two `lumbar_length_along_centerline.py` scripts)
and the Fudan dataset (`~/data/spinal-levels/lumbar_fudan/L1-L5_lengths.xlsx`) into a per-subject
figure (stratified by sex, sorted by height) and a correlation with height (all subjects pooled). Sex is shown by color
and the dataset by marker.

```bash
python lumbar-combined/combine_balgrist_epfl.py
```

Outputs (in the folder passed to `-o`, default `~/data/spinal-levels/results_and_figures_balgrist_epfl_fudan`):
`lumbar_length_L1-L5_balgrist_epfl_fudan.csv`, `figure_lumbar_length_L1-L5_balgrist_epfl_fudan.png` and
`figure_correlation_lumbar_length_L1-L5_balgrist_epfl_fudan_height.png`.
