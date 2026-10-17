# Lumbar Balgrist

Lumbar spinal cord length (top of L1 to bottom of L5, or L1–S1 midpoints) from manually labeled spinal cord
segmentations (`sub-*/*_spinallevels_dseg.nii.gz`, 1=L1 ... 7=S2) in `~/data/spinal-levels/lumbar_rootlets_Balgrist/bids`
(default `-i`). The CSV files and figures are saved to `~/data/spinal-levels/lumbar_rootlets_Balgrist/results_and_figures`
(default `-o`).

| Script | What it does |
|---|---|
| `lumbar_length_count_slices.py` | Length = number of axial slices × slice thickness (assumes a straight cord). |
| `lumbar_length_along_centerline.py` | Length along the spinal cord centerline (accounts for the cord curvature). |
| `correlation_length_height.py` | Correlation between the lumbar length and subject height. |
| `snap_conus.py` | Place the conus label (60) on the centerline at the CISS conus world-Z. |
| `compute_cord_length.py` | Whole cord length PMJ→conus = centerline arc length → `cord_length.csv`. |
| `correlation_cord_length.py` | Correlation of the whole cord length with height (and with the lumbar L1–L5 length), by sex. |

```bash
python lumbar_length_count_slices.py
python lumbar_length_along_centerline.py
python correlation_length_height.py -i ~/data/spinal-levels/lumbar_rootlets_Balgrist/results_and_figures/lumbar_length_L1-L5.csv
```

`lumbar_length_count_slices.py` and `lumbar_length_along_centerline.py` require SCT (`conda activate venv_sct`); add
`-metric midpoints` for the L1–S1 midpoints distance.

## Whole spinal cord length (PMJ to conus)

In addition to the lumbar length, the total spinal cord length from the pontomedullary junction (PMJ) to the conus
tip is measured along the cord centerline on the whole-spine localizers. The centerline is traced with SCT
(`sct_get_centerline -method viewer`) and the PMJ (label 50) placed manually; the conus (label 60) is read on the
high-resolution lumbar CISS, converted to a world-Z coordinate and snapped onto the centerline (`snap_conus.py`).
`compute_cord_length.py` then measures the arc length between the two labels, and `correlation_cord_length.py`
correlates it with height and with the lumbar L1–L5 length (Spearman, coloured by sex).

```bash
python snap_conus.py -ctl <subject>_label-centerline.csv -label <subject>_label-PMJtip_dlabel.nii.gz -z <conus_Z_mm>
python compute_cord_length.py -i <folder with *_label-centerline.csv and *_label-PMJtip_dlabel.nii.gz>
python correlation_cord_length.py -lumbar <results>/lumbar_length_L1-L5_centerline.csv
```
