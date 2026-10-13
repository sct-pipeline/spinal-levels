# Lumbar Balgrist

Lumbar spinal cord length (top of L1 to bottom of L5, or L1–S1 midpoints) from manually labeled spinal cord
segmentations (`sub-*/*_spinallevels_dseg.nii.gz`, 1=L1 ... 7=S2) in
`~/results/lumbar_rootlets_Balgrist/spinal_level_segmentation_L1-S2`.

| Script | What it does |
|---|---|
| `lumbar_length_count_slices.py` | Length = number of axial slices × slice thickness (assumes a straight cord). |
| `correlation_length_height.py` | Correlation between the length and subject height. |

```bash
python lumbar_length_count_slices.py -i ~/results/lumbar_rootlets_Balgrist/spinal_level_segmentation_L1-S2
python correlation_length_height.py -i ~/results/lumbar_rootlets_Balgrist/spinal_level_segmentation_L1-S2/lumbar_length_L1-L5.csv
```

`lumbar_length_count_slices.py` requires SCT (`conda activate venv_sct`); add `-metric midpoints` for the L1–S1
midpoints distance.
