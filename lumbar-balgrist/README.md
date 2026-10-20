# Lumbar Balgrist

Lumbar spinal cord length (top of L1 to bottom of L5, or L1–S1 midpoints) from manually labeled spinal cord
segmentations (`sub-*/*_spinallevels_dseg.nii.gz`, 1=L1 ... 7=S2) in `~/data/spinal-levels/lumbar_Balgrist/bids`
(default `-i`). The CSV files and figures are saved to `~/data/spinal-levels/lumbar_Balgrist/results_and_figures`
(default `-o`).

| Script | What it does |
|---|---|
| `lumbar_length_count_slices.py` | Length = number of axial slices × slice thickness (assumes a straight cord). |
| `lumbar_length_along_centerline.py` | Length along the spinal cord centerline (accounts for the cord curvature). |
| `correlation_length_height.py` | Correlation between the length and subject height. |

```bash
python lumbar_length_count_slices.py
python lumbar_length_along_centerline.py
python correlation_length_height.py -i ~/data/spinal-levels/lumbar_Balgrist/results_and_figures/lumbar_length_L1-L5.csv
```

`lumbar_length_count_slices.py` and `lumbar_length_along_centerline.py` require SCT (`conda activate venv_sct`); add
`-metric midpoints` for the L1–S1 midpoints distance.
