# Lumbar EPFL

Lumbar spinal cord length (top of L1 to bottom of L5, or L1–S1 midpoints) from the spinal level masks in
`~/data/spinal-levels/lumbar_epfl/spinalcord_masks`.

| Script | What it does |
|---|---|
| `lumbar_length_count_slices.py` | Length = number of axial slices × slice thickness (assumes a straight cord). |
| `lumbar_length_along_centerline.py` | Length along the spinal cord centerline (accounts for the cord curvature). |

```bash
python lumbar_length_count_slices.py -i ~/data/spinal-levels/lumbar_epfl/spinalcord_masks
python lumbar_length_along_centerline.py -i ~/data/spinal-levels/lumbar_epfl/spinalcord_masks
```

Both scripts require SCT (`conda activate venv_sct`); add `-metric midpoints` for the L1–S1 midpoints distance.
