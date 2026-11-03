# Lumbar Marseille

Lumbar spinal cord length (top of L1 to bottom of L5) from manual rootlets and spinal cord segmentations
(`~/data/data.neuro.polymtl.ca/lumbar-marseille/derivatives/labels`).

| Script | What it does |
|---|---|
| `rootlets_to_spinal_levels.py` | Spinal levels L1–L5 from the rootlets ZEP (Zone of Entry of the rootlets), following the SOP Rootlet segmentation (Balgrist Campus, v1.0 2026). Rootlet values follow the SCT convention (L1 = 21, ..., L5 = 25, S1 = 26). |
| `lumbar_length_along_centerline.py` | Length from the top of L1 to the bottom of L5 along the spinal cord centerline (same measurement as `lumbar-epfl/lumbar_length_along_centerline.py`). |

```bash
python rootlets_to_spinal_levels.py -i ~/data/data.neuro.polymtl.ca/lumbar-marseille/derivatives/labels -ofolder ~/data/spinal-levels/lumbar-marseille
python lumbar_length_along_centerline.py -i ~/data/spinal-levels/lumbar-marseille
python ../lumbar-combined/combine_lumbar_lengths.py -sort-by length
```

Both scripts require SCT (`conda activate venv_sct`). `participants.tsv` has no sex or height, so the Marseille subjects
are shown as "Sex unknown" in the combined per-subject figure (`-sort-by length` only) and are excluded from the
correlation with height.

## Left/right ZEP discrepancy

The L1–L5 length depends only on two ZEPs: the top of L1 (ZEP-L1) and the bottom of L5 (one slice above ZEP-S1).

The ZEP-L1 and ZEP-S1 differ between the left and right side by up to 21 slices (0.5 mm slices). Length (number of
slices × 0.5 mm; within ~0.5 mm of the centerline length) using the left-side ZEPs only vs. the right-side ZEPs only:

| Subject | ZEP-L1 L/R diff [slices] | ZEP-S1 L/R diff [slices] | Left only [mm] | Right only [mm] | Used [mm] | Left − right [mm] |
|---|---|---|---|---|---|---|
| sub-CTS04 | 9 | 4 | 41.5 | 35.0 | 38.5 | 6.5 (17%) |
| sub-CTS05 | 1 | 0 | 36.0 | 35.5 | 36.1 | 0.5 (1%) |
| sub-CTS09 | 6 | 5 | 49.0 | 48.5 | 48.5 | 0.5 (1%) |
| sub-CTS10 | 12 | 4 | 35.0 | 39.0 | 37.0 | −4.0 (−11%) |
| sub-CTS14 | 7 | 8 | 49.0 | 48.5 | 49.0 | 0.5 (1%) |
| sub-CTS15 | 21 | 6 | 34.5 | 42.0 | 38.6 | −7.5 (−20%) |
| sub-CTS20 | 8 | 20 | 52.5 | 38.5 | 45.6 | 14.0 (31%) |

- **To be checked (4–14 mm, 11–31%):** the length depends on the side used (the reported length is roughly the middle of the two):
    - sub-CTS20: ZEP-S1 (20 slices)
    - sub-CTS15: ZEP-L1 (21 slices)
    - sub-CTS10: ZEP-L1 (12 slices)
    - sub-CTS04: ZEP-L1 (9 slices)
