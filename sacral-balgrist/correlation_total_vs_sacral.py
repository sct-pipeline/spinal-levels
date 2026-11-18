"""
Plot the correlation between the total spinal cord length (PMJ to conus, whole-cord) and the sacral spinal cord
length (top of S1 to conus) for the Balgrist lumbar (14_LTR) cohort, using the shared correlation figure
(visualisation_utils.plot_correlation), so the figure matches the cervical and combined panels (sex by color,
dataset by marker, linear fit + 95% CI band, Spearman rho [95% CI] in the upper-left corner, legend above).

This answers "is a longer whole cord associated with a longer sacral segment?" (the sacral analogue of the
cervical-combined `-pmj-to-conus` cord-length-vs-cervical-length figure).

Input:
- -sacral : sacral length CSV from sacral_length_along_centerline.py
            (subject, length_S1-conus_centerline_mm, sex, height).
- -total  : total cord length CSV (PMJ to conus) with 'participant_id' and 'cord_length_mm' columns
            (derivatives/labels/cord_length.csv).

Output (saved to the sacral CSV folder):
- figure_correlation_total_vs_sacral_length.png

Example:
    python correlation_total_vs_sacral.py

Author: Elia Rochiccioli
"""

import os
import sys
import argparse
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from visualisation_utils import plot_correlation, spearman_ci   # noqa: E402

DATASET = 'Balgrist'
SACRAL_CSV = os.path.expanduser('~/Desktop/Balgrist Dataset/sacral_length/sacral_length_S1-conus_centerline.csv')
TOTAL_CSV = os.path.expanduser('~/Desktop/Balgrist Dataset/derivatives/labels/cord_length.csv')


def main():
    parser = argparse.ArgumentParser(description='Plot the correlation between the total (PMJ-conus) cord length and '
                                                 'the sacral (S1-conus) length for the Balgrist cohort.')
    parser.add_argument('-sacral', default=SACRAL_CSV, help=f'Sacral length CSV. Default: {SACRAL_CSV}')
    parser.add_argument('-total', default=TOTAL_CSV, help=f'Total (PMJ-conus) cord length CSV. Default: {TOTAL_CSV}')
    parser.add_argument('-panel-letter', default='none',
                        help='Panel letter in the upper-left corner (e.g. "D"); "none" disables it. Default: none')
    args = parser.parse_args()
    panel_letter = None if args.panel_letter.strip().lower() in ('', 'none') else args.panel_letter

    sacral = pd.read_csv(os.path.expanduser(args.sacral))
    total = pd.read_csv(os.path.expanduser(args.total)).rename(columns={'participant_id': 'subject'})

    df = sacral.merge(total[['subject', 'cord_length_mm']], on='subject', how='inner')
    df['sex'] = df['sex'].astype(str).str.upper()
    df['dataset'] = DATASET
    df['sacral_length_cm'] = df['length_S1-conus_centerline_mm'] / 10
    df['total_length_cm'] = df['cord_length_mm'] / 10
    df = df.dropna(subset=['sacral_length_cm', 'total_length_cm', 'sex'])

    rho, p = stats.spearmanr(df['sacral_length_cm'], df['total_length_cm'])
    rho_lo, rho_hi = spearman_ci(rho, len(df))
    print(f'{DATASET}: n = {len(df)}, total vs sacral Spearman rho = {rho:.2f} '
          f'[95% CI {rho_lo:.2f}, {rho_hi:.2f}], p = {p:.3f}')

    out_dir = os.path.dirname(os.path.abspath(args.sacral))
    plot_correlation(df, 'sacral_length_cm', 'total_length_cm',
                     'Sacral cord length S1 to conus tip [cm]', 'Total cord length PMJ to conus tip [cm]',
                     os.path.join(out_dir, 'figure_correlation_total_vs_sacral_length.png'),
                     panel_letter=panel_letter)


if __name__ == '__main__':
    main()
