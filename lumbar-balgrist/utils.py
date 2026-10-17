"""
Metric definitions, statistics and plotting shared by the lumbar Balgrist scripts (lumbar_length_count_slices.py,
lumbar_length_along_centerline.py and correlation_length_height.py).

Author: Jan Valosek
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

LEVELS = {1: 'L1', 2: 'L2', 3: 'L3', 4: 'L4', 5: 'L5', 6: 'S1', 7: 'S2'}

# Metrics: first and last spinal level (values in the segmentation), and names for the outputs and figures
METRICS = {
    'top-bottom': {'first': 1, 'last': 5, 'name': 'length_L1-L5', 'label': 'Length top L1 – bottom L5'},
    'midpoints': {'first': 1, 'last': 6, 'name': 'midpoints_distance_L1-S1', 'label': 'Distance L1–S1 midpoints'},
}

FONT_SIZE = 14

# Data: labeled spinal cord segmentations (sub-*/*_spinallevels_dseg.nii.gz) and the Excel file with the subject sex
# and height ('sex' and 'Height (cm)' columns in the 'Spinal level heights (mm)' sheet)
DATA_DIR = os.path.expanduser('~/data/spinal-levels/lumbar_rootlets_Balgrist')
INPUT_DIR = os.path.join(DATA_DIR, 'bids')
PARTICIPANTS_XLSX = os.path.join(INPUT_DIR, 'spinal_level_heights_top_L1-bottom_S2_with_chart.xlsx')
# Output folder for the CSV files and figures
OUTPUT_DIR = os.path.join(DATA_DIR, 'results_and_figures')
PARTICIPANTS_SHEET = 'Spinal level heights (mm)'

# Female and male colors and markers
SEX_STYLE = {'F': {'color': '#2a78d6', 'marker': 'o', 'label': 'Female'},
             'M': {'color': '#eb6834', 'marker': 's', 'label': 'Male'}}


def add_sex_height(df, fname_xlsx):
    """
    Add the subject sex ('F'/'M') and height (cm) from the Excel file to the dataframe.
    :param df: dataframe with a 'subject' column
    :param fname_xlsx: Excel file with the 'sex' and 'Height (cm)' columns in the PARTICIPANTS_SHEET sheet
    :return: dataframe with 'sex' and 'height' columns (NaN for subjects not in the Excel file)
    """
    df_xlsx = pd.read_excel(fname_xlsx, sheet_name=PARTICIPANTS_SHEET)
    df_xlsx = df_xlsx[df_xlsx['Participants'].astype(str).str.startswith('sub-')]
    df_xlsx['sex'] = df_xlsx['sex'].str.upper()   # 'f'/'m' -> 'F'/'M'
    df_xlsx = df_xlsx.rename(columns={'Height (cm)': 'height'})
    return df.merge(df_xlsx[['Participants', 'sex', 'height']], how='left', left_on='subject',
                    right_on='Participants').drop(columns='Participants')


def spearman_ci(rho, n, alpha=0.05):
    """
    Confidence interval of the Spearman correlation coefficient using the Fisher z-transformation with the
    Bonett & Wright (2000) standard error, sqrt(1.06 / (n - 3)).
    """
    z = np.arctanh(rho)
    se = np.sqrt(1.06 / (n - 3))
    z_crit = stats.norm.ppf(1 - alpha / 2)
    return np.tanh(z - z_crit * se), np.tanh(z + z_crit * se)


def linear_fit_ci_band(x, y, x_fit, alpha=0.05):
    """
    Linear fit and the confidence band of the fitted mean (ordinary least squares).
    :return: y_fit, lower bound, upper bound evaluated at x_fit
    """
    n = len(x)
    slope, intercept = np.polyfit(x, y, 1)
    y_fit = slope * x_fit + intercept
    residuals = y - (slope * x + intercept)
    s = np.sqrt(np.sum(residuals ** 2) / (n - 2))   # residual standard error
    se_fit = s * np.sqrt(1 / n + (x_fit - x.mean()) ** 2 / np.sum((x - x.mean()) ** 2))
    t_crit = stats.t.ppf(1 - alpha / 2, n - 2)
    return y_fit, y_fit - t_crit * se_fit, y_fit + t_crit * se_fit


def generate_figure(df, dir_out, metric):
    """
    Generate a figure with subjects on the x-axis (sorted by height) and the length on the y-axis.
    :param df: dataframe with 'subject' and '<metric name>_mm' columns
    :param dir_out: output folder where the figure will be saved
    :param metric: dict from METRICS
    The dataframe must also contain a 'sex' column ('F'/'M') used to color the points and a 'height' column (cm) used
    to sort the subjects.
    """
    # Sort the subjects by height (from the shortest to the tallest); subjects without height at the end
    df_plot = df.sort_values('height', na_position='last').reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(max(6, len(df_plot) * 0.6), 5))
    # Points colored by sex; subjects without sex information in gray
    x = np.arange(1, len(df_plot) + 1)
    y = (df_plot[f"{metric['name']}_mm"]).to_numpy() / 10  # convert mm to cm for plotting
    for sex, style in SEX_STYLE.items():
        mask = (df_plot['sex'] == sex).to_numpy()
        ax.scatter(x[mask], y[mask], color=style['color'], marker=style['marker'], s=50, zorder=3,
                   label=f"{style['label']} (n = {mask.sum()})")
    mask = (~df_plot['sex'].isin(list(SEX_STYLE))).to_numpy()
    if mask.any():
        ax.scatter(x[mask], y[mask], color='gray', marker='x', s=50, zorder=3,
                   label=f'Sex unknown (n = {mask.sum()})')

    # Subject names as x-axis labels
    ax.set_xticks(x)
    ax.set_xticklabels(df_plot['subject'], rotation=45, ha='right')
    ax.set_xlim(0.5, len(df_plot) + 0.5)
    ax.set_xlabel('Subjects sorted by height', fontsize=FONT_SIZE)
    ax.set_ylim(0, 12)

    ax.set_ylabel(f"{metric['label']} [cm]", fontsize=FONT_SIZE)
    ax.tick_params(axis='y', labelsize=FONT_SIZE - 2)
    ax.tick_params(axis='x', labelsize=FONT_SIZE - 2)
    ax.grid(axis='y', alpha=0.2)
    ax.set_axisbelow(True)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    # Legend above the plot so it does not overlap the points
    ax.legend(loc='lower right', bbox_to_anchor=(1, 1), ncol=3, frameon=False, fontsize=FONT_SIZE - 4,
              handletextpad=0.3, columnspacing=1)

    plt.tight_layout()

    fname_figure = os.path.join(dir_out, f"figure_lumbar_{metric['name']}.png")
    fig.savefig(fname_figure, dpi=300)
    print(f'Figure saved to {fname_figure}')
    plt.show()


def correlation_scatter(df, x_col, y_col, x_label, y_label, title, fname_out):
    """
    Scatter plot of y_col vs x_col, colored by sex, with a least-squares linear fit (visual guide) and its
    95% confidence band. The Spearman correlation (rho, 95% CI, p and n) is printed above the plot and the sex
    legend below the plot, both outside the axes. Used by correlation_cord_length.py.
    :param df: dataframe with x_col, y_col and a 'sex' column ('F'/'M')
    :return: dict with the Spearman rho, p, 95% CI and n
    """
    d = df.dropna(subset=[x_col, y_col]).copy()
    x = d[x_col].to_numpy(float)
    y = d[y_col].to_numpy(float)
    n = len(d)

    rho, p = stats.spearmanr(x, y)
    rho_lo, rho_hi = spearman_ci(rho, n)

    fig, ax = plt.subplots(figsize=(7, 6))
    for sex, style in SEX_STYLE.items():
        mask = (d['sex'] == sex).to_numpy()
        ax.scatter(x[mask], y[mask], color=style['color'], marker=style['marker'], s=60, zorder=3,
                   label=f"{style['label']} (n = {mask.sum()})")

    x_fit = np.linspace(x.min(), x.max(), 100)
    y_fit, y_lo, y_hi = linear_fit_ci_band(x, y, x_fit)
    ax.fill_between(x_fit, y_lo, y_hi, color='gray', alpha=0.2, linewidth=0, zorder=1)
    ax.plot(x_fit, y_fit, color='black', linestyle='--', linewidth=2, zorder=2)

    ax.set_xlabel(x_label, fontsize=FONT_SIZE)
    ax.set_ylabel(y_label, fontsize=FONT_SIZE)
    ax.tick_params(axis='both', labelsize=FONT_SIZE - 2)
    ax.grid(alpha=0.2)
    ax.set_axisbelow(True)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    ax.set_title(title, fontsize=FONT_SIZE + 1, pad=34)
    ax.text(0.5, 1.015, f'Spearman ρ = {rho:.2f} [95% CI {rho_lo:.2f}, {rho_hi:.2f}], p = {p:.3f}    n = {n}',
            transform=ax.transAxes, ha='center', va='bottom', fontsize=FONT_SIZE - 2)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False, fontsize=FONT_SIZE - 2)

    fig.subplots_adjust(top=0.86, bottom=0.17)
    fig.savefig(fname_out, dpi=300, bbox_inches='tight')
    print(f'Figure saved to {fname_out}')
    plt.close(fig)
    return {'rho': rho, 'p': p, 'ci': (rho_lo, rho_hi), 'n': n}
