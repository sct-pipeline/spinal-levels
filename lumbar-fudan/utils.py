"""
Statistics and plotting shared by the lumbar Fudan scripts (lumbar_fudan_L1-L5_length.py and
correlation_length_height.py).

The style (SEX_STYLE colors/markers), the Spearman confidence interval (spearman_ci) and the linear-fit
confidence band (linear_fit_ci_band) match ../lumbar-balgrist/utils.py so that the figures are consistent
across the lumbar cohorts (Balgrist, EPFL, Fudan).

Author: Elia Rochiccioli
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

FONT_SIZE = 14

# Female and male colors and markers (same as ../lumbar-balgrist/utils.py)
SEX_STYLE = {'F': {'color': '#2a78d6', 'marker': 'o', 'label': 'Female'},
             'M': {'color': '#eb6834', 'marker': 's', 'label': 'Male'}}


def spearman_ci(rho, n, alpha=0.05):
    """
    Confidence interval of the Spearman correlation coefficient using the Fisher z-transformation with the
    Bonett & Wright (2000) standard error, sqrt(1.06 / (n - 3)). Same as ../lumbar-balgrist/utils.py.
    """
    z = np.arctanh(rho)
    se = np.sqrt(1.06 / (n - 3))
    z_crit = stats.norm.ppf(1 - alpha / 2)
    return np.tanh(z - z_crit * se), np.tanh(z + z_crit * se)


def linear_fit_ci_band(x, y, x_fit, alpha=0.05):
    """
    Linear fit and the confidence band of the fitted mean (ordinary least squares).
    Same as ../lumbar-balgrist/utils.py.
    :return: y_fit, lower bound, upper bound evaluated at x_fit
    """
    n = len(x)
    slope, intercept = np.polyfit(x, y, 1)
    y_fit = slope * x_fit + intercept
    residuals = y - (slope * x + intercept)
    s = np.sqrt(np.sum(residuals ** 2) / (n - 2))
    se_fit = s * np.sqrt(1 / n + (x_fit - x.mean()) ** 2 / np.sum((x - x.mean()) ** 2))
    t_crit = stats.t.ppf(1 - alpha / 2, n - 2)
    return y_fit, y_fit - t_crit * se_fit, y_fit + t_crit * se_fit


def correlation_scatter(df, x_col, y_col, x_label, y_label, title, fname_out):
    """
    Scatter plot of y_col vs x_col, colored by sex, with a least-squares linear fit (visual guide) and its
    95% confidence band. The Spearman correlation (rho, 95% CI, p and n) is printed above the plot and the
    sex legend below the plot, both outside the axes.
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
