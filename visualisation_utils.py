import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.markers import MarkerStyle
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
from scipy import stats

SPINAL_REGIONAL_LENGTH_FIGSIZE = (12, 6)
DATASET_MARKERS = {
    'spine-generic': 'o',       # Dataset 1: circle
    'whole-spine': 's',                 # Dataset 2: square
    'Balgrist': (6, 2, 0),           # Dataset 3: asterisk
    'EPFL': '^',                 # Dataset 4: triangle
    'Fudan': 'D',                 # Dataset 5: diamond
    'KCL': 'v',                   # Dataset 6: inverted triangle
    'NeuroRestore': 'p',          # Dataset 7: pentagon
}

FONT_SIZE = 14

# Sex: order on the x-axis, labels and colors
SEX_ORDER = ['F', 'M', 'unknown']
SEX_LABEL = {'F': 'Female', 'M': 'Male', 'unknown': 'Sex unknown'}
SEX_COLOR = {'F': '#d62728', 'M': '#2a78d6', 'unknown': 'gray'}

# Other names of the same dataset used by some scripts (e.g. the folder name used by default as the dataset name)
# -> name in DATASET_MARKERS, e.g. {'data-multi-subject': 'EPFL'}
DATASET_ALIASES = {'Balgrist localizer': 'Balgrist'}

# Marker sizes (scatter `s`); line-only markers (asterisk) are drawn larger and thicker to match filled markers
MARKER_SIZE = 20
UNFILLED_MARKER_SIZE_FACTOR = 1.6
UNFILLED_LINEWIDTH = 1.2
LEGEND_MARKER_SIZE = 7
# Color of the dataset markers in the legend (dataset is shown by marker only)
LEGEND_DATASET_COLOR = 'dimgray'

CORRELATION_FIGSIZE = (6, 5)
CORRELATION_LEGEND_NCOL = 3
FIT_COLOR = 'black'
FIT_LINEWIDTH = 2
FIT_BAND_COLOR = 'gray'
FIT_BAND_ALPHA = 0.2
FIT_LEGEND_LABEL = 'Linear fit, 95% CI'
# Extra space above the data (fraction of the y-range) for the correlation text in the upper left corner
CORRELATION_HEADROOM = 0.2


def canonical_dataset(name):
    """Name of the dataset in DATASET_MARKERS (resolves DATASET_ALIASES)."""
    return DATASET_ALIASES.get(name, name)


def get_dataset_markers(datasets):
    """
    Return the fixed marker of each given dataset (by name or alias), ordered as DATASET_MARKERS.
    """
    datasets = list(dict.fromkeys(datasets))  # unique, keep the given names
    unknown = sorted({d for d in datasets if canonical_dataset(d) not in DATASET_MARKERS})
    if unknown:
        raise ValueError(f'No marker defined for dataset(s) {unknown}. Add them to DATASET_MARKERS (or '
                         f'DATASET_ALIASES) in visualisation_utils.py (defined: {list(DATASET_MARKERS)}).')
    order = list(DATASET_MARKERS)
    datasets.sort(key=lambda d: order.index(canonical_dataset(d)))
    return {d: DATASET_MARKERS[canonical_dataset(d)] for d in datasets}

def scatter_points(ax, x, y, color, marker, white_edge=False, size=MARKER_SIZE, zorder=3):
    """
    Scatter points with the shared marker style. Line-only markers (asterisk) are drawn in `color` with a thicker
    line and a larger size; `white_edge` (white outline, e.g. for overlapping points) applies to filled markers only,
    since a white edge would make a line-only marker invisible.
    """
    if MarkerStyle(marker).is_filled():
        kwargs = dict(edgecolors='white', linewidths=0.5) if white_edge else {}
        return ax.scatter(x, y, color=color, marker=marker, s=size, zorder=zorder, **kwargs)
    return ax.scatter(x, y, color=color, marker=marker, s=size * UNFILLED_MARKER_SIZE_FACTOR,
                      linewidths=UNFILLED_LINEWIDTH, zorder=zorder)


def legend_handles(df, dataset_markers, sex_col='sex', dataset_col='dataset'):
    """
    Legend handles: sex (colored rectangle) followed by dataset (marker, gray), each with the number of subjects.
    Sex is shown as a rectangle (not a marker) so it cannot be confused with a dataset marker (e.g. the square).
    Only the sexes and datasets present in `df` are listed.
    """
    handles = [Patch(facecolor=SEX_COLOR[sex], edgecolor='none',
                     label=f"{SEX_LABEL[sex]} (n = {(df[sex_col] == sex).sum()})")
               for sex in SEX_ORDER if (df[sex_col] == sex).any()]
    for dataset, marker in dataset_markers.items():
        filled = MarkerStyle(marker).is_filled()
        handles.append(Line2D([], [], color=LEGEND_DATASET_COLOR, marker=marker, linestyle='',
                              markersize=LEGEND_MARKER_SIZE * (1 if filled else 1.25),
                              markeredgewidth=1 if filled else UNFILLED_LINEWIDTH,
                              label=f"{dataset} (n = {(df[dataset_col] == dataset).sum()})"))
    return handles


def add_legend_above(ax, handles, ncol=None):
    """Legend above the plot (upper right), without frame; one row unless `ncol` is given."""
    ax.legend(handles=handles, loc='lower right', bbox_to_anchor=(1.0, 1.0), ncol=ncol or len(handles),
              frameon=False, fontsize=FONT_SIZE - 3, handlelength=1.2, handletextpad=0.3, columnspacing=1.2,
              borderaxespad=0.2)


def format_p(p):
    """'p < 0.001' or 'p = 0.xxx'."""
    return 'p < 0.001' if p < 0.001 else f'p = {p:.3f}'


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
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    n = len(x)
    slope, intercept = np.polyfit(x, y, 1)
    y_fit = slope * x_fit + intercept
    residuals = y - (slope * x + intercept)
    s = np.sqrt(np.sum(residuals ** 2) / (n - 2))   # residual standard error
    se_fit = s * np.sqrt(1 / n + (x_fit - x.mean()) ** 2 / np.sum((x - x.mean()) ** 2))
    t_crit = stats.t.ppf(1 - alpha / 2, n - 2)
    return y_fit, y_fit - t_crit * se_fit, y_fit + t_crit * se_fit


def plot_correlation(df, x_col, y_col, xlabel, ylabel, fname_figure, sex_col='sex', dataset_col='dataset',
                     integer_xticks=False):
    """
    Shared correlation figure: scatter plot (color = sex, marker = dataset, white edge on filled markers), linear fit
    across all subjects (visual guide) with its 95% confidence band, the Spearman correlation (all subjects pooled)
    in the upper left corner and the shared legend above the plot (sex, datasets, linear fit).
    :param df: dataframe with one row per subject (no missing values in x_col, y_col, sex_col)
    :param x_col: column on the x-axis (already in the plotted unit, e.g. cm)
    :param y_col: column on the y-axis (already in the plotted unit, e.g. cm)
    :param xlabel: x-axis label
    :param ylabel: y-axis label
    :param fname_figure: output PNG path
    :param sex_col: column with 'F' / 'M'
    :param dataset_col: column with the dataset name (key of DATASET_MARKERS or DATASET_ALIASES)
    :param integer_xticks: integer x ticks (e.g. for height in cm)
    :return: fig, ax
    """
    x = df[x_col].astype(float).to_numpy()
    y = df[y_col].astype(float).to_numpy()
    rho, p = stats.spearmanr(x, y)
    rho_lo, rho_hi = spearman_ci(rho, len(df))
    dataset_markers = get_dataset_markers(df[dataset_col].unique())

    fig, ax = plt.subplots(figsize=CORRELATION_FIGSIZE)
    for dataset, marker in dataset_markers.items():
        for sex in SEX_ORDER:
            mask = ((df[dataset_col] == dataset) & (df[sex_col] == sex)).to_numpy()
            if mask.any():
                # White edge so that overlapping points remain distinguishable (filled markers only)
                scatter_points(ax, x[mask], y[mask], color=SEX_COLOR[sex], marker=marker, white_edge=True)

    # Linear fit across all subjects (only as a visual guide) with its 95% confidence band
    x_fit = np.linspace(x.min(), x.max(), 100)
    y_fit, y_lo, y_hi = linear_fit_ci_band(x, y, x_fit)
    ax.fill_between(x_fit, y_lo, y_hi, color=FIT_BAND_COLOR, alpha=FIT_BAND_ALPHA, linewidth=0, zorder=1)
    ax.plot(x_fit, y_fit, color=FIT_COLOR, linewidth=FIT_LINEWIDTH, zorder=2)

    # Spearman correlation across all subjects in the upper left corner
    ax.text(0.03, 0.97, f'n = {len(df)}\nSpearman ρ = {rho:.2f} [95% CI {rho_lo:.2f}, {rho_hi:.2f}], {format_p(p)}',
            transform=ax.transAxes, fontsize=FONT_SIZE - 4, va='top')

    ax.set_xlabel(xlabel, fontsize=FONT_SIZE)
    ax.set_ylabel(ylabel, fontsize=FONT_SIZE)
    if integer_xticks:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, steps=[1, 2, 5, 10]))
    ax.tick_params(axis='both', labelsize=FONT_SIZE - 2)
    ax.grid(alpha=0.2)
    ax.set_axisbelow(True)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    # Headroom above the data for the correlation text
    y_min, y_max = ax.get_ylim()
    ax.set_ylim(y_min, y_max + CORRELATION_HEADROOM * (y_max - y_min))

    # Shared legend above the plot: sex (rectangles), datasets (markers), linear fit with its CI
    handles = legend_handles(df, dataset_markers, sex_col=sex_col, dataset_col=dataset_col)
    handles.append(Line2D([], [], color=FIT_COLOR, linewidth=FIT_LINEWIDTH, label=FIT_LEGEND_LABEL))
    add_legend_above(ax, handles, ncol=CORRELATION_LEGEND_NCOL)

    plt.tight_layout()
    fig.savefig(fname_figure, dpi=300)
    print(f'Figure saved to {fname_figure}')
    return fig, ax