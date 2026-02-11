import matplotlib.pyplot as plt
import numpy as np
from decimal import Decimal, ROUND_UP
from scipy.ndimage import gaussian_filter1d

from matplotlib.patches import FancyArrowPatch
from matplotlib.patheffects import Stroke, Normal

# plt.rcParams.update({
#     # 'text.usetex': True,
#     'font.family': 'serif',
#     'font.serif': ['STIXGeneral', 'Nimbus Roman', 'Nimbus Roman No9 L', 'Times', 'DejaVu Serif'],
#     'mathtext.fontset': 'stix',  # matches Times aesthetics for math
# })

def annotate_common_factor_to_peaks(
    ax, xs, ys, peak_indices, factor, colors=None,
    anchor=None, anchor_offset=(0.0, 0.07),
    arrowstyle='->', mutation_scale=12, linewidth=1.4,
    curve=0.2, text_kwargs=None, arrow_kwargs=None,
):
    """
    Draw a single label '×1/factor' and connect it with proper arrows to multiple peaks.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    x : 1D array
        Common x for all curves.
    ys : list of 1D arrays
        List of y arrays (one per line).
    peak_indices : list of int
        Indices of the peaks in each y that you want to annotate.
    factor : float
        Common reduction factor (label text is '×1/factor').
    colors : list or None
        List of colors for arrows (usually match the line colors). If None, uses 'black'.
    anchor : tuple (x_text, y_text) or None
        If given, position for the single text label.
        If None, it will be auto-placed above the average y of the peaks.
    anchor_offset : (dx_frac, dy_frac)
        Fraction of x- and y-range to offset the anchor (applied after auto-placement).
        Ignored if `anchor` is provided.
    arrowstyle, mutation_scale, linewidth : arrow appearance.
    curve : float
        Positive values produce a slight curve to reduce overlaps (via connectionstyle).
    text_kwargs : dict
        Extra kwargs for the text (fontsize, weight, color, etc.).
    arrow_kwargs : dict
        Extra kwargs for arrows (e.g., alpha).
    """
    if text_kwargs is None:
        text_kwargs = {}
    if arrow_kwargs is None:
        arrow_kwargs = {}

    # Prepare colors
    if colors is None:
        colors = ["black"] * len(ys)
    else:
        # If a single color passed, broadcast
        if not hasattr(colors, "__len__") or isinstance(colors, str):
            colors = [colors] * len(ys)

    # Gather peak coordinates
    pts = []
    for x, y, idx in zip(xs, ys, peak_indices):
        if idx is None:
            continue
        pts.append((x[idx], y[idx]))
    if len(pts) == 0:
        return  # nothing to annotate

    # Determine anchor (text) position
    if anchor is None:
        xv = np.array([p[0] for p in pts])
        yv = np.array([p[1] for p in pts])

        # Place above the centroid of peaks, slightly offset
        x_mid = float(np.mean(xv))
        y_mid = float(np.mean(yv))
        xr = ax.get_xlim()
        yr = ax.get_ylim()
        dx = anchor_offset[0] * (xr[1] - xr[0])
        dy = anchor_offset[1] * (yr[1] - yr[0])
        x_text, y_text = x_mid + dx, y_mid + dy
    else:
        x_text, y_text = anchor

    # Draw the single label
    txt = f"×{factor:g}"
    # Optional white halo for readability
    default_text_effects = [Stroke(linewidth=3.0, foreground="white"), Normal()]
    text = ax.text(
        x_text, y_text, txt,
        ha="center", va="bottom",
        zorder=6,
        path_effects=text_kwargs.pop("path_effects", default_text_effects),
        **text_kwargs
    )

    # Draw arrows from the text to each peak
    for (xt, yt), color in zip(pts, colors):
        arrow = FancyArrowPatch(
            posA=(x_text, y_text),
            posB=(xt, yt),
            arrowstyle=arrowstyle,
            mutation_scale=mutation_scale,
            lw=linewidth,
            color=color,
            shrinkA=0.0, shrinkB=0.0,
            connectionstyle=f"arc3,rad={curve}",
            zorder=5,
            **arrow_kwargs
        )
        ax.add_patch(arrow)

    # Keep label inside axes if possible
    ax.figure.canvas.draw_idle()

def reduce_in_range(x, y, x_min, x_max, factor):
    """
    Apply amplitude reduction only within [x_min, x_max].
    Returns a SINGLE combined y array (continuous line).
    """
    y_mod = y.copy()
    mask = (x >= x_min) & (x <= x_max)
    y_mod[mask] = y[mask] * factor
    return y_mod, mask


def annotate_peak(ax, x, y, peak_idx, factor, text_offset=0.05):
    """
    Place a text label + arrow pointing to the peak at index peak_idx.
    """
    xv = x[peak_idx]
    yv = y[peak_idx]

    ax.annotate(
        f"×{factor}",
        xy=(xv, yv),
        xytext=(xv, yv + text_offset * (ax.get_ylim()[1] - ax.get_ylim()[0])),
        ha="center",
        arrowprops=dict(arrowstyle="->", color="black", lw=1.2),
        fontsize=10,
    )


def first_peak_index_in_range_numpy(x, y, x_min, x_max):
    """
    Return the index of the FIRST local maximum of y whose x is in [x_min, x_max].
    Local maximum condition: y[i-1] < y[i] > y[i+1]
    Returns None if no such peak exists.
    """
    x = np.asarray(x)
    y = np.asarray(y)

    mask = (x >= x_min) & (x <= x_max)
    idx_region = np.where(mask)[0]
    if idx_region.size < 3:
        return None  # not enough points to define a local max

    # We cannot test the very first or last index of the region for a 3-point local max
    start = idx_region[0]
    end   = idx_region[-1]

    # Candidate indices (exclude endpoints)
    i = np.arange(start+1, end)

    y_prev = y[i-1]
    y_curr = y[i]
    y_next = y[i+1]

    # Handle NaNs: they cannot form a valid peak
    valid = ~np.isnan(y_prev) & ~np.isnan(y_curr) & ~np.isnan(y_next)

    # Strict local-maximum condition
    is_peak = (y_prev < y_curr) & (y_curr > y_next)

    cand = i[valid & is_peak]
    if cand.size == 0:
        return None
    return int(cand[0])




data_dir = "data_highres/"
figure_dir = "figures/"
ldos_file = "ldos_particle_atImp_U_"
energies_file = "ldos_particle_atImp_energies_U_"

plotsize = 2.5

U_vals = [0.0, 0.75, -0.75 ]

significant_digits = 6

delta = 0.3

ylim = [-0.01 ,0.69]
xlim = [-1.3, 1.3]

U_vals_plot = []

energies_plot = []
nr_eigvals = 0



fig, ax = plt.subplots(1,1, figsize=[1.618*plotsize,plotsize], sharex=True, sharey=False) # applyPlotSettings(num_plots,1)
fig.patch.set_facecolor('gray')
fig.patch.set_alpha(0.0)

energy_list = []
smooth_dos_list = []
peak_idxs = []
factor = 0.05
for U in U_vals:
    energies = np.loadtxt(data_dir + energies_file + "{0:.6f}.csv".format(round(U,significant_digits)))
    energies = np.array(energies)/delta
    dos = np.loadtxt(data_dir + ldos_file + "{0:.6f}.csv".format(round(U,significant_digits)))
    label = r'$\varepsilon_i = {0:.2f}$'.format(U)
    smoothed_dos = gaussian_filter1d(dos, sigma=1.5)


    x_max = 0.9
    x_min = -x_max
    # Create the *combined* y array
    smoothed_dos_reduced, mask = reduce_in_range(energies, smoothed_dos, x_min, x_max, factor)
    peak_idx = first_peak_index_in_range_numpy(energies, smoothed_dos, x_min, x_max)
    print(energies[peak_idx])
    ax.plot(energies, smoothed_dos_reduced, label=label)

    # Annotate highest peak in reduced interval
    if peak_idx is not None:
        # annotate_peak(ax, energies, smoothed_dos_reduced, peak_idx, factor)
        peak_idxs.append(peak_idx)
        smooth_dos_list.append(smoothed_dos_reduced)
        energy_list.append(energies)
print(peak_idxs)

annotate_common_factor_to_peaks(
    ax, energy_list, smooth_dos_list, peak_idxs, factor, colors=None,
    anchor=None, anchor_offset=(-0.075, 0.035),
    arrowstyle='->', mutation_scale=12, linewidth=1.0,
    curve=0.0, text_kwargs=None, arrow_kwargs=None,
)



ax.set_xlim(xlim)
ax.set_ylim(ylim)

ax.set_xlabel(r'$E\,/\, \Delta$')
ax.set_ylabel(r'LDOS')
linewidth=0.5
# ax.axhline(y=0, color='k',linewidth=linewidth)
ax.grid(True, alpha=0.3)

# ax.legend()
leg = ax.legend(markerscale=0.7, handlelength=0.9, fontsize=8, loc="upper right")
fig.tight_layout()
plt.savefig(figure_dir + "ldos.pdf", bbox_inches="tight")   # vector
plt.savefig(figure_dir + "ldos.pgf", bbox_inches="tight")   # vector
plt.savefig(figure_dir + "ldos.png", dpi=300, bbox_inches="tight")  # raster

import matplot2tikz
matplot2tikz.Flavors.latex.preamble()
# matplot2tikz.clean_figure()
matplot2tikz.save(figure_dir + "ldos.tex")
# matplot2tikz.clean_figure(figure=fig_bottom)
# matplot2tikz.save("figures/Thesis/anomalous_greens_compare_bottom.tex", figure=fig_bottom)