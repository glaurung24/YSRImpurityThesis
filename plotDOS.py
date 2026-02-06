import matplotlib.pyplot as plt
import numpy as np
from decimal import Decimal, ROUND_UP
from scipy.ndimage import gaussian_filter1d



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

U_vals = [0.0, 0.5, -0.5 ]

significant_digits = 6

delta = 0.3

ylim = [0,4.5]
xlim = [-1.3, 1.3]

U_vals_plot = []

energies_plot = []
nr_eigvals = 0



fig, ax = plt.subplots(1,1, figsize=[1.618*3,3], sharex=True, sharey=False) # applyPlotSettings(num_plots,1)
fig.patch.set_facecolor('gray')
fig.patch.set_alpha(0.0)

for U in U_vals:
    energies = np.loadtxt(data_dir + energies_file + "{0:.6f}.csv".format(round(U,significant_digits)))
    energies = np.array(energies)/delta
    dos = np.loadtxt(data_dir + ldos_file + "{0:.6f}.csv".format(round(U,significant_digits)))
    label = r'$U = {0:.1f}$'.format(U)
    smoothed_dos = gaussian_filter1d(dos, sigma=1)

    factor = 1e-2
    x_max = 0.9
    x_min = -x_max
    # Create the *combined* y array
    smoothed_dos_reduced, _ = reduce_in_range(energies, smoothed_dos, x_min, x_max, factor)
    peak_idx = first_peak_index_in_range_numpy(energies, smoothed_dos, x_min, x_max)

    ax.plot(energies, smoothed_dos_reduced, label=label)

    # Annotate highest peak in reduced interval
    if peak_idx is not None:
        annotate_peak(ax, energies, smoothed_dos_reduced, peak_idx, factor)



ax.set_xlim(xlim)
# ax.set_ylim(ylim)

ax.set_xlabel(r'$E\,/\, \Delta$')
ax.set_ylabel(r'LDOS')
linewidth=0.5
# ax.axhline(y=0, color='k',linewidth=linewidth)
ax.grid(True, alpha=0.3)

ax.legend()
fig.tight_layout()
fig.savefig(figure_dir + "ldos.png")
import matplot2tikz
matplot2tikz.Flavors.latex.preamble()
# matplot2tikz.clean_figure()
matplot2tikz.save(figure_dir + "ldos.tex")
# matplot2tikz.clean_figure(figure=fig_bottom)
# matplot2tikz.save("figures/Thesis/anomalous_greens_compare_bottom.tex", figure=fig_bottom)