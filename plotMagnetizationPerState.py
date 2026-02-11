import matplotlib.pyplot as plt
import numpy as np
from decimal import Decimal, ROUND_UP


# plt.rcParams.update({
#     'text.usetex': True,
#     'font.family': 'serif',
#     'font.serif': ['Times New Roman'],
# })


plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['STIXGeneral', 'Nimbus Roman', 'Nimbus Roman No9 L', 'Times', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',  # matches Times aesthetics for math
})




def findYSRIndex(eigenvalues):
    
    energy_low = -4.
    energy_high = 4.
    idx_low = -1
    idx_high = -1
    for idx, ev in enumerate(eigenvalues):
        if(energy_high > ev and ev > 0):
            energy_high = ev
            idx_high = idx
        if(energy_low < ev and ev < 0):
            energy_low = ev
            idx_low = idx
    return idx_low, idx_high

def zero_crossings(x, y, direction='any', return_indices=False):
    """
    Find zero-crossing x-locations by linearly interpolating between samples.

    Parameters
    ----------
    x : (N,) array_like
        X-coordinates of the sampled line (not necessarily sorted).
    y : (N,) array_like
        Y-values at the corresponding x's.
    direction : {'any', 'rising', 'falling'}, optional
        - 'any'     : all crossings
        - 'rising'  : crossings where y goes from negative to positive
        - 'falling' : crossings where y goes from positive to negative
    return_indices : bool, optional
        If True, also return the index `i` of the left point of the segment
        (i.e., crossing occurs between points i and i+1 in the **sorted** arrays).

    Returns
    -------
    x0 : (M,) ndarray
        Interpolated x-positions where the line crosses y=0.
    idx : (M,) ndarray, optional
        Indices of left segment endpoints in the sorted arrays (only if return_indices=True).

    Notes
    -----
    - Linear interpolation is used within each segment [i, i+1]:
        y = y_i + t * (y_{i+1} - y_i),  solve for y=0 => t = -y_i / (y_{i+1} - y_i),
        then x0 = x_i + t * (x_{i+1} - x_i)
    - Segments containing NaNs are ignored.
    - If y[i] == 0 exactly, that x is included once; if a flat zero span exists,
      its endpoints will be included (you can `np.unique` afterward if needed).
    """
    x = np.asarray(x)
    y = np.asarray(y)
    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape.")
    if x.ndim != 1:
        raise ValueError("x and y must be 1-D arrays.")

    # Sort by x to avoid weirdness with unsorted input
    order = np.argsort(x)
    x = x[order]
    y = y[order]

    # Remove segments with NaNs
    isnan = np.isnan(x) | np.isnan(y)
    if np.any(isnan):
        # We'll mask out any segment where either endpoint is NaN
        valid = ~(isnan[:-1] | isnan[1:])
    else:
        valid = np.ones(len(x) - 1, dtype=bool)

    x0_list = []
    idx_list = []

    xi, xj = x[:-1], x[1:]
    yi, yj = y[:-1], y[1:]

    # Consider three cases:
    # 1) Proper sign change: yi * yj < 0
    # 2) Exact zero at endpoint: yi == 0 or yj == 0
    # 3) Flat segment at zero: yi == 0 and yj == 0 (include both ends once)

    # Case 1: Proper sign change
    sign_change = (yi * yj < 0) & valid
    if np.any(sign_change):
        i = np.where(sign_change)[0]
        # Filter by direction
        if direction == 'rising':
            i = i[yi[i] < 0]  # - to +
        elif direction == 'falling':
            i = i[yi[i] > 0]  # + to -
        # Linear interpolation parameter t in [0,1]
        t = -yi[i] / (yj[i] - yi[i])
        x0 = xi[i] + t * (xj[i] - xi[i])
        x0_list.append(x0)
        idx_list.append(i)

    # Case 2: Exact zeros at endpoints (count once)
    # We'll include xi where yi==0 and the neighboring point is nonzero OR
    # include xj where yj==0 and the neighboring point is nonzero.
    # This avoids double counting with sign-change case.
    exact_left = (yi == 0) & (yj != 0) & valid
    exact_right = (yj == 0) & (yi != 0) & valid

    def filter_direction_for_exact(i, left=True):
        if direction == 'any':
            return i
        if left:
            # Crossing depends on neighbor sign relative to zero
            # If yi==0 and yj>0, then it's rising only if previous y (if exists) < 0.
            # That's ambiguous without a previous point. As a simple rule,
            # we infer direction from the nonzero neighbor only:
            # yi==0, yj>0 => rising; yi==0, yj<0 => falling.
            if direction == 'rising':
                return i[yj[i] > 0]
            else:  # 'falling'
                return i[yj[i] < 0]
        else:
            # yj==0; infer direction from yi
            if direction == 'rising':
                return i[yi[i] < 0]
            else:
                return i[yi[i] > 0]

    if np.any(exact_left):
        i = np.where(exact_left)[0]
        i = filter_direction_for_exact(i, left=True)
        if i.size:
            x0_list.append(xi[i].astype(float))
            idx_list.append(i)

    if np.any(exact_right):
        i = np.where(exact_right)[0]
        i = filter_direction_for_exact(i, left=False)
        if i.size:
            x0_list.append(xj[i].astype(float))
            idx_list.append(i)

    # Case 3: Flat zero segments (yi==0 & yj==0) — include the left endpoint once.
    flat_zero = (yi == 0) & (yj == 0) & valid
    if np.any(flat_zero):
        i = np.where(flat_zero)[0]
        # Direction is undefined for a flat zero; only return for 'any'
        if direction == 'any':
            x0_list.append(xi[i].astype(float))
            idx_list.append(i)

    if not x0_list:
        return (np.array([], dtype=float), np.array([], dtype=int)) if return_indices else np.array([], dtype=float)

    x0_all = np.concatenate(x0_list)
    idx_all = np.concatenate(idx_list) if return_indices else None

    # Sort crossings by x for neatness (and deduplicate near-equals)
    order2 = np.argsort(x0_all)
    x0_all = x0_all[order2]
    if return_indices:
        idx_all = idx_all[order2]

    # Deduplicate very close duplicates (e.g., exact zeros adjacent to sign changes)
    # Tolerance can be adjusted; here we use absolute tol relative to data scale.
    if x0_all.size > 1:
        tol = 1e-12 * max(1.0, np.nanmax(np.abs(x)))
        keep = np.ones_like(x0_all, dtype=bool)
        keep[1:] = np.abs(np.diff(x0_all)) > tol
        x0_all = x0_all[keep]
        if return_indices:
            idx_all = idx_all[keep]

    return (x0_all, idx_all) if return_indices else x0_all


def first_zero_crossing(x, y, direction='any'):
    """
    Convenience wrapper: return the first zero crossing by x (or None if none).
    """
    x0 = zero_crossings(x, y, direction=direction, return_indices=False)
    return x0[0] if x0.size else None



import numpy as np
import matplotlib.pyplot as plt

def mark_zero_crossings(ax, x, y, *,
                        color='k',
                        linestyle='--',
                        linewidth=1.2,
                        span='down',
                        show_points=True,
                        annotate_values=False,
                        # --- New tick/label controls ---
                        add_tick=True,
                        tick_height_frac=0.02,
                        tick_linewidth=1.5,
                        tick_color=None,
                        label_text=r'$J_C$',
                        label_offset_pts=10,
                        label_color=None,
                        label_fontsize=None):
    """
    Mark zero crossings of y(x) with vertical dashed lines, a tick at the x-axis, and a custom label.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to draw on.
    x, y : array_like
        Sampled data for the curve (same length).
    color : str
        Color for the vertical guides (and points if enabled).
    linestyle : str
        Line style for the vertical guides.
    linewidth : float
        Line width for the vertical guides.
    span : {'down', 'up', 'full', 'notch'}
        'down' : draw from y_min (or 0 if y_min>0) up to 0
        'up'   : draw from 0 up to y_max (or 0 if y_max<0)
        'full' : draw from y_min to y_max
        'notch': draw a short vertical mark centered at 0 (no tall guide)
    show_points : bool
        If True, put a dot at (x0, 0).
    annotate_values : bool
        If True, annotate each crossing with its numerical x0 value.
    add_tick : bool
        If True, draw a short tick crossing the axis at y=0.
    tick_height_frac : float
        Height of the tick as a fraction of current y-range (e.g., 0.02 = 2%).
        The tick is drawn from y=0 downwards by this amount.
    tick_linewidth : float
        Line width of the tick.
    tick_color : str or None
        Color of the tick; defaults to `color` if None.
    label_text : str
        The special string to place under the tick, e.g., r'$J_C$'.
        If empty or None, no label is drawn.
    label_offset_pts : float
        Vertical offset (in points) below the axis for the label text.
    label_color : str or None
        Color for the label text; defaults to `tick_color` or `color`.
    label_fontsize : float or None
        Font size for the label text.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape.")
    if x.ndim != 1:
        raise ValueError("x and y must be 1-D arrays.")

    # --- Find zero crossings by linear interpolation between consecutive points
    sign_change = y[:-1] * y[1:] < 0
    idx = np.where(sign_change)[0]
    x0 = x[idx] - y[idx] * (x[idx+1] - x[idx]) / (y[idx+1] - y[idx])

    if x0.size == 0:
        return x0

    # Current limits after the main curve is plotted
    ymin, ymax = ax.get_ylim()
    yr = ymax - ymin if ymax > ymin else 1.0

    # --- Draw vertical guides ---
    if span == 'down':
        y1 = np.minimum(0, ymin)
        y2 = 0.0
        ax.vlines(x0, y1, y2, colors=color, linestyles=linestyle, linewidth=linewidth)
    elif span == 'up':
        y1 = 0.0
        y2 = np.maximum(0, ymax)
        ax.vlines(x0, y1, y2, colors=color, linestyles=linestyle, linewidth=linewidth)
    elif span == 'full':
        ax.vlines(x0, ymin, ymax, colors=color, linestyles=linestyle, linewidth=linewidth)
    elif span == 'notch':
        # handled in tick block below
        pass
    else:
        raise ValueError("span must be one of {'down','up','full','notch'}")

    # --- Optional point markers at (x0, 0) ---
    if show_points:
        ax.scatter(x0, np.zeros_like(x0), color=color, zorder=3)

    # --- Add tick on the x-axis at y=0 and custom label ---
    if add_tick or (label_text not in (None, "")):
        tcol = tick_color or color
        lcol = label_color or tcol or color
        tick_h = tick_height_frac * yr

        if add_tick:
            # Draw a short vertical segment that crosses y=0 (downward notch)
            ax.vlines(x0, 0.0, -tick_h, colors=tcol, linewidth=tick_linewidth)

        if label_text not in (None, ""):
            # Place label below the axis at y=0 (using offset in points)
            for xi in np.atleast_1d(x0):
                ax.annotate(label_text, xy=(xi, 0.0), xycoords='data',
                            xytext=(0, -label_offset_pts), textcoords='offset points',
                            ha='center', va='top', color=lcol, fontsize=label_fontsize)

    # --- Optional numeric annotations near the point ---
    if annotate_values:
        for xi in x0:
            ax.annotate(f'{xi:.3g}', xy=(xi, 0.0), xytext=(0, -1.8*label_offset_pts),
                        textcoords='offset points', ha='center', va='top',
                        color=color)

    return x0


data_dir = "data/"
figure_dir = "figures/"

plotsize = 3

significant_digits = 6

delta = 0.3

xlim = [0,4.5]
ylim = [-1.3, 1.3]

j_values = np.loadtxt(data_dir + "jvalues.csv")
j_vals_plot = []

eigenvalues_plot = []
nr_eigvals = 0
YSR_energy = [[],[]]
YSR_magnetization = [[], []]

for j in j_values:
    try:
        eigenvalues = np.loadtxt(data_dir + "eigenvalues_J_{0:.6f}.csv".format(round(j,significant_digits)))
        nr_eigvals = len(eigenvalues)
        magnetizations_per_state = np.loadtxt(data_dir + "magnetizations_per_state_J_{0:.6f}.csv".format(round(j,significant_digits)))
        eigenvalues, magnetizations_per_state = zip(*sorted(zip(eigenvalues, magnetizations_per_state )))
        YSR_low, YSR_high = findYSRIndex(eigenvalues)
        YSR_energy[0].append(eigenvalues[YSR_low])
        YSR_energy[1].append(eigenvalues[YSR_high])
        YSR_magnetization[0].append(magnetizations_per_state[YSR_low])
        YSR_magnetization[1].append(magnetizations_per_state[YSR_high])
        eigenvalues = np.delete(eigenvalues, [YSR_low, YSR_high])
        eigenvalues_plot.append(eigenvalues)
        j_vals_plot.append(j)
    except:
        print("Error could not find j value {0:.6f}".format(round(j,significant_digits)))



############### Do the eigenvalues plot with the ysr energy colored by the magnetization ###########


fig, ax = plt.subplots(1,1, figsize=[1.618*plotsize,plotsize], sharex=True, sharey=False) # applyPlotSettings(num_plots,1)
fig.patch.set_facecolor('gray')
fig.patch.set_alpha(0.0)
ax.grid(True, alpha=0.3, zorder=-1)
ax.set_zorder(3)
eigenvalues_plot = np.array(eigenvalues_plot)


ysr_energies_plot = [[], []]
for idx, mags in enumerate(YSR_magnetization):
    for idx_vec, mag in enumerate(mags):
        if(mag > 0):
            ysr_energies_plot[idx].append(YSR_energy[0][idx_vec])
        else:
            ysr_energies_plot[idx].append(YSR_energy[1][idx_vec])




ysr_energies_plot[0][0] = YSR_energy[0][0]
ysr_energies_plot[1][0] = YSR_energy[1][0]


# Label the two YSR states in the plot
# Where to place the labels
j_label_positions = [0.75, 0.75]
# label_texts = [r'$\left| \uparrow \right \rangle$', r'$\left| \downarrow \right \rangle$']
label_texts = [r'$\left \langle s_z  \right \rangle_{YSR} = \frac{1}{2}$', r'$\left \langle s_z  \right \rangle_{YSR} = -\frac{1}{2}$']


colors = ['b', 'r']

for idx, ysr in enumerate(ysr_energies_plot):
    ysr = np.array(ysr)
    ax.plot(j_vals_plot, ysr/delta, color=colors[idx])

    y_label = np.interp(j_label_positions[idx], j_vals_plot, ysr/delta)
    color = colors[idx]
    # Plot label, offset slightly so it doesn’t overlap the line
    offset = 0.15*np.sign(ysr[0])
    ax.text(j_label_positions[idx], y_label + offset, label_texts[idx],
            color=color,
            fontsize=10,
            ha='left', va='center',
            clip_on=True)   # respects axis limits




# for idx_ev in range(len(eigenvalues_plot[0,:])):
#     if(np.min(np.abs(eigenvalues_plot[:,idx_ev]))> ylim[1]*delta):
#         continue
#     ax.plot(j_vals_plot, eigenvalues_plot[:,idx_ev]/delta, color='dimgray')

ax.axhspan(ylim[0], -1, color='dimgray', zorder=4)
ax.axhspan(1, ylim[1], color='dimgray', zorder=4)


mark_zero_crossings(ax,j_vals_plot, ysr_energies_plot[0], annotate_values=False, show_points=False, linewidth=0.75, span='notch', tick_height_frac=0.025, tick_linewidth=1.2, label_offset_pts=7)


ax.set_xlim(xlim)
ax.set_ylim(ylim)

ax.set_xlabel(r'$J$')
ax.set_ylabel(r'$E\,/\,\Delta$')
linewidth=0.5
ax.axhline(y=0, color='k',linewidth=linewidth)



# fig.tight_layout()
plt.savefig(figure_dir + "eigenvalues.pdf", bbox_inches="tight")   # vector
plt.savefig(figure_dir + "eigenvalues.pgf", bbox_inches="tight")   # vector
plt.savefig(figure_dir + "eigenvalues.png", dpi=300, bbox_inches="tight")  # raster
import matplot2tikz
matplot2tikz.Flavors.latex.preamble()
# matplot2tikz.clean_figure()
matplot2tikz.save(figure_dir + "eigenvalues.tex")
# matplot2tikz.clean_figure(figure=fig_bottom)
# matplot2tikz.save("figures/Thesis/anomalous_greens_compare_bottom.tex", figure=fig_bottom)