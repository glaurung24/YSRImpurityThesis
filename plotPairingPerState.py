import matplotlib.pyplot as plt
import numpy as np
from decimal import Decimal, ROUND_UP


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import TwoSlopeNorm
from matplotlib.cm import ScalarMappable


# plt.rcParams.update({
#     'font.family': 'serif',
#     'font.serif': ['STIXGeneral', 'Nimbus Roman', 'Nimbus Roman No9 L', 'Times', 'DejaVu Serif'],
#     'mathtext.fontset': 'stix',  # matches Times aesthetics for math
# })


def scatter_spectrum(
    x, Ys, Ws,
    ax=None,
    cmap="seismic",
    norm=None,
    markersize=12,
    alpha=0.9,
    vmin=None, vmax=None,
    center=0.0,
    colorbar=False,
    cbar_label="",
    **scatter_kw
):
    """
    Plot a collection of energy curves as colored scatter points.

    Parameters
    ----------
    x : 1D array (N,)
        The parameter (e.g. lambda)
    Ys : list of 1D arrays
        Each y-array is one energy branch of shape (N,)
    Ws : list of 1D arrays
        Each weight array (wavefunction weight) of shape (N,)
    ax : Matplotlib Axes
    cmap : str
        Diverging colormap (negative→blue, positive→red)
    markersize : float
    alpha : float
    vmin, vmax : float
        Optional color scale limits. If None, symmetric from global |Ws|.
    center : float
        Center value for diverging colormap (usually 0)
    colorbar : bool
        Whether to draw a colorbar
    cbar_label : str
        Label for the colorbar
    scatter_kw : other keyword arguments passed to ax.scatter

    Returns
    -------
    ax : Matplotlib Axes
    norm : Normalize
        Color normalization object
    """
    if ax is None:
        ax = plt.gca()

    # Convert Ys and Ws to lists
    Ys = list(Ys)
    Ws = list(Ws)

    # Determine global color range
    if vmin is None or vmax is None:
        all_w = np.concatenate([np.asarray(w) for w in Ws])
        a = np.nanmax(np.abs(all_w)) if all_w.size else 1.0
        vmin, vmax = -a, +a

    if not norm:
        # Diverging normalization: negative→blue, positive→red, zero→white
        norm = TwoSlopeNorm(vmin=vmin, vcenter=center, vmax=vmax)

    # Draw each branch as a scatter
    for y, w in zip(Ys, Ws):
        ax.scatter(
            x, y,
            c=w,
            cmap=cmap,
            norm=norm,
            s=markersize,
            alpha=alpha,
            linewidths=0,
            **scatter_kw
        )

    # Optional colorbar
    if colorbar:
        sm = ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label(cbar_label)

    return ax, norm




def align_ylabels(ax_list, x=-0.12, pad=10, use_constrained=True):
    """Align y-labels across axes and optionally use constrained_layout."""
    fig = ax_list[0].figure
    if use_constrained:
        # Only effective if subplots were created with constrained_layout=True
        fig.align_ylabels(ax_list)
    # Force identical label coords and padding (works regardless of layout engine)
    for ax in ax_list:
        ax.set_ylabel(ax.get_ylabel(), labelpad=pad)
        ax.yaxis.set_label_coords(x, 0.5, transform=ax.transAxes)


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

"""
Plot a 2D line (x,y) whose color varies with c, using a diverging colormap.

Parameters
----------
x, y : array-like (N,)
    Coordinates of the line.
c : array-like (N,)
    Per-point values to color by (e.g., third variable). Should align with x,y.
ax : matplotlib.axes.Axes or None
    Target axes; if None, uses current axes.
cmap : str or Colormap
    Colormap; 'seismic' maps negative->blue, positive->red with white at 0.
linewidth : float
    Line width in points.
vmin, vmax : float or None
    Color scale limits; if None, symmetric limits are computed from |c|.
center : float
    The neutral center value for diverging normalization (usually 0.0).
rasterized : bool
    If True, rasterize collection (useful for many points).
**line_kw :
    Passed to LineCollection (e.g., 'alpha', 'antialiased', 'zorder').

Returns
-------
lc : LineCollection
    The colored line collection.
norm : Normalize
    The normalization used (TwoSlopeNorm).
"""
def colored_line(x, y, c, ax=None, cmap='seismic', linewidth=2.0,
                 vmin=None, vmax=None, center=0.0, rasterized=False, norm=None, zorder=None,
                 **line_kw):

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    c = np.asarray(c, dtype=float)
    if not (x.shape == y.shape == c.shape):
        raise ValueError("x, y, c must have the same shape (N,)")

    # Build line segments between consecutive points
    points = np.column_stack([x, y]).reshape(-1, 1, 2)         # (N,1,2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)  # (N-1,2,2)

    # Color each segment by the average c on its endpoints
    c_seg = 0.5 * (c[:-1] + c[1:])

    # Remove segments containing NaNs so gaps appear where data is missing
    valid = ~np.isnan(segments).any(axis=(1, 2)) & ~np.isnan(c_seg)
    segments = segments[valid]
    c_seg = c_seg[valid]

    # Choose symmetric limits if not provided
    if vmin is None or vmax is None:
        a = np.nanmax(np.abs(c_seg)) if c_seg.size > 0 else 1.0
        vmin, vmax = -a, +a

    # Diverging normalization: center -> white
    if(norm is None):
        norm = TwoSlopeNorm(vmin=vmin, vcenter=center, vmax=vmax)

    # Create the colored line collection
    lc = LineCollection(
        segments,
        cmap=cmap,
        norm=norm,
        linewidth=linewidth,
        rasterized=rasterized,
        zorder=zorder,
        joinstyle="round",
        capstyle="round",
        **line_kw
    )
    lc.set_array(c_seg)

    if ax is None:
        ax = plt.gca()
    ax.add_collection(lc)
    ax.autoscale()  # autoscale view to data extent
    return lc, norm

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


def compute_ldos(energies, amplitudes, E_grid, sigma):
    """
    energies   : array of eigenvalues E_n
    amplitudes : array of |psi_n(i)|^2 for fixed site i
    E_grid     : energy grid for LDOS
    sigma      : Gaussian broadening
    """
    ldos = np.zeros_like(E_grid, dtype='float64')

    prefactor = 1.0 / (sigma * np.sqrt(2 * np.pi))

    for idx in range(len(energies)):
        gaussian = prefactor * np.exp(-0.5 * ((E_grid - energies[idx]) / sigma)**2)
        ldos += amplitudes[idx] * gaussian

    return ldos


def compute_edges(x):
    """Compute monotonic bin edges from non-uniform centers x (1D)."""
    x = np.asarray(x)
    mids = 0.5 * (x[1:] + x[:-1])
    # Extrapolate edges at both ends
    first = x[0] - (mids[0] - x[0])
    last  = x[-1] + (x[-1] - mids[-1])
    edges = np.concatenate([[first], mids, [last]])
    return edges




data_dir = "data_ultrares2/"
figure_dir = "figures/"
# plotTargetFiles = "pairings_per_state_non_local_real_J_"
lower_threshold_pairing = 1e-5
delta = 0.3
plotLine = False


sc = True
if sc:
    j_file = "jvalues_sc.csv"
    pairings_file = "pairings_per_state_sc_real_J_"
    energies_file = "eigenvalues_sc_J_"
    delta_dist = np.loadtxt(data_dir + "delta_sc_J_0.000000.csv")
    delta_file = "delta_sc_J_"
    delta = delta_dist[0]
else:
    j_file = "jvalues.csv"
    pairings_file = "pairings_per_state_real_J_"
    energies_file = "eigenvalues_J_"
    delta_file = "delta_J_"

significant_digits = 6

plotsize = 2.75

xlim = [0,3]
ylim = [-1.3, 1.3]

j_values = np.loadtxt(data_dir + j_file)
j_vals_plot = []

eigenvalues_plot = []
pairings_plot = []
nr_eigvals = 0
delta_plot = []

YSR_energy = [[],[]]
YSR_pairing= [[], []]

eigenvalues_all = []
pairings_all = []

j_values = np.loadtxt(data_dir + "j_files_sc_list.txt")

for j in j_values:
    try:
        j = float(j)
        eigenvalues = np.loadtxt(data_dir + energies_file + "{0:.6f}.csv".format(round(j,significant_digits)))
        delta_vec = np.loadtxt(data_dir + delta_file + "{0:.6f}.csv".format(round(j,significant_digits)))
        size = int(np.sqrt(len(delta_vec)))
        delta_array = delta_vec.reshape((size,size))
        middle = int(size/2)
        nr_eigvals = len(eigenvalues)
        pairings_per_state = np.loadtxt(data_dir + pairings_file + "{0:.6f}.csv".format(round(j,significant_digits)))
        eigenvalues, pairings_per_state = zip(*sorted(zip(eigenvalues, pairings_per_state )))
        eigenvalues_all.append(eigenvalues)
        pairings_all.append(pairings_per_state)
        YSR_low, YSR_high = findYSRIndex(eigenvalues)
        if(pairings_per_state[YSR_low] < 0.):
            YSR_low, YSR_high = YSR_high, YSR_low
        YSR_energy[0].append(eigenvalues[YSR_low])
        YSR_energy[1].append(eigenvalues[YSR_high])
        YSR_pairing[0].append(pairings_per_state[YSR_low])
        YSR_pairing[1].append(pairings_per_state[YSR_high])
        eigenvalues = np.delete(eigenvalues, [YSR_low, YSR_high])
        pairings_per_state = np.delete(pairings_per_state, [YSR_low, YSR_high])

        eigenvalues_plot.append(eigenvalues)
        pairings_plot.append(pairings_per_state)
        delta_plot.append(delta_array[middle, middle])
        j_vals_plot.append(j)
    except:
        print("Error could not find j value {0:.6f}".format(round(j,significant_digits)))







############### Do the eigenvalues plot with the energies colored by the pairings ###########


fig, ax = plt.subplots(1,1, figsize=[1.618*plotsize,plotsize], sharex=True, sharey=False) # applyPlotSettings(num_plots,1)
fig.patch.set_facecolor('gray')
fig.patch.set_alpha(0.0)

fig2, (ax_top, ax_bottom) = plt.subplots(2,1, figsize=[1.618*plotsize,plotsize], sharex=True, sharey=False, gridspec_kw={'height_ratios': [1, 2], 'hspace': 0}) #figsize=(7, 5)gridspec_kw={'hspace': 0}
# Remove vertical space between axes
fig2.subplots_adjust(hspace=0)  # exact zero space

linewidth = 1.5

delta_plot = np.array(delta_plot)
ax_top.plot(j_vals_plot, delta_plot/delta, linewidth=linewidth)

qpt = mark_zero_crossings(ax_top, j_vals_plot, delta_plot, linewidth=0.75, linestyle="-", span='notch', show_points=False, annotate_values=False, add_tick=False, label_text='')



eigenvalues_plot = np.array(eigenvalues_plot)
pairings_plot = np.array(pairings_plot)

eigenvalues_plot_trimmed = []
pairings_plot_trimmed = []
for idx_ev in range(len(eigenvalues_plot[0][:])):
    if(np.min(np.abs(eigenvalues_plot[:,idx_ev])) > ylim[1]*delta):
        continue
    eigenvalues_plot_trimmed.append(eigenvalues_plot[:,idx_ev])
    pairings_plot_trimmed.append(pairings_plot[:,idx_ev])

eigenvalues_plot_trimmed = np.array(eigenvalues_plot_trimmed)
print(np.shape(eigenvalues_plot_trimmed))



# Suppose shapes: lam: (N,), E: (M, N), S: (M, N)
# Compute a global symmetric range
j_vals_plot = np.array(j_vals_plot)
S = np.array(pairings_plot_trimmed)
lam = np.array(j_vals_plot)
E = np.array(eigenvalues_plot_trimmed)/delta
a = np.nanmax(np.abs(S))
YSR_energy = np.array(YSR_energy)/delta


# from matplotlib.colors import TwoSlopeNorm
# norm = TwoSlopeNorm(vmin=-a, vcenter=0.0, vmax=+a)


from matplotlib.colors import SymLogNorm
norm = SymLogNorm(linthresh=lower_threshold_pairing, vmin=-a, vmax=a, base=10)
# # Use 'seismic' with ScalarMappable as above



cmap = 'seismic'
alpha=0.85
linewidth_non_ysr=0.35
linewidth_ysr = 1.0
zorder_lines = -10
for n in range(E.shape[0]):
    # Reuse the same norm so the color meaning is consistent
    lc, _ = colored_line(lam, E[n], S[n], ax=ax, cmap=cmap, linewidth=linewidth_non_ysr,
                         vmin=-a, vmax=+a, norm=norm, zorder=zorder_lines)
    # lc, _ = colored_line(lam, E[n], S[n], ax=ax_bottom, cmap=cmap, linewidth=linewidth,
    #                      vmin=-a, vmax=+a, norm=norm, alpha=alpha)

for n in range(len(YSR_energy)):
    # Reuse the same norm so the color meaning is consistent
    lc, _ = colored_line(lam, YSR_energy[n], YSR_pairing[n], ax=ax, cmap=cmap, linewidth=linewidth_ysr,
                         vmin=-a, vmax=+a, norm=norm, zorder=zorder_lines)
    # lc, _ = colored_line(lam, YSR_energy[n], YSR_pairing[n], ax=ax_bottom, cmap=cmap, linewidth=linewidth,
    #                      vmin=-a, vmax=+a, norm=norm)


ax.set_rasterization_zorder(zorder_lines+1)



# ax_bottom.grid(True, alpha=0.3)


shading = 'auto' # 'auto' 'flat'
plotPairingSmoothed = False

cbar_label_bottom = r'$\delta_{\nu, \mathbf{x}_0}$'
from matplotlib.cm import ScalarMappable
if(plotPairingSmoothed):
    ############### Calculate "ldos" of the pairngs #####################

    from matplotlib.colors import SymLogNorm
    a = np.max(ldos_pairing)
    norm_bottom = SymLogNorm(linthresh=1e-3, vmin=-a, vmax=a, base=10)
    fig_test, ax_test = plt.subplots(1,1)
    ax_test.plot(energy_grid, ldos_pairing[0,:])
    ax_test.plot(energy_grid, ldos_pairing[20,:])
    fig_test.savefig(figure_dir + "test.png")

    resolution = len(j_vals_plot)
    sigma = (ylim[1]-ylim[0])*delta/resolution/3
    energy_grid = np.linspace(delta*ylim[0], delta*ylim[1], resolution)
    print(sigma)

    eigenvalues_all = np.array(eigenvalues_all)
    pairings_all = np.array(pairings_all)
    eigenvalues_all_trimmed = []
    pairings_all_trimmed = []
    for idx_ev in range(len(eigenvalues_all[0][:])):
        if(np.min(np.abs(eigenvalues_all[:,idx_ev])) > ylim[1]*delta*1.1):
            continue
        eigenvalues_all_trimmed.append(eigenvalues_all[:,idx_ev])
        pairings_all_trimmed.append(pairings_all[:,idx_ev])

    eigenvalues_all_trimmed = np.array(eigenvalues_all_trimmed)
    pairings_all_trimmed = np.array(pairings_all_trimmed)
    ldos_pairing = []

    for idx in range(len(eigenvalues_all_trimmed[:][0])):
        ldos = compute_ldos(eigenvalues_all_trimmed[:,idx], pairings_all_trimmed[:,idx], energy_grid, sigma)
        ldos_pairing.append(ldos)

    ldos_pairing = np.array(ldos_pairing)
    if(shading == 'gouraud'):
        ax_bottom.pcolormesh(j_vals_plot,energy_grid/delta, np.transpose(ldos_pairing), 
                        #   extent=[ylim[0], ylim[1],
                        #           j_vals_plot.min(), j_vals_plot.max()],
                        #   aspect='auto', origin='lower', 
                        shading=shading,
                        # shading='gouraud',
                        # shading='flat',
                        norm=norm_bottom,
                        cmap=cmap,
                        rasterized=True
                        )
    else:
        ax_bottom.pcolormesh(compute_edges(j_vals_plot),compute_edges(energy_grid/delta), np.transpose(ldos_pairing), 
                        #   extent=[ylim[0], ylim[1],
                        #           j_vals_plot.min(), j_vals_plot.max()],
                        #   aspect='auto', origin='lower', 
                        shading=shading,
                        # shading='gouraud',
                        # shading='flat',
                        norm=norm_bottom,
                        cmap=cmap,
                        rasterized=True
                        )
    sm_bottom = ScalarMappable(norm=norm_bottom, cmap=cmap)
    sm_bottom.set_array([])
    cbar = fig.colorbar(sm_bottom, ax=ax, pad=0.02)
    cbar_bottom = fig.colorbar(sm_bottom, ax=ax_bottom, pad=0.02)
    cbar_bottom.set_label(cbar_label_bottom)
else:
    if(plotLine):
        for n in range(E.shape[0]):
        # Reuse the same norm so the color meaning is consistent
            lc, _ = colored_line(lam, E[n], S[n], ax=ax_bottom, cmap=cmap, linewidth=linewidth_non_ysr,
                                vmin=-a, vmax=+a, norm=norm, zorder=zorder_lines)

        for n in range(len(YSR_energy)):
            # Reuse the same norm so the color meaning is consistent
            lc, _ = colored_line(lam, YSR_energy[n], YSR_pairing[n], ax=ax_bottom, cmap=cmap, linewidth=linewidth_ysr,
                                vmin=-a, vmax=+a, norm=norm, zorder=zorder_lines)
    else:
        # for n in range(E.shape[0]):
        # # Reuse the same norm so the color meaning is consistent
        #     scatter_spectrum(lam, E[n], S[n], ax=ax_bottom, cmap=cmap, markersize=linewidth_non_ysr,
        #                         vmin=-a, vmax=+a, norm=norm, zorder=zorder_lines)
        scatter_spectrum(lam, E, S, ax=ax_bottom, cmap=cmap, markersize=linewidth_non_ysr,
                    vmin=-a, vmax=+a, norm=norm, zorder=zorder_lines, alpha=0.8)
        # for n in range(len(YSR_energy)):
        #     scatter_spectrum(lam, YSR_energy[n], YSR_pairing[n], ax=ax_bottom, cmap=cmap, markersize=linewidth_ysr,
        #                         vmin=-a, vmax=+a, norm=norm, zorder=zorder_lines)
        scatter_spectrum(lam, YSR_energy, YSR_pairing, ax=ax_bottom, cmap=cmap, markersize=linewidth_ysr,
                            vmin=-a, vmax=+a, norm=norm, zorder=zorder_lines)

    ax_bottom.set_rasterization_zorder(zorder_lines+1)
    sm_bottom = ScalarMappable(norm=norm, cmap=cmap)
    sm_bottom.set_array([])
    cbar = fig.colorbar(sm_bottom, ax=ax, pad=0.02)
    cbar_bottom = fig.colorbar(sm_bottom, ax=ax_bottom, pad=0.02)
    cbar_bottom.set_label(cbar_label_bottom)




# One colorbar for all
sm = ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, pad=0.02)
cbar.set_label(r'$u^{(n)}_\uparrow v^{*(n)}_\downarrow$')



# # ---- Control ticks: show every second major tick and no minor ticks ----
# cbar.ax.yaxis.set_major_locator(mticker.LogLocator(base=10))   # decades
# cbar.ax.yaxis.set_major_formatter(mticker.LogFormatter())      # 10^n style
# cbar.ax.yaxis.set_minor_locator(mticker.NullLocator())         # no minor ticks

ticks = cbar.get_ticks()  # or cbar.ax.get_yticks()
if len(ticks) > 0:
    cbar.set_ticks(ticks[::2])  # keep every second tick label


ax.set_xlabel(r'$J$')
ax.set_ylabel(r'$E\,/\,\Delta$')
ax_top.set_ylabel(r'$\Delta_{\mathbf{x}_0}\,/\,\Delta$')
ax_bottom.set_xlabel(r'$J$')
ax_bottom.set_ylabel(r'$E\,/\,\Delta$')
# ax.set_title('Energy Spectrum (multiple branches) with S(λ) as color')


ax.set_xlim(xlim)
ax.set_ylim(ylim)
ax_bottom.set_xlim(xlim)
ax_bottom.set_ylim(ylim)



pos2 = ax_bottom.get_position()
pos1 = ax_top.get_position()
ax_top.set_position([pos2.x0, pos1.y0, pos2.width, pos1.height])


linewidth=0.5
ax.axhline(y=0, color='k',linewidth=linewidth)
ax.grid(True, alpha=0.3)
ax_bottom.axhline(y=0, color='k',linewidth=linewidth)
ax_top.axhline(y=0, color='k',linewidth=linewidth)

linewidth_jc = 0.5
ax_top.axvline(x=qpt, linewidth=linewidth_jc, color='k')
ax_bottom.axvline(x=qpt, linewidth=linewidth_jc, color='k')


ticks = list(ax_bottom.get_xticks())
labels = [str(lbl) for lbl in ticks]

del ticks[3]
del labels[3]

# Add your custom tick
ticks.append(qpt[0])              # numeric x‑coordinate for the tick
labels.append(r'$J_C$')            # the string label

print(ticks)
print(labels)

ax_bottom.set_xticks(ticks, labels=labels)


align_ylabels([ax_top, ax_bottom], use_constrained=False)

# ax_top.grid(True, alpha=0.3)

# fig2.tight_layout()
fig2.savefig(figure_dir + "pairing_delta.pdf", dpi=600, bbox_inches="tight")   # vector
fig2.savefig(figure_dir + "pairing_delta.png", dpi=600, bbox_inches="tight")  # raster
fig2.savefig(figure_dir + "pairing_delta.pgf", dpi=600, bbox_inches="tight")   # vector

fig.tight_layout()
fig.savefig(figure_dir + "pairing.pdf", dpi=600, bbox_inches="tight")   # vector
fig.savefig(figure_dir + "pairing.pgf", dpi=600, bbox_inches="tight")   # vector
fig.savefig(figure_dir + "pairing.png", dpi=600, bbox_inches="tight")  # raster