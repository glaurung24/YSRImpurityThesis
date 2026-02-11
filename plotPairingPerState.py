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
                 vmin=None, vmax=None, center=0.0, rasterized=False, norm=None,
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




data_dir = "data_ultrares/"
figure_dir = "figures/"
# plotTargetFiles = "pairings_per_state_non_local_real_J_"
lower_threshold_pairing = 1e-7
delta = 0.3

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

plotsize = 2.9

xlim = [0,4.5]
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


for j in j_values:
    try:
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

############### Calculate "ldos" of the pairngs #####################
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
for n in range(E.shape[0]):
    # Reuse the same norm so the color meaning is consistent
    lc, _ = colored_line(lam, E[n], S[n], ax=ax, cmap=cmap, linewidth=linewidth,
                         vmin=-a, vmax=+a, norm=norm)
    # lc, _ = colored_line(lam, E[n], S[n], ax=ax_bottom, cmap=cmap, linewidth=linewidth,
    #                      vmin=-a, vmax=+a, norm=norm, alpha=alpha)

for n in range(len(YSR_energy)):
    # Reuse the same norm so the color meaning is consistent
    lc, _ = colored_line(lam, YSR_energy[n], YSR_pairing[n], ax=ax, cmap=cmap, linewidth=linewidth,
                         vmin=-a, vmax=+a, norm=norm)
    # lc, _ = colored_line(lam, YSR_energy[n], YSR_pairing[n], ax=ax_bottom, cmap=cmap, linewidth=linewidth,
    #                      vmin=-a, vmax=+a, norm=norm)

from matplotlib.colors import SymLogNorm
a = np.max(ldos_pairing)
norm_bottom = SymLogNorm(linthresh=1e-3, vmin=-a, vmax=a, base=10)
fig_test, ax_test = plt.subplots(1,1)
ax_test.plot(energy_grid, ldos_pairing[0,:])
ax_test.plot(energy_grid, ldos_pairing[20,:])
fig_test.savefig(figure_dir + "test.png")

# ax_bottom.grid(True, alpha=0.3)


shading = 'auto' # 'auto' 'flat'

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



# One colorbar for all
from matplotlib.cm import ScalarMappable
sm = ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, pad=0.02)
cbar.set_label(r'$u^{(n)}_\uparrow v^{*(n)}_\downarrow$')


# --- Attach colorbar WITHOUT changing layout ---
# from mpl_toolkits.axes_grid1 import make_axes_locatable
# divider_top = make_axes_locatable(ax_top)
# cax_top = divider_top.append_axes("right", size="3%", pad=0.15)

# divider = make_axes_locatable(ax_bottom)
# cax = divider.append_axes("right", size="3%", pad=0.15)
sm_bottom = ScalarMappable(norm=norm_bottom, cmap=cmap)
sm_bottom.set_array([])
cbar = fig.colorbar(sm_bottom, ax=ax, pad=0.02)
cbar_bottom = fig.colorbar(sm_bottom, ax=ax_bottom, pad=0.02)
cbar_bottom.set_label(r'$f_n$')


# # ---- Control ticks: show every second major tick and no minor ticks ----
# cbar.ax.yaxis.set_major_locator(mticker.LogLocator(base=10))   # decades
# cbar.ax.yaxis.set_major_formatter(mticker.LogFormatter())      # 10^n style
# cbar.ax.yaxis.set_minor_locator(mticker.NullLocator())         # no minor ticks

ticks = cbar.get_ticks()  # or cbar.ax.get_yticks()
if len(ticks) > 0:
    cbar.set_ticks(ticks[::2])  # keep every second tick label


ax.set_xlabel(r'$J$')
ax.set_ylabel(r'$\epsilon\,/\,\Delta$')
ax_top.set_ylabel(r'$\Delta( \mathbf{x}_0)\,/\,\Delta$')
ax_bottom.set_xlabel(r'$J$')
ax_bottom.set_ylabel(r'$\epsilon\,/\,\Delta$')
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
# ax_top.grid(True, alpha=0.3)

# fig2.tight_layout()
fig2.savefig(figure_dir + "pairing_delta.pdf", dpi=400, bbox_inches="tight")   # vector
fig2.savefig(figure_dir + "pairing_delta.png", dpi=400, bbox_inches="tight")  # raster
fig2.savefig(figure_dir + "pairing_delta.pgf", dpi=400, bbox_inches="tight")   # vector

fig.tight_layout()
fig.savefig(figure_dir + "pairing.pdf", dpi=300, bbox_inches="tight")   # vector
fig.savefig(figure_dir + "pairing.pgf", dpi=300, bbox_inches="tight")   # vector
fig.savefig(figure_dir + "pairing.png", dpi=300, bbox_inches="tight")  # raster


# import matplot2tikz
# matplot2tikz.Flavors.latex.preamble()
# matplot2tikz.clean_figure()
# matplot2tikz.save(figure_dir + "pairing.tex")
# matplot2tikz.clean_figure(figure=fig_bottom)
# matplot2tikz.save("figures/Thesis/anomalous_greens_compare_bottom.tex", figure=fig_bottom)