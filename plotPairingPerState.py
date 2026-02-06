import matplotlib.pyplot as plt
import numpy as np
from decimal import Decimal, ROUND_UP


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import TwoSlopeNorm
from matplotlib.cm import ScalarMappable

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
                 vmin=None, vmax=None, center=0.0, rasterized=False,
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


data_dir = "data_test/"
figure_dir = "figures/"

significant_digits = 6

delta = 0.3

xlim = [0,4.5]
ylim = [-1.5, 1.5]

j_values = np.loadtxt(data_dir + "jvalues.csv")
j_vals_plot = []

eigenvalues_plot = []
pairings_plot = []
nr_eigvals = 0
for j in j_values:
    try:
        eigenvalues = np.loadtxt(data_dir + "eigenvalues_J_{0:.6f}.csv".format(round(j,significant_digits)))
        nr_eigvals = len(eigenvalues)
        pairings_per_state = np.loadtxt(data_dir + "pairings_per_state_real_J_{0:.6f}.csv".format(round(j,significant_digits)))
        eigenvalues, pairings_per_state = zip(*sorted(zip(eigenvalues, pairings_per_state )))
        eigenvalues_plot.append(eigenvalues)
        pairings_plot.append(pairings_per_state)
        j_vals_plot.append(j)
    except:
        print("Error could not find j value {0:.6f}".format(round(j,significant_digits)))



############### Do the eigenvalues plot with the energies colored by the pairings ###########


fig, ax = plt.subplots(1,1, figsize=[1.618*3,3], sharex=True, sharey=False) # applyPlotSettings(num_plots,1)
fig.patch.set_facecolor('gray')
fig.patch.set_alpha(0.0)
eigenvalues_plot = np.array(eigenvalues_plot)
pairings_plot = np.array(pairings_plot)

eigenvalues_plot_trimmed = []
pairings_plot_trimmed = []
for idx_ev in range(len(eigenvalues_plot[0][:])):
    if(np.min(np.abs(eigenvalues_plot[:,idx_ev])) > ylim[1]*delta):
        continue
    eigenvalues_plot_trimmed.append(eigenvalues_plot[:,idx_ev])
    pairings_plot_trimmed.append(pairings_plot[:,idx_ev])



# Suppose shapes: lam: (N,), E: (M, N), S: (M, N)
# Compute a global symmetric range
S = np.array(pairings_plot_trimmed)
lam = np.array(j_vals_plot)
E = np.array(eigenvalues_plot_trimmed)/delta
a = np.nanmax(np.abs(S))

print(S.shape)
print(lam.shape)
print(E.shape)

# from matplotlib.colors import TwoSlopeNorm
# norm = TwoSlopeNorm(vmin=-a, vcenter=0.0, vmax=+a)


from matplotlib.colors import SymLogNorm
norm = SymLogNorm(linthresh=1e-40, vmin=-a, vmax=a, base=10)
# # Use 'seismic' with ScalarMappable as above


fig, ax = plt.subplots(figsize=(7, 5))
cmap = 'seismic'

for n in range(E.shape[0]):
    # Reuse the same norm so the color meaning is consistent
    lc, _ = colored_line(lam, E[n], S[n], ax=ax, cmap=cmap, linewidth=1.2,
                         vmin=-a, vmax=+a)

# One colorbar for all
from matplotlib.cm import ScalarMappable
sm = ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, pad=0.02)
cbar.set_label('Pairing')

ax.set_xlabel(r'$J$')
ax.set_ylabel(r'$\epsilon\,/\,\Delta$')
# ax.set_title('Energy Spectrum (multiple branches) with S(λ) as color')
ax.grid(True, alpha=0.3)

ax.set_xlim(xlim)
ax.set_ylim(ylim)

linewidth=0.5
ax.axvline(x=0, color='k',linewidth=linewidth)

fig.tight_layout()
fig.savefig(figure_dir + "pairing.png")
import matplot2tikz
matplot2tikz.Flavors.latex.preamble()
# matplot2tikz.clean_figure()
matplot2tikz.save(figure_dir + "pairing.tex")
# matplot2tikz.clean_figure(figure=fig_bottom)
# matplot2tikz.save("figures/Thesis/anomalous_greens_compare_bottom.tex", figure=fig_bottom)