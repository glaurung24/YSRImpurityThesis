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




data_dir = "data_noJ/"
figure_dir = "figures/"
dos_file = "dos_noJ_delta_0.300000.csv"
dos_normal_file = "dos_noJ_delta_0.000000.csv"
energies_file = "dos_noJ_energies_delta_0.300000.csv"
energies_normal_file = "dos_noJ_energies_delta_0.000000.csv"

plotsize = 2.5

significant_digits = 6

delta = 0.3

ylim = [-0.01 ,0.69]
xlim = [-2, 2]


energies_plot = []
nr_eigvals = 0



fig, ax = plt.subplots(1,1, figsize=[1.618*plotsize,plotsize], sharex=True, sharey=False) # applyPlotSettings(num_plots,1)
fig.patch.set_facecolor('gray')
fig.patch.set_alpha(0.0)

energy_list = []
smooth_dos_list = []
peak_idxs = []
factor = 0.05

energies = np.loadtxt(data_dir + energies_file)
energies = np.array(energies)/delta
dos = np.loadtxt(data_dir + dos_file)
dos_normal = np.loadtxt(data_dir + dos_normal_file)

smoothed_dos = gaussian_filter1d(dos, sigma=1.5)
smoothed_dos_normal = gaussian_filter1d(dos_normal, sigma=1.5)


x_max = 0.9
x_min = -x_max
# Create the *combined* y array
ax.plot(energies, smoothed_dos, label="Superconductor")
ax.plot(energies, smoothed_dos_normal, "--", label="Normal state")

ax.set_xlim(xlim)
# ax.set_ylim(ylim)

ax.set_xlabel(r'$E\,/\, \Delta$')
ax.set_ylabel(r'DOS')
linewidth=0.5
# ax.axhline(y=0, color='k',linewidth=linewidth)
ax.grid(True, alpha=0.3)

# ax.legend()
leg = ax.legend(markerscale=0.7, handlelength=0.9, fontsize=8, loc="upper right")
fig.tight_layout()
plt.savefig(figure_dir + "sc_dos.pdf", bbox_inches="tight")   # vector
plt.savefig(figure_dir + "sc_dos.pgf", bbox_inches="tight")   # vector
plt.savefig(figure_dir + "sc_dos.png", dpi=300, bbox_inches="tight")  # raster

# import matplot2tikz
# matplot2tikz.Flavors.latex.preamble()
# # matplot2tikz.clean_figure()
# matplot2tikz.save(figure_dir + "sc_dos.tex")
# # matplot2tikz.clean_figure(figure=fig_bottom)
# # matplot2tikz.save("figures/Thesis/anomalous_greens_compare_bottom.tex", figure=fig_bottom)