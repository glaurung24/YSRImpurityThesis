import matplotlib.pyplot as plt
import numpy as np
from decimal import Decimal, ROUND_UP


import numpy as np
import matplotlib.pyplot as plt

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


data_dir = "data/"
figure_dir = "figures/"
delta = 0.3

sc = True
if sc:
    j_file = "jvalues_sc.csv"
    pairings_file = "pairings_per_state_sc_real_J_"
    energies_file = "eigenvalues_sc_J_"
    delta_dist = np.loadtxt(data_dir + "delta_sc_J_0.000000.csv")
    delta = delta_dist[0]
else:
    j_file = "jvalues.csv"
    pairings_file = "pairings_per_state_real_J_"
    energies_file = "eigenvalues_J_"

significant_digits = 6


radius = 10
xlim = [0,4.5]
ylim = [-1.3, 1.3]

j_values = np.loadtxt(data_dir + j_file)
j_vals_plot = []

eigenvalues_plot = []
pairings_plot = []
nr_eigvals = 0

YSR_energy = [[],[]]
YSR_pairing= [[], []]

for j in j_values:
    try:
        eigenvalues = np.loadtxt(data_dir + energies_file + "{0:.6f}.csv".format(round(j,significant_digits)))
        nr_eigvals = len(eigenvalues)
        pairings_per_state = np.loadtxt(data_dir + pairings_file + "{0:.6f}.csv".format(round(j,significant_digits)))
        eigenvalues, pairings_per_state = zip(*sorted(zip(eigenvalues, pairings_per_state )))
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
YSR_energy = np.array(YSR_energy)/delta

print(S.shape)
print(lam.shape)
print(E.shape)

# from matplotlib.colors import TwoSlopeNorm
# norm = TwoSlopeNorm(vmin=-a, vcenter=0.0, vmax=+a)


from matplotlib.colors import SymLogNorm
norm = SymLogNorm(linthresh=lower_threshold_pairing, vmin=-a, vmax=a, base=10)
# # Use 'seismic' with ScalarMappable as above


fig, ax = plt.subplots(figsize=(7, 5))
cmap = 'seismic'

for n in range(E.shape[0]):
    # Reuse the same norm so the color meaning is consistent
    lc, _ = colored_line(lam, E[n], S[n], ax=ax, cmap=cmap, linewidth=1.2,
                         vmin=-a, vmax=+a, norm=norm)

for n in range(len(YSR_energy)):
    # Reuse the same norm so the color meaning is consistent
    lc, _ = colored_line(lam, YSR_energy[n], YSR_pairing[n], ax=ax, cmap=cmap, linewidth=1.2,
                         vmin=-a, vmax=+a, norm=norm)


# One colorbar for all
from matplotlib.cm import ScalarMappable
sm = ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, pad=0.02)
cbar.set_label('Pairing')

ax.set_xlabel(r'$J$')
ax.set_ylabel(r'$\epsilon\,/\,\Delta$')
# ax.set_title('Energy Spectrum (multiple branches) with S(λ) as color')


ax.set_xlim(xlim)
ax.set_ylim(ylim)

linewidth=0.5
ax.axhline(y=0, color='k',linewidth=linewidth)
ax.grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig(figure_dir + "pairing.png")
import matplot2tikz
matplot2tikz.Flavors.latex.preamble()
# matplot2tikz.clean_figure()
matplot2tikz.save(figure_dir + "pairing.tex")
# matplot2tikz.clean_figure(figure=fig_bottom)
# matplot2tikz.save("figures/Thesis/anomalous_greens_compare_bottom.tex", figure=fig_bottom)