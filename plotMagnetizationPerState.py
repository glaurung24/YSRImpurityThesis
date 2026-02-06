import matplotlib.pyplot as plt
import numpy as np
from decimal import Decimal, ROUND_UP

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
nr_eigvals = 0
YSR_energy = [[],[]]
YSR_magnetization = [[], []]

for j in j_values:
    try:
        eigenvalues = np.loadtxt(data_dir + "eigenvalues_J_{0:.6f}.csv".format(round(j,significant_digits)))
        nr_eigvals = len(eigenvalues)
        magnetizations_per_state = np.loadtxt(data_dir + "magnetizations_per_state_J_{0:.6f}.csv".format(round(j,significant_digits)))
        eigenvalues, magnetizations_per_state, pairings_per_state = zip(*sorted(zip(eigenvalues, magnetizations_per_state, pairings_per_state )))
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


fig, ax = plt.subplots(1,1, figsize=[1.618*3,3], sharex=True, sharey=False) # applyPlotSettings(num_plots,1)
fig.patch.set_facecolor('gray')
fig.patch.set_alpha(0.0)
eigenvalues_plot = np.array(eigenvalues_plot)


for idx_ev in range(len(eigenvalues_plot[0,:])):
    if(np.min(np.abs(eigenvalues_plot[:,idx_ev]))> ylim[1]*delta):
        continue
    ax.plot(j_vals_plot, eigenvalues_plot[:,idx_ev]/delta, color='k')

ysr_energies_plot = [[], []]
for idx, mags in enumerate(YSR_magnetization):
    for idx_vec, mag in enumerate(mags):
        if(mag > 0):
            ysr_energies_plot[idx].append(YSR_energy[0][idx_vec])
        else:
            ysr_energies_plot[idx].append(YSR_energy[1][idx_vec])

ysr_energies_plot[0][0] = YSR_energy[0][0]
ysr_energies_plot[1][0] = YSR_energy[1][0]

colors = ['b', 'r']
for idx, ysr in enumerate(ysr_energies_plot):
    ysr = np.array(ysr)
    ax.plot(j_vals_plot, ysr/delta, color=colors[idx])
ax.set_xlim(xlim)
ax.set_ylim(ylim)

ax.set_xlabel(r'$J$')
ax.set_ylabel(r'$\epsilon\,/\,\Delta$')
linewidth=0.5
ax.axvline(x=0, color='k',linewidth=linewidth)

fig.tight_layout()
fig.savefig(figure_dir + "eigenvalues.png")
import matplot2tikz
matplot2tikz.Flavors.latex.preamble()
# matplot2tikz.clean_figure()
matplot2tikz.save(figure_dir + "eigenvalues.tex")
# matplot2tikz.clean_figure(figure=fig_bottom)
# matplot2tikz.save("figures/Thesis/anomalous_greens_compare_bottom.tex", figure=fig_bottom)