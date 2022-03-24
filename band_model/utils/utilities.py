""" Module with several functions to simplify some calculations/plots
that need to be done several times """
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from . import absorption as ab
from .. import qd_base_data as qbd


def pso_it_abs_preview(ax,
                       it_name,
                       energy=np.linspace(0.4, 3, 600),
                       save=False,
                       savename="Absorption_Coefs.svg"):
    """
    Import data from a particular PSO optimization and plot the
    absorption for all particles
    Args:
        ax: List of axes to make the plots (obtained from plt.subplots)
        it_name: name of the file with the iteration
        energy: energy to plot the absorption profiles
        save: Save the resulting image
    """
    # Columns from the optimization
    columns = [
        "QSize", "me", "mh", "Pl", "Pt", "Eg", "offset", "FoM", "vQSize",
        "vme", "vmh", "vPl", "vPt", "vEg", "voffset"
    ]
    data = pd.read_csv(it_name, sep=" ", names=columns)
    results = []
    # Import the data to a list of dataframes
    for data_i in data.apply(tuple, axis=1):
        Vcb = (data_i[5] - 0.4) * data_i[6]
        Vvb = (data_i[5] - 0.4) * (1 - data_i[6])
        qd_cb = (data_i[0], Vcb, data_i[1], data_i[1])
        qd_vb = (data_i[0], Vvb, data_i[2], data_i[2])
        sim_properties = (15, 0.65, data_i[5], (data_i[3], data_i[4]))
        results.append(
            ab.interband_absorption(energy, qd_cb, qd_vb, sim_properties))
    # Create the array with the flat indexes
    flat_index = np.arange(0, np.prod(ax.shape)).reshape(ax.shape)
    for i in range(ax.shape[0]):
        for k in range(ax.shape[1]):
            ax[i, k].plot(results[0]["Energy"],
                          results[flat_index[i, k]]["Total"])
            ax[i, k].set_title(f"FoM={data['FoM'][flat_index[i, k]]:.2f}",
                               fontsize=18)
            ax[i, k].tick_params(labelsize=14)
            if k == 0:
                ax[i, k].set_ylabel("Absorption Coef (nm$^3$/cm)", fontsize=16)
            if i == 2:
                ax[i, k].set_xlabel("Energy (eV)", fontsize=16)
    if save:
        plt.savefig(savename, transparent=True)


def pso_it_energy_profile(ax,
                          it_name,
                          save=False,
                          savename="Band_diagrams.svg"):
    """
    Utility function to grab the data for all the particles in a iteration
    and then create a profile of the energy levels for each particle
    Args:
        ax: list with the axes (obtained from plt.subplots)
        it_name: name for the particular iteration
        save: wether to save the file
        savename: Name to save the file
    """
    # Columns for the optimization
    columns = [
        "QSize", "me", "mh", "Pl", "Pt", "Eg", "offset", "FoM", "vQSize",
        "vme", "vmh", "vPl", "vPt", "vEg", "voffset"
    ]
    data = pd.read_csv(it_name, sep=" ", names=columns)
    ordered_data = data.apply(tuple, axis=1).values.reshape(ax.shape)
    # Normalized x to organize the standar information
    x = np.linspace(0, 1, 50)
    for i, line in enumerate(ordered_data):
        for j, column in enumerate(line):
            Vcb = (column[5] - 0.4) * column[6]
            Vvb = (column[5] - 0.4) * (1 - column[6])
            qd_cb = qbd.qd_results(column[0],
                                   Vcb,
                                   column[1],
                                   column[1],
                                   band="CB1")
            qd_vb = qbd.qd_results(column[0],
                                   Vvb,
                                   column[2],
                                   column[2],
                                   band="VB1")
            # Define plot limits
            ax[i, j].set_ylim(0, column[5])
            ax[i, j].spines['right'].set_visible(False)
            ax[i, j].spines['top'].set_visible(False)
            ax[i, j].spines['bottom'].set_visible(False)
            # Create the band diagram from Eg, Vcb, Vvb
            # Left/Righ sides of the bands
            ax[i, j].axhline(0, xmin=0, xmax=0.25, color="k")
            ax[i, j].axhline(column[5], xmin=0, xmax=0.25, color="k")
            ax[i, j].axhline(0, xmin=0.65, xmax=1, color="k")
            ax[i, j].axhline(column[5], xmin=0.65, xmax=1, color="k")
            # Mid Line
            ax[i, j].axhline(Vvb, xmin=0.25, xmax=0.65, color="k")
            ax[i, j].axhline(Vvb + 0.4, xmin=0.25, xmax=0.65, color="k")
            # Add vertical bariers
            ax[i, j].axvline(0.25, ymin=0, ymax=Vvb / column[5], color="k")
            ax[i, j].axvline(0.65, ymin=0, ymax=Vvb / column[5], color="k")
            ax[i, j].axvline(0.25,
                             ymin=(Vvb + 0.4) / column[5],
                             ymax=1,
                             color="k")
            ax[i, j].axvline(0.65,
                             ymin=(Vvb + 0.4) / column[5],
                             ymax=1,
                             color="k")
            ax[i, j].set_xticks(ticks=[])
            ax[i, j].tick_params(axis="y", labelsize=16)
            # Plot the energy spectrum (from -0.25 to 0.25)
            for e_cb in qd_cb.e_levels.values.flatten():
                ax[i, j].axhline(column[5] + e_cb, xmin=0.25, xmax=0.65)
            for e_vb in qd_vb.e_levels.values.flatten():
                ax[i, j].axhline(-e_vb, xmin=0.25, xmax=0.65)
    if save:
        plt.savefig(savename, transparent=True)
