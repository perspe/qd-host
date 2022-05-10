""" Module with several functions to simplify some calculations/plots
that need to be done several times """
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
import os
from matplotlib import cm
import seaborn as sns
from . import absorption as ab
from .. import qd_base_data as qbd


def mc_random(qd_size,
              me,
              mh,
              Pl,
              Pt,
              Eg,
              offset,
              tol=0.1,
              n_rnd=500,
              **kwargs):
    """ Calculate n_rnd simulations tol% around the given values
    Args: Basically the optimization variables
    """
    logging.info(f"Random MC...{tol=} {n_rnd=}")
    me_rnd = np.random.uniform((1 - tol) * me, (1 + tol) * me, size=(n_rnd))
    mh_rnd = np.random.uniform((1 - tol) * mh, (1 + tol) * mh, size=(n_rnd))
    qd_size_rnd = np.random.uniform((1 - tol) * qd_size, (1 + tol) * qd_size,
                                    size=(n_rnd))
    Pl_rnd = np.random.uniform((1 - tol) * Pl, (1 + tol) * Pl, size=(n_rnd))
    Pt_rnd = np.random.uniform((1 - tol) * Pt, (1 + tol) * Pt, size=(n_rnd))
    Eg_rnd = np.random.uniform((1 - tol) * Eg, (1 + tol) * Eg, size=(n_rnd))
    offset_rnd = np.random.uniform((1 - tol) * offset, (1 + tol) * offset,
                                   size=(n_rnd))
    logging.debug(me_rnd, mh_rnd, qd_size_rnd, Pl_rnd, Pt_rnd, Eg_rnd,
                  offset_rnd)

    # Run all the random iterations
    mc_res = ab.opt_function(qd_size_rnd, me_rnd, mh_rnd, Pl_rnd, Pt_rnd,
                             Eg_rnd, offset_rnd, **kwargs)
    return mc_res


def mc_pso_best_param(best_it,
                      n_rnd=1000,
                      tol=0.05,
                      save=False,
                      savename="best_particles.txt",
                      **const_args):
    """ Take the best result of the optimization and do a mc analysis around
    the maximum value for each parameter """
    # Import the respective data from the given iteration
    columns = [
        "qd_size", "me", "mh", "Pl", "Pt", "Eg", "offset", "FoM", "vQSize",
        "vme", "vmh", "vPl", "vPt", "vEg", "voffset"
    ]
    data = pd.read_csv(best_it, sep=" ", names=columns)
    data.sort_values(by="FoM",
                     ascending=False,
                     ignore_index=True,
                     inplace=True)
    # Create a dictionary with the list of best parameters
    best_res = data.loc[0][:7].to_dict()
    best_res_array = {
        key: value * np.ones(n_rnd)
        for key, value in best_res.items()
    }
    # Copy the best dictionary and change each parameter for the random sweep at a time
    results = []
    for key in best_res_array.keys():
        logging.info(f"Running... {key=}")
        sim_array = best_res_array.copy()
        sim_array[key] = np.random.uniform((1 - tol) * best_res[key],
                                           (1 + tol) * best_res[key],
                                           size=(n_rnd))
        sim_array.update(const_args)
        results.append(ab.opt_function(**sim_array))
    if save:
        np.savetxt(savename,
                   np.array(results).T,
                   comments=" ".join(list(best_res_array.keys())))

    return np.array(results)


def pso_best_abs(it_name,
                 energy=np.linspace(0.5, 4, 600),
                 save=False,
                 savename="PSO_Best_2.svg"):
    """ Plot the best optimization result """
    # Get all the parameters from the best optimization result
    columns = [
        "QSize", "me", "mh", "Pl", "Pt", "Eg", "offset", "FoM", "vQSize",
        "vme", "vmh", "vPl", "vPt", "vEg", "voffset"
    ]
    data = pd.read_csv(it_name, sep=" ", names=columns)
    data.sort_values(by="FoM",
                     ascending=False,
                     ignore_index=True,
                     inplace=True)
    # Unpack into the several variables
    qsize, me, mh, Pl, Pt, Eg, offset, *_ = tuple(data.loc[0])
    # Contruct the best qds and properties
    sim_setup = (25, 0.65, Eg, (Pl, Pt))
    qd1_prop = (qsize, (Eg - 0.4) * offset, me, me)
    qd2_prop = (qsize, (Eg - 0.4) * (1 - offset), mh, mh)
    # Calculate the best properties
    best_abs = ab.interband_absorption(energy, qd1_prop, qd2_prop, sim_setup)
    logging.info(best_abs)
    # Plot the best optimization absorption profile
    best_abs.plot("Energy")
    plt.xlabel("Energy (eV)")
    plt.ylabel("Absorption coefficient")
    plt.tick_params(axis='both')
    for index, (name, value) in enumerate(zip(columns, data.loc[0])):
        if index > 7:
            break
        plt.annotate(f"$\\bf{name}$={value:.3g}$\pm${value*0.05:.2f}",
                     (0.02, 0.9 - index / 14),
                     xycoords="axes fraction")
    if save:
        plt.savefig(savename, transparent=True)


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
                ax[i, k].set_ylabel("Absorption Coef (nm$^3$/cm)")
            if i == 2:
                ax[i, k].set_xlabel("Energy (eV)")
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
            ax[i, j].tick_params(axis="y")
            # Plot the energy spectrum (from -0.25 to 0.25)
            for e_cb in qd_cb.e_levels.values.flatten():
                ax[i, j].axhline(column[5] + e_cb, xmin=0.25, xmax=0.65)
            for e_vb in qd_vb.e_levels.values.flatten():
                ax[i, j].axhline(-e_vb, xmin=0.25, xmax=0.65)
    if save:
        plt.savefig(savename, transparent=True)


def pso_summary(results_folder,
                columns,
                iterations,
                n_particles,
                export=False,
                **plot_kwargs):
    """ Make a summary of all the optimization results.
    Args:
        results folder: folder where the results where stored
        columns: a list with the column names from the optimization
                (should have FoM for the FoM)
        iterations: the number of iterations
        export: whether to export data or not (it export as svg)
        **plot_kwargs: mostly to pass the figure size of the different plots
    It will plot:
        Parameter histogram
        Particle movement for each parameter
        FOM variation for all particles
        The first two but for the velocities
    """
    if not os.path.isdir(results_folder):
        raise Exception("Provided path is not a directory!")
    if len(columns) % 2 != 1:
        raise Exception("Invalid number of columns....")
    # Import data into a pandas DF
    import_data = []
    for i in range(iterations):
        import_data.append(
            pd.read_csv(f"{results_folder}/results_{i}",
                        sep=" ",
                        names=columns))
    data = pd.concat(import_data, keys=list(range(iterations)))
    n_params = int((len(columns) - 1) / 2)
    colors = cm.get_cmap("jet", lut=n_params)
    # Plot the histograms
    _, ax = plt.subplots(1, n_params, **plot_kwargs)
    plt.subplots_adjust(wspace=0.2, left=0.05, right=0.95)
    # Plot the histograms
    for index, variable in enumerate(columns[:n_params]):
        sns.histplot(data=data,
                     y=variable,
                     color=colors(index),
                     kde=True,
                     bins=50,
                     ax=ax[index])
        ax[index].tick_params(axis="y", rotation=90)
        ax[index].set_ylabel("")
        ax[index].set_title(variable)
    ax[0].set_ylabel("Variable")
    if export:
        plt.savefig(f"{results_folder}/pso_property_hist.svg",
                    transparent=True)

    # Plot values per iteration
    _, ax = plt.subplots(1, n_params, **plot_kwargs)
    plt.subplots_adjust(wspace=0.2, left=0.05, right=0.95)
    # Plot the histograms
    for index, variable in enumerate(columns[:n_params]):
        for it in range(iterations):
            ax[index].scatter(np.zeros(n_particles) + it,
                              data.loc[it, variable],
                              color=colors(index),
                              s=3)
        ax[index].tick_params(axis="y", rotation=90)
        ax[index].set_ylabel("")
        ax[index].set_title(variable)
    ax[0].set_ylabel("Variable")
    if export:
        plt.savefig(f"{results_folder}/pso_property_iter.svg",
                    transparent=True)

    # Plot the FoM for the various iterations
    _, ax = plt.subplots(1, 1)
    for it in range(iterations):
        ax.scatter(np.zeros(n_particles) + it,
                   data.loc[it, "FoM"],
                   s=3)
    ax.set_ylabel("FoM")
    ax.set_xlabel("Iteration")
    if export:
        plt.savefig(f"{results_folder}/pso_FoM.svg", transparent=True)

    # Plot for the velocities
    _, ax = plt.subplots(1, n_params, **plot_kwargs)
    for index, variable in enumerate(columns[8:]):
        _ = sns.histplot(data=data,
                         y=variable,
                         color=colors(index),
                         kde=True,
                         bins=50,
                         ax=ax[index])
        ax[index].tick_params(axis="y", rotation=90)
        ax[index].set_ylabel("")
        ax[index].set_title(variable)
    ax[0].set_ylabel("Variable")
    if export:
        plt.savefig(f"{results_folder}/pso_property_vhist.svg",
                    transparent=True)

    _, ax = plt.subplots(1, n_params, **plot_kwargs)
    plt.subplots_adjust(wspace=0.2, left=0.05, right=0.95)
    for index, variable in enumerate(columns[n_params + 1:]):
        for it in range(iterations):
            ax[index].scatter(np.zeros(n_particles) + it,
                              data.loc[it, variable],
                              color=colors(index),
                              s=3)
        ax[index].tick_params(axis="y", rotation=90)
        ax[index].set_ylabel("")
        ax[index].set_title(variable)
    ax[0].set_ylabel("Variable")
    if export:
        plt.savefig(f"{results_folder}/pso_property_viter.svg",
                    transparent=True)
