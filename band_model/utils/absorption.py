""" General tools to determine absorption properties of the QD@Host
system
This module has:
    absorption_ij, interband_absorption (Calculate inter and intraband
                                    absorptions)
    opt_function: Optimization method that uses the pso algorithm
    bruggerman_dispersion: Determine the bruggerman_dispersion between 2
                            materials
"""
import numpy as np
import math
from itertools import product
import logging
from .. import qd_base_data as qbd
import pandas as pd
import scipy.integrate as sci
import scipy.constants as scc
import scipy.optimize as sco
import multiprocessing
from multiprocessing import Pool
from functools import partial
""" FuncTions to determine absorption coefficients (inter and intraband) """


def absorption_ij(energy,
                  t_energy,
                  matrix_elements,
                  gauss_dispersion=0.025,
                  n_index=2.5):
    """Gaussian absorption profile per density for an energy
        transition from an initial to a final state
    Args:
        energy (array): Base array [eV]
        t_energy (double): Transition energy (energy difference between the
                            initial and final states) [eV]
        gauss_dispersion (double): Dispersion value for the Gaussian profile
                            (default - 0.025 eV)
        matrix_elements (tuple): Matrix elements for the different
                            polarizations [nm]
        n_index (array): Refractive index array - should be the same shape
                        as the energy or single valued (default = 2)
    Returns:
        abs_ij (array): Absorption for the transition i->j [cm-1]
    """
    # Initialize necessary constants
    # C, J.s, m/s, C^2/(N.m^2) the powers cut in the fraction
    q, h, c, e0 = 1.6022, 6.626, 2.9979, 8.8542
    # Calculate the fraction responsible for the units
    constant_fraction = ((2 * np.pi**2 * q**2 * energy) /
                         (n_index * c * h * e0))
    # Determine the gaussian approximation of the delta peak
    delta = (1 / (np.sqrt(np.pi) * gauss_dispersion)) * \
        np.exp(-((energy - t_energy) / gauss_dispersion)
               ** 2)
    # Absorption (the 1e7 term moves the units to nm3cm-1)
    abs_ij = 2 * matrix_elements**2 * constant_fraction * \
        delta * 1e7
    return abs_ij


def envolute_matrix_element(qd_i, e_i, qd_f, e_f, sim_properties):
    """
    Determines the Mx, My and Mz, envolute matrix elements between
    state e_i (ni, li) in QD qd_i (class qd_results) and e_f (nf, lf)
    of QD qd_f.
    sim_properties is a tuple with (sim_size, lat_size, Eg, (Pl, Pt))

    Returns: Mx, My, Mz (averaged through all m values)
    """
    if qd_i.band == qd_f.band:
        raise Exception("Bands should be different")
    envolutes = ["L6mt", "L6mb", "L6pt", "L6pb"]
    sim_size, lat_size, Eg, P = sim_properties
    ni, li = e_i
    nf, lf = e_f
    Mx_array = []
    My_array = []
    Mz_array = []
    # Define the iterator from the input values
    if li != 0 and lf != 0:
        iterator = product(range(-li, li + 1), range(-lf, lf + 1))
    elif li == 0 and lf != 0:
        iterator = product([0], range(-lf, lf + 1))
    elif li != 0 and lf == 0:
        iterator = product(range(-li, li + 1), [0])
    else:
        iterator = zip([0], [0])
    # Loop through all possible m value combinations
    for m_i_i, m_f_i in iterator:
        # Calculate the matrix element for transition between 2 elements
        # (for a specific m value)
        # This element is determined from the sum of the elements
        # from the 4 envolutes
        Mx, My, Mz = 0, 0, 0
        for envolute in envolutes:
            XX, YY, ZZ, env_i, _ = qd_i.norm_envolute(sim_size, lat_size,
                                                      envolute, li, m_i_i, ni,
                                                      Eg, P)
            _, _, _, env_f, _ = qd_f.norm_envolute(sim_size, lat_size,
                                                   envolute, lf, m_f_i, nf, Eg,
                                                   P)
            # Function integration (Simple Rienman integration)
            Mx += np.sum(env_i * XX * np.conjugate(env_f))
            My += np.sum(env_i * YY * np.conjugate(env_f))
            Mz += np.sum(env_i * ZZ * np.conjugate(env_f))
        # Determine the matrix element from n1, l1, m1 to n2, l2, m2
        Mx_array.append(np.abs(Mx) * lat_size**3)
        My_array.append(np.abs(My) * lat_size**3)
        Mz_array.append(np.abs(Mz) * lat_size**3)
    Mx_array = np.array(Mx_array)
    My_array = np.array(My_array)
    Mz_array = np.array(Mz_array)
    # Create a mask, for calculating the average where all elements bellow
    # Use the threshold of 1e-3 are considered to be values too small
    mx_zeros = Mx_array > 1e-3
    my_zeros = My_array > 1e-3
    mz_zeros = Mz_array > 1e-3
    # Check if the mask does not sum to 0 and determine the average
    if True in mx_zeros:
        Mx_final = np.average(Mx_array, weights=mx_zeros)
    else:
        Mx_final = 0
    if True in my_zeros:
        My_final = np.average(My_array, weights=my_zeros)
    else:
        My_final = 0
    if True in mz_zeros:
        Mz_final = np.average(Mz_array, weights=mz_zeros)
    else:
        Mz_final = 0
    return Mx_final, My_final, Mz_final


def interband_transition_elements(qd1, qd2, sim_properties, count=10000):
    """
    Calculates all the interband transition elements between qd1 and qd2
        sim_properties: (sim_size, lat_size, Eg, (Pl, Pt))
        count (int): Number of transitions to calculate
    Returns:
        transition_list (list): List with the transition elements
                                (Transition energy, (Mx, My, Mz))
    """
    if qd1.e_levels.values == [] or qd2.e_levels.values == []:
        raise Exception("One of the Quantum Dots has no energy levels")
    # Create a list with all the present (n, l) level combinations for each qd
    qd1_e_values = qd1.e_levels.values
    qd1_levels = list(
        product(range(qd1_e_values.shape[0]), range(qd1_e_values.shape[1])))
    qd2_e_values = qd2.e_levels.values
    qd2_levels = list(
        product(range(qd2_e_values.shape[0]), range(qd2_e_values.shape[1])))
    # Cicle through all the (n, l) combination of both qds
    # coupled with each particular energy level
    # 1st determine all the transition energies
    trn_energy = np.array([
        sim_properties[2] + qd1_e + qd2_e for qd1_e, qd2_e in product(
            qd1_e_values.flatten(), qd2_e_values.flatten())
    ])
    # 2nd Create an array with all the iterable information
    # (trn_energy, (level_i, level_f)) - trn_energy is then used to sort
    iter_dtype = [("trn_energy", float), ("levels", list)]
    iter_data = np.array(list(zip(trn_energy, product(qd1_levels,
                                                      qd2_levels))),
                         dtype=iter_dtype)
    iter_data = np.sort(iter_data, order="trn_energy")
    # Create the list with all the transitions
    transition_list = []
    for trn_energy_i, (qd1_level, qd2_level) in iter_data:
        if math.isnan(trn_energy_i):
            break
        transition_list.append(
            (trn_energy_i,
             envolute_matrix_element(qd1, qd1_level, qd2, qd2_level,
                                     sim_properties)))
        # Limit the calculation to a certain number of transitions
        if len(transition_list) >= count:
            break
    logging.debug(f"{transition_list=}")
    return transition_list


def avg_trn_elements(qd1_prop, qd2_prop, sim_properties, trn=0):
    """
    Function to determine the averaged transition element for a particular set
    of QD Properties
    This function only calculates the values for a certain transition as well
    as provides all the information for that particular transition
    """
    # Unpack parameters for each QD
    qd1_size, Vvb, mh, mh = qd1_prop
    logging.debug(f"{qd1_size=}\t{Vvb=}\t{mh=}\t{mh=}")
    qd2_size, Vcb, me, me = qd2_prop
    logging.debug(f"{qd2_size=}\t{Vcb=}\t{me=}\t{me=}")
    bi = ["VB1", "VB2"]
    bf = ["CB1", "CB2"]
    avg_elements = []
    trn_elements = []
    trn_energy = []
    # Create a list with the transition elements for each band transition
    for b_i, b_f in product(bi, bf):
        logging.debug(f"{b_i=} {b_f=}")
        qd1 = qbd.qd_results(qd1_size, Vvb, mh, mh, b_i)
        qd2 = qbd.qd_results(qd2_size, Vcb, me, me, b_f)
        transition = interband_transition_elements(qd1,
                                                   qd2,
                                                   sim_properties,
                                                   count=trn + 1)
        logging.debug(f"{transition=}")
        # Average the results of each band transition
        if len(transition) > trn:
            band_trn = np.pi / 4 * transition[trn][1][
                0]**2 + np.pi / 4 * transition[trn][1][
                    1]**2 + np.pi / 2 * transition[trn][1][2]**2
            trn_elements.append([
                transition[trn][1][0], transition[trn][1][1],
                transition[trn][1][2]
            ])
            avg_elements.append(band_trn)
        else:
            logging.info(f"{trn=} is outside the number of valid transitions")
            trn_elements.append(np.NaN)
            avg_elements.append(np.NaN)
            break
    # Also export all the transition energies
    trn_energy = [trn[0] for trn in transition]
    logging.debug(f"{trn_energy = }")
    logging.debug(f"{avg_elements = }")
    avg_elements = np.array(avg_elements)
    avg_trn_final = np.average(avg_elements)
    # else:
    #     avg_trn_final = 0
    return avg_trn_final, avg_elements, trn_elements, trn_energy


def all_avg_trn_elements(qd1_prop, qd2_prop, sim_properties):
    """
    Function to determine all the avg transition elements for 2 qds
    Args:
        qd1_prop/qd2_prop (tuple): (qd_size, V, me, me)
        sim_properties (tuple): (s_size, l_size, Eg, (Pl, Pt))
    Returns:
        - e_trn (list): List with all the transition energies
        - band_elements (list): List with all the averaged transition elements
    """
    # Unpack parameters for each QD
    qd1_size, Vvb, mh, mh = qd1_prop
    logging.debug(f"{qd1_size=}\t{Vvb=}\t{mh=}\t{mh=}")
    qd2_size, Vcb, me, me = qd2_prop
    logging.debug(f"{qd2_size=}\t{Vcb=}\t{me=}\t{me=}")
    bi = ["VB1", "VB2"]
    bf = ["CB1", "CB2"]
    band_elements = []
    transition = []
    # Create a list with the transition elements for each band transition
    for b_i, b_f in product(bi, bf):
        logging.debug(f"{b_i=} {b_f=}")
        qd1 = qbd.qd_results(qd1_size, Vvb, mh, mh, b_i)
        qd2 = qbd.qd_results(qd2_size, Vcb, me, me, b_f)
        # List with all the transitions
        transition = interband_transition_elements(qd1, qd2, sim_properties)
        logging.debug(f"{transition=}")
        # Average the results of each band transition and for each transition
        band_trn = [
            np.pi / 4 * trn[1][0]**2 + np.pi / 4 * trn[1][1]**2 +
            np.pi / 2 * trn[1][2]**2 for trn in transition
        ]
        band_elements.append(band_trn)
    # Get all the transition energies
    e_trn = [trn[0] for trn in transition]
    logging.debug(f"{e_trn = }")
    band_elements = np.array(band_elements)
    logging.debug(f"{band_elements=}")
    # Check if there are transitions
    if band_elements.shape[1] < 1:
        logging.debug("No transition determined for this system")
        return np.nan, np.nan
    avg_trn_final = np.average(band_elements, axis=0)
    return e_trn, avg_trn_final


def interband_absorption(e_array,
                         qd_cb,
                         qd_vb,
                         sim_properties,
                         n_index=2.5,
                         peak_dispersion=0.025):
    """ Calculate the interdand absorption given 2 qds and the simulation conditions
    Args:
        e_array (array): x-array with the energy values
        qd_cb/qd_vb (tuple): (q_size, V, m, m)
        sim_properties (tuple): (Ssize, LSize, Eg, (Pl, Pt))
        n_index (float/array): Array with the refractive index values
        peak_dispersion (float): Dispersion of the absorption peal
    Returns:
        absorption (DataFrame) with:
            absorption_per_density (nm3/cm): final total absorption
                                             coefficient per density
            absorption_per_peak: array with absorption coefficient
                                 for each peak
    
    """
    # Initialize necessary constants
    # C, J.s, m/s, C^2/(N.m^2) the powers cut in the fraction
    q, h, c, e0 = 1.6022, 6.626, 2.9979, 8.8542
    # Calculate the fraction responsible for the units
    constant_fraction = ((2 * np.pi**2 * q**2 * e_array) /
                         (n_index * c * h * e0))
    trn_energies, trn_elements = all_avg_trn_elements(qd_vb, qd_cb,
                                                      sim_properties)
    logging.debug(f"{trn_energies=}\n{trn_elements}")
    results = pd.DataFrame()
    results["Energy"] = e_array
    if np.all(np.isnan(trn_energies)):
        logging.warning(
            f"No transitions for qd: {qd_cb=} {qd_vb=} {sim_properties=}")
        results["Total"] = np.nan
        return results
    results["Total"] = np.zeros_like(e_array)
    for (index, trn_energy), trn_element in zip(enumerate(trn_energies),
                                                trn_elements):
        logging.debug(f"Transition: {trn_energy = } and {trn_element = }")
        # Determine the gaussian approximation of the delta peak
        delta_peak = (1 / (np.sqrt(np.pi) * peak_dispersion)) * np.exp(-(
            (e_array - trn_energy) / peak_dispersion)**2)
        delta_peak *= 2 * trn_element * constant_fraction * 1e7
        results[f"T{index}({trn_energy:.3f})"] = delta_peak
        # Absorption (the 1e7 term moves the units to nm3cm-1)
        results["Total"] += delta_peak
    return results


""" Functions for FoM and optimization function for the pso algorithm """

def opt_data(qd_size_i,
             Vcb_i,
             Vvb_i,
             me_i,
             mh_i,
             Pl_i,
             Pt_i,
             Eg_i,
             offset_i,
             energy=np.linspace(2, 3, 200),
             lat_size=0.8,
             sim_size=25):
    """
    Determine the basic information used for the calculation of the FoM
    """
    # Assuming the energy variation
    logging.info(
        f"Single FoM:{qd_size_i=:.2f}  {me_i=:.2f}  {mh_i=:.2f}  {Pl_i=:.2g}" +
        f"  {Pt_i=:.2g}  {Eg_i=:.2f}  {offset_i=:.2f}" +
        f"  {Vcb_i=:.2f}  {Vvb_i=:.2f}")
    # Calculate the energy level distance
    qd_data = qbd.qd_results(qd_size_i, Vcb_i, me_i, me_i, "CB1")
    ideal_ib = -0.9
    # Calculate normalized value (by ideal_ib)
    levels_ib = np.abs((qd_data.e_levels.values - ideal_ib)/ideal_ib)
    if levels_ib.size == 0:
        similarity = np.nan 
    else:
        similarity = np.nanmin(levels_ib)
    sim_properties = (sim_size, lat_size, Eg_i, (Pl_i, Pt_i))
    data = interband_absorption(energy, (qd_size_i, Vcb_i, me_i, me_i),
                                (qd_size_i, Vvb_i, mh_i, mh_i), sim_properties)
    Ntrn = data.shape[1] - 2
    # The 1e2 converts the wavelength from m to cm
    int_abs = -sci.simpson(data["Total"],
                       (scc.h * scc.c) / (data["Energy"] * scc.e) * 1e2)
    logging.info(f"{int_abs=}::{qd_size_i=}::{similarity=}")
    return int_abs, similarity, Ntrn


def FoM_similatiry_size(qd_size_i,
                        Vcb_i,
                        Vvb_i,
                        me_i,
                        mh_i,
                        Pl_i,
                        Pt_i,
                        Eg_i,
                        offset_i,
                        energy=np.linspace(2, 3, 200),
                        lat_size=0.8,
                        sim_size=25):
    """
    Calculate the FoM for a single combination of parameters
    This FoM Considers:
        - Number of energy levels (smaller the better)
        - Energy Level close to 0.9 in the CB
        - Smaller QD size
        - Better Absorption
    """
    """ Calculate the FoM for a single combination of parameters """
    sim_properties = (sim_size, lat_size, Eg_i, (Pl_i, Pt_i))
    # Assuming the energy variation
    logging.info(
        f"Single FoM:{qd_size_i=:.2f}  {me_i=:.2f}  {mh_i=:.2f}  {Pl_i=:.2g}" +
        f"  {Pt_i=:.2g}  {Eg_i=:.2f}  {offset_i=:.2f}" +
        f"  {Vcb_i=:.2f}  {Vvb_i=:.2f}")
    # Calculate the energy level distance
    qd_data = qbd.qd_results(qd_size_i, Vcb_i, me_i, me_i, "CB1")
    ideal_ib = -0.9
    levels_ib = qd_data.e_levels.values - ideal_ib
    if levels_ib.size == 0:
        similarity = np.nan 
    else:
        similarity = np.abs(np.nanmin(levels_ib))
    data = interband_absorption(energy, (qd_size_i, Vcb_i, me_i, me_i),
                                (qd_size_i, Vvb_i, mh_i, mh_i), sim_properties)
    # The 1e2 serves to convert the wavelength from m to cm
    FoM = -sci.simpson(data["Total"],
                       (scc.h * scc.c) / (data["Energy"] * scc.e) * 1e2)
    Nlevels = data.shape[1] - 2
    res = np.cbrt(FoM) / (qd_size_i * Nlevels * similarity)
    logging.debug(f"{FoM=}::{qd_size_i=}::{similarity=}\n{res=}")
    return res


def FoM_int_size_nlevels(qd_size_i,
                         Vcb_i,
                         Vvb_i,
                         me_i,
                         mh_i,
                         Pl_i,
                         Pt_i,
                         Eg_i,
                         offset_i,
                         energy=np.linspace(2, 3, 200),
                         lat_size=0.8,
                         sim_size=25):
    """ Calculate the FoM for a single combination of parameters """
    sim_properties = (sim_size, lat_size, Eg_i, (Pl_i, Pt_i))
    # Assuming the energy variation
    logging.info(
        f"Single FoM:{qd_size_i=:.2f}  {me_i=:.2f}  {mh_i=:.2f}  {Pl_i=:.2g}" +
        f"  {Pt_i=:.2g}  {Eg_i=:.2f}  {offset_i=:.2f}" +
        f"  {Vcb_i=:.2f}  {Vvb_i=:.2f}")
    data = interband_absorption(energy, (qd_size_i, Vcb_i, me_i, me_i),
                                (qd_size_i, Vvb_i, mh_i, mh_i), sim_properties)
    # The 1e2 serves to convert the wavelength from m to cm
    FoM = -sci.simpson(data["Total"],
                       (scc.h * scc.c) / (data["Energy"] * scc.e) * 1e2)
    Nlevels = data.shape[1] - 2
    logging.debug(f"FoM={FoM/(qd_size_i**3 * Nlevels)}")
    return FoM / (Nlevels)


def _single_FoM(qd_size_i,
                Vcb_i,
                Vvb_i,
                me_i,
                mh_i,
                Pl_i,
                Pt_i,
                Eg_i,
                offset_i,
                energy=np.linspace(2, 3, 200),
                lat_size=0.8,
                sim_size=25):
    """ Calculate the FoM for a single combination of parameters """
    sim_properties = (sim_size, lat_size, Eg_i, (Pl_i, Pt_i))
    # Assuming the energy variation
    logging.info(
        f"Single FoM:{qd_size_i=:.2f}  {me_i=:.2f}  {mh_i=:.2f}  {Pl_i=:.2g}" +
        f"  {Pt_i=:.2g}  {Eg_i=:.2f}  {offset_i=:.2f}" +
        f"  {Vcb_i=:.2f}  {Vvb_i=:.2f}")
    data = interband_absorption(energy, (qd_size_i, Vcb_i, me_i, me_i),
                                (qd_size_i, Vvb_i, mh_i, mh_i), sim_properties)
    # The 1e2 serves to convert the wavelength from m to cm
    FoM = -sci.simpson(data["Total"],
                       (scc.h * scc.c) / (data["Energy"] * scc.e) * 1e2)
    return FoM / qd_size_i**3


def opt_function_general(func,
                         qd_size,
                         me,
                         mh,
                         Pl,
                         Pt,
                         Eg,
                         offset,
                         energy=np.linspace(2, 3, 200),
                         lat_size=0.8,
                         sim_size=25):
    """
    Generic representation of the optimization function
    Args:
        func: used to optimize (defined similar to _single_FoM
        qd_properties: (qd_size, me, m, Pl, Pt, Eg, offset)
        sim_args: (energy, lat_size, sim_size) passed to func
    """
    logging.info("Running Optimization Function....")
    Vcb = (Eg - 0.4) * offset
    Vvb = (Eg - 0.4) - Vcb
    # Create a partial funcion with some parameters constant
    __func = partial(func, energy=energy, lat_size=lat_size, sim_size=sim_size)
    with Pool(multiprocessing.cpu_count() - 2) as p:
        res = p.starmap(__func,
                        zip(qd_size, Vcb, Vvb, me, mh, Pl, Pt, Eg, offset))
    # Convert the NaN to 0 (as we want to maximize the results)
    res = np.array(res)
    res[np.isnan(res)] = 0
    return res


def opt_function(qd_size,
                 me,
                 mh,
                 Pl,
                 Pt,
                 Eg,
                 offset,
                 energy=np.linspace(2, 3, 200),
                 lat_size=0.8,
                 sim_size=25):
    """
    Optimization function for the particle swarm algorithm
    The input parameters are an array with the properties, and the output is
    the array with the results
    """
    logging.info("Running Optimization Function....")
    Vcb = (Eg - 0.4) * offset
    Vvb = (Eg - 0.4) - Vcb
    # Create a partial funcion with some parameters constant
    __single_FoM = partial(_single_FoM,
                           energy=energy,
                           lat_size=lat_size,
                           sim_size=sim_size)
    with Pool(multiprocessing.cpu_count() - 2) as p:
        res = p.starmap(__single_FoM,
                        zip(qd_size, Vcb, Vvb, me, mh, Pl, Pt, Eg, offset))
    # Convert the NaN to 0 (as we want to maximize the results)
    res = np.array(res)
    res[np.isnan(res)] = 0
    return res


""" Bruggerman equation to determine the combined refractive index """


def _bruggerman(n_eff, n_qd: complex, n_pvk: complex, p_qd: float):
    """ Bruggerman equation:
    n_eff is similar to x in a function
    The point where the returned y=0 is the n_eff value
    """
    y = p_qd * (n_qd - n_eff) / (n_qd + 2 * n_eff) + (1 - p_qd) * (
        n_pvk - n_eff) / (n_pvk + 2 * n_eff)
    return y


def bruggerman_dispersion(n1, n2, p):
    """
    Calculate the effective medium dispersion between two materials 1 and 2
    Args:
        n1, n2 (complex arrays): refractive indices for each material
        p (float): fraction
    Returns:
        n_eff (complex array): effective medium
    """
    n_eff = np.ones_like(n1, dtype=np.complex128)
    for (index, n1_i), n2_i in zip(enumerate(n1), n2):
        n_eff[index] = sco.newton(_bruggerman, (n1_i + n2_i) / 2,
                                  args=(n1_i, n2_i, p))
    return n_eff
