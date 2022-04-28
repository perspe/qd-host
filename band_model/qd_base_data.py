"""
Base module to calculate the QD properties
Contains the main QD class (qd_results), and 2 helper functions
(spherical hankel and spherical bound_states)
"""
import numpy as np
import pandas as pd
# Necessary scipy functions
from scipy import special
from scipy.fft import fftn, ifftn
from scipy.integrate import quad
from sympy.physics.quantum.cg import CG

from .cython_utils.zeros_calc import zeros_f
from .utils import envelope_functions as env_f
# from .utils import absorption as absp


def spherical_hankel(n, x, kind, **kwargs):
    """Use the Hankel functions to calculate the spherical hankel functions
    Args:
        n (int): order of the spherical hankel function
        x (array): array with x data
        kind (int): kind of the hankel function (1 or 2)
    **kwargs:
        derivative (bool): if true returns the derivative instead
    Return:
        sph_hankel (array): Spherical hankel or derivative of kind k
    """
    if kind == 1:  # Calculation for 1st kind
        if kwargs.get("derivative"):  # Check derivative argument
            return np.sqrt(
                (np.pi /
                 (2 * x))) * (special.hankel1(n - 1 / 2, x) -
                              ((n + 1) / x) * special.hankel1(n + 1 / 2, x))
        else:
            return np.sqrt((np.pi / (2 * x))) * special.hankel1(n + 1 / 2, x)
    else:  # Calculation for 2nd kind
        if kwargs.get("derivative"):  # Check derivative argument
            return np.sqrt(
                (np.pi /
                 (2 * x))) * (special.hankel2(n - 1 / 2, x) -
                              ((n + 1) / x) * special.hankel2(n + 1 / 2, x))
        else:
            return np.sqrt((np.pi / (2 * x))) * special.hankel2(n + 1 / 2, x)


def spherical_bound_states(energy, m1, m2, a, V0, ang_mom):
    """Calculate the values to solve the eigenvalue condition
    For l = 0 values with the analytical solution
    For l > 0 values with the simplified eigenvalue condition
    Args:
        energy (array): Array with the energy values [eV]
        m1, m2 (double): Effective masses inside
                        and outside the well [m = m*.m_e]
        a (double): Radius of the well [nm]
        V0 (double): Potential barrier of the well [eV]
        l (int): Order of the bessel function
    Returns:
        f_E: Function result
    """
    # Normalization factors
    m1, m2, h_bar, V0 = m1 * 9.1094, m2 * 9.1094, 1.05457, V0 * 1.6022
    energy = energy * 1.6022
    if ang_mom == 0:  # (l = 0) Solve with the analytical solution
        arg_cot = np.sqrt(2.0 * m1 * (energy + V0)) * (a / h_bar)
        f_E = 1.0 / np.tan(arg_cot) + np.sqrt(-(m2 / m1) * (energy /
                                                            (energy + V0)))
    else:  # (l > 0 ) Solve with the simplified condition
        energy_j = np.sqrt(2.0 * m1 * (energy + V0)) * (a / h_bar)
        energy_h = np.sqrt(-2.0 * m2 * energy) * (a / h_bar) * 1j
        j_term = special.spherical_jn(
            ang_mom - 1, energy_j) / special.spherical_jn(ang_mom, energy_j)
        h_term = np.sqrt(-(m2 / m1) * (energy / (energy + V0))) * np.real(
            1j * (spherical_hankel(ang_mom - 1, energy_h, 1) /
                  spherical_hankel(ang_mom, energy_h, 1)))
        f_E = np.add(j_term, -h_term)
    # if export:
    #     export_data = np.c_[energy, f_E]
    #     np.savetxt(
    #         f"bound_states_l{ang_mom}_({m1:.2f}_{m2:.2f}_{a:.2f}_{V0:.2f}).txt",
    #         export_data)
    return f_E


####################################################
# Quantum Dot class
####################################################


class qd_results():
    """Class to calculate energy levels,  wavefunctions and matrix elements of
    a spherical qd
    Methods:
        __init__: Initializes the variables (a, V0, m1, m2, band) necessary
            for the calculations
        energy_levels (Automatic initialization): Creates a  dataframe
            attribute (e_levels) with the energy levels
        norm_constants (Automatic initialization): Creates a dataframe
            attribute (k_in, k_out, norm_A, norm_B) with k values and
            normalization constants
        norm_radial_wavefunction: Calculates the radial normalized probability
            density in a specific range (default = 2*a)
        norm_wavefunction: Probability density for a given l (orbital angular
            momentum), n (energy level index) values
        norm_envolute: Return a specific envolute (L6mt, L6mb, L6pt, L6pb),
            for the QD in the band specified initialy
        rad_matrix_element: Radial matrix element between an initial (li, ni)
            and final (lf, nf) states
        ang_matrix_element: Angular matrix components element between an
            initial (li, mi) and final (lf, mf) states
        all_ang_matrix_elements: Calculates all the angular matrix elements
            between an initial and final l state of the QD
    """
    def __init__(self, a, V0, m1, m2, band, mode='advanced'):
        """Initialization function
        Args:
            a (float): Size of the qd [nm]
            V0 (float): Energy of the potential barrier [eV]
            m1 and m2 (float): Effective masses inside and outside
                the qd [m = m*.m_e]
            band (str): QD band - CB1, CB2, VB1, VB2
            mode ('advanced'/'simple'): mode to calculate results
                    - advanced (default) :  calculates everything
                    - simple : calculates just for the first energy level
        """
        self.a = a
        self.V0 = V0
        self.m1 = m1
        self.m2 = m2
        allowed_bands = ["CB1", "CB2", "VB1", "VB2"]
        if band not in allowed_bands:
            raise Exception("Unkown provided band! allowed (" +
                            ", ".join(allowed_bands) + ")")
        self.band = band
        if mode == 'advanced':  # Calculate everything
            self.energy_levels()
            self.norm_constants()
        else:
            # e values following geometrical progression
            e_geom = -(self.V0 + 0.001) + np.geomspace(
                0.001, self.V0, 10000, endpoint=True)
            ang_mom = mode[1]
            self.e_levels = zeros_f(spherical_bound_states,
                                    e_geom,
                                    args=(self.m1, self.m2, self.a, self.V0,
                                          ang_mom))

    def energy_levels(self):
        """Calculates the energy eigenvalue (is immediately initialized when
            the class is called)
        """
        e_geom = -(self.V0 + 0.001) + np.geomspace(
            0.001, self.V0, 10000, endpoint=True)
        zeros = dict()
        l_ind = 0
        # Cycle to calculate all zeros
        while True:
            zeros[f"l{l_ind}"] = zeros_f(spherical_bound_states,
                                         e_geom,
                                         args=(self.m1, self.m2, self.a,
                                               self.V0, l_ind))
            if not zeros[f"l{l_ind}"]:
                del zeros[f"l{l_ind}"]
                break
            l_ind += 1
        self.e_levels = pd.DataFrame.from_dict(zeros,
                                               orient='index').transpose()
        # Filter all values bellow 0.01
        self.e_levels[self.e_levels > -0.07] = np.nan

    def norm_constants(self):
        """Calculate the k_in,  k_out and normalization constants
        (is immediately initialized when the class is called)
        """
        # Change units necessary for calculations
        m_1, m_2, h, = self.m1 * 9.1094, self.m2 * 9.1094, 1.05457
        V_0 = self.V0 * 1.6022
        # Calculate k_in and k_out for all energy eigenvalues
        self.k_in = pd.DataFrame(
            np.sqrt(2 * m_1 * (self.e_levels.values * 1.6022 + V_0)) / h,
            columns=self.e_levels.columns)
        self.k_out = pd.DataFrame(
            np.sqrt(-2 * m_2 * self.e_levels.values * 1.6022) / h,
            columns=self.e_levels.columns)
        # Initialize dataframe for the constant values and the wavefunctions
        self.norm_A = pd.DataFrame()
        self.norm_B = pd.DataFrame()
        # Calculate normalization constants for each l value
        for li in self.k_in.columns:
            j_li_ka = np.absolute(
                special.spherical_jn(int(li[1:]),
                                     self.k_in[li].values * self.a))**2
            h_li_ka = np.absolute(
                spherical_hankel(int(li[1:]),
                                 1j * self.k_out[li].values * self.a,
                                 kind=1))**2
            a_j_li = np.array([
                quad(
                    lambda x: special.spherical_jn(int(li[1:]), k_in_i * x)**2
                    * x**2, 0, self.a)[0] if k_in_i != np.nan else np.nan
                for k_in_i in self.k_in[li].values
            ])
            b_h_li = np.array([
                quad(
                    lambda x: np.absolute(
                        spherical_hankel(int(li[1:]), 1j * k_out_i * x, kind=1)
                    )**2 * x**2, self.a, np.inf)[0]
                if k_out_i != np.nan else np.nan
                for k_out_i in self.k_out[li].values
            ])
            self.norm_A[li] = h_li_ka / (a_j_li * h_li_ka + b_h_li * j_li_ka)
            self.norm_B[li] = j_li_ka / (a_j_li * h_li_ka + b_h_li * j_li_ka)

    def norm_radial_wavefunction(self, ang_mom, n, r_factor=2):
        """Calculates the normalized probability density
            for a range (default = 2*a)
        Args:
            ang_mom (int) - Angular momentum
            n (int) - Energy level index for a given l
            r_factor (int) - Radius multiplicative factor to calculate results
        Returns:
            radius - The radius values used for the calculation [nm]
            wavefunction - Normalized probabuility density
        """
        r_in = np.linspace(0, self.a, 200)
        r_out = np.linspace(self.a, self.a * r_factor, 200)
        radius = np.concatenate([r_in, r_out])
        wavefunction = np.concatenate([
            self.norm_A.iloc[n, ang_mom] * special.spherical_jn(
                ang_mom, r_in * self.k_in.iloc[n, ang_mom])**2,
            self.norm_B.iloc[n, ang_mom] * np.absolute(
                spherical_hankel(ang_mom,
                                 1j * r_out * self.k_out.iloc[n, ang_mom],
                                 kind=1))**2
        ])
        return radius, wavefunction

    def norm_wavefunction(self, x, y, z, ang_mom, z_ang_mom, n):
        """Calculates the 3D normalized wavefunction in cartesian coordinates
        """
        XX, YY, ZZ = np.meshgrid(x, y, z)
        RR = np.sqrt(XX**2 + YY**2 + ZZ**2)
        THETA = np.arctan(YY / XX)
        PHI = np.arctan(np.sqrt(XX**2 + YY**2) / ZZ)
        qd_center_mask = RR <= self.a
        qd_outside_mask = RR > self.a
        wavefunction = np.zeros_like(RR, dtype='complex128')
        wavefunction[qd_center_mask] = np.sqrt(
            self.norm_A.iloc[n, ang_mom]) * special.spherical_jn(
                ang_mom, RR[qd_center_mask] * self.k_in.iloc[n, ang_mom])
        wavefunction[qd_outside_mask] = np.sqrt(
            self.norm_B.iloc[n, ang_mom]) * spherical_hankel(
                ang_mom,
                1j * RR[qd_outside_mask] * self.k_out.iloc[n, ang_mom],
                kind=1)

        y_lm = special.sph_harm(z_ang_mom, ang_mom, THETA, PHI)
        return XX, YY, ZZ, wavefunction * y_lm

    def norm_envolute(self, half_sim_size, lat_size, envolute, ang_mom,
                      z_ang_mom, n, Eg, P):
        """
        Return the specific envolute for a QD in a specific band
        Args:
            half_sim_size (float): Half size to calculate the envolute
            lat_size (float): Size steps (defines the number of points to be
                simulated
            envolute (string): Envolute to calculate (L6mt, L6mb, L6pt, L6pb)
            ang_mom (int): l angular momentum for the wavefunction
            z_ang_mom (int): m angular momentum for the wavefunction
            n (int): Energy level value
            Eg (float): Structure Bandgap
            P (tuple - Pl, Pt) - Structural Parameters
        """
        Pl, Pt = P
        coord = np.arange(-half_sim_size, half_sim_size, lat_size)
        # Avoid 0 in coordinates (problem calculating the norm_wavefunction)
        coord = coord[coord != 0]
        k = (np.pi / (half_sim_size * lat_size)) * coord
        XX, YY, ZZ, wavefunction = self.norm_wavefunction(
            coord, coord, coord, ang_mom, z_ang_mom, n)
        if envolute == "L6mt":
            t_matrix = env_f.T_L6mt(k,
                                    k,
                                    k,
                                    Eg,
                                    self.m1,
                                    Pl,
                                    Pt,
                                    diag_band=self.band).astype('complex128')
        elif envolute == "L6mb":
            t_matrix = env_f.T_L6mb(k,
                                    k,
                                    k,
                                    Eg,
                                    self.m1,
                                    Pl,
                                    Pt,
                                    diag_band=self.band).astype('complex128')
        elif envolute == "L6pt":
            t_matrix = env_f.T_L6pt(k,
                                    k,
                                    k,
                                    Eg,
                                    self.m1,
                                    Pl,
                                    Pt,
                                    diag_band=self.band).astype('complex128')
        elif envolute == "L6pb":
            t_matrix = env_f.T_L6pb(k,
                                    k,
                                    k,
                                    Eg,
                                    self.m1,
                                    Pl,
                                    Pt,
                                    diag_band=self.band).astype('complex128')
        else:
            raise Exception(
                "Unknown envolute type - valid (L6mt, L6mb, L6pt, L6pb)")
        return XX, YY, ZZ, ifftn(
            t_matrix * fftn(wavefunction, s=wavefunction.shape)), t_matrix

    def rad_matrix_element(self, i_state, f_state):
        """Calculate matrix elements for a certain initial and final state
        Args:
            i_state (tuple): index of the initial state (l, n)
            f_state (tuple): index of the final state
        Returns:
            rad_element: radial element of the transition [nm]
        """
        # Obtain the respective normalization constants
        norm_Ai, norm_Af = np.sqrt(
            self.norm_A.iloc[i_state[1], i_state[0]]), np.sqrt(
                self.norm_A.iloc[f_state[1], f_state[0]])
        norm_Bi, norm_Bf = np.sqrt(
            self.norm_B.iloc[i_state[1], i_state[0]]), np.sqrt(
                self.norm_B.iloc[f_state[1], f_state[0]])
        # Obtain the respective k_values involved
        k_in_i, k_in_f = self.k_in.iloc[i_state[1],
                                        i_state[0]], self.k_in.iloc[f_state[1],
                                                                    f_state[0]]
        k_out_i, k_out_f = self.k_out.iloc[
            i_state[1], i_state[0]], self.k_out.iloc[f_state[1], f_state[0]]
        # Integration of the bessel part
        bessel_integration = quad(
            lambda x: np.conj(norm_Af) * norm_Ai * special.spherical_jn(
                i_state[0], x * k_in_i) * special.spherical_jn(
                    f_state[0], x * k_in_f) * x**3, 0, self.a)[0]
        # Integration of the hankel part
        hankel_integration = quad(
            lambda x: np.conj(norm_Bf) * norm_Bi * np.conj(
                spherical_hankel(f_state[0], 1j * k_out_f * x, kind=1)
            ) * spherical_hankel(i_state[0], 1j * k_out_i * x, kind=1) * x**3,
            self.a, np.inf)[0]
        return bessel_integration + hankel_integration

    def ang_matrix_element(self, i_state, f_state):
        """Calculates the angular part of the matrix elements
        Args:
            i_state (tuple): l and m values for the initial state
            f_state (tuple): l and m values for the final state
        Returns:
            ep_x, ep_circ_left, ep_circ_right: Components for each polarization
        """
        li, mi = i_state
        lf, mf = f_state
        pre_factor = np.sqrt(
            (2 * li + 1) / (2 * lf + 1)) * CG(1, 0, li, 0, lf, 0).doit()
        ep_z = pre_factor * CG(1, 0, li, mi, lf, mf).doit()  # z polarization
        # left circular polarization (ex+iey)
        ep_circ_left = pre_factor * \
            CG(1, -1, li, mi, lf, mf).doit()*(1/np.sqrt(2))
        # right circular polarization (-ex+iey)
        ep_circ_right = pre_factor * \
            CG(1, 1, li, mi, lf, mf).doit()*(1/np.sqrt(2))
        return ep_z, ep_circ_left, ep_circ_right

    def all_ang_matrix_elements(self, li, lf):
        """Calculates all the angular matrix elements between an initial and
            final l state
        Args:
            li (int): Initial value for the orbital angular momentum
            lf (int): Final value for the orbital angular momentum
        Retuns:
            ang_elements (array): Angular matrix elements
                        - array of size (2*lf+1, 2*li+1, 3)
            average_ang_elements (array): Average angular elements for each
                        polarization
        """
        # Necessary m values
        m_i = np.arange(-li, li + 1)
        m_f = np.arange(-lf, lf + 1)
        M_i, M_f = np.meshgrid(m_i, m_f)  # Grid with the values variation
        # Flatten the data to use in the map function
        M_i, M_f = list(M_i.flatten()), list(M_f.flatten())
        ang_elements = np.array(list(
            map(lambda x, y: self.ang_matrix_element((li, x), (lf, y)), M_i,
                M_f)),
                                dtype=float).reshape(
                                    (2 * lf + 1, 2 * li + 1, 3))
        # Weight factor to average without counting the zeros
        avg_weight_factor = np.zeros_like(ang_elements)
        avg_weight_factor[ang_elements != 0] = 1
        if avg_weight_factor.sum() == 0:
            average_ang_elements = np.average(ang_elements, axis=(0, 1))
        else:
            average_ang_elements = np.average(ang_elements,
                                              axis=(0, 1),
                                              weights=avg_weight_factor)
        return ang_elements, average_ang_elements
