"""
Calculate the envelope function for the 4-band EKPH
Method:
    - Calculate the normal wavefunction for each band
    - Calculate the Discrete Fourier Transform
    - Multiply by the Transformation matrix T
    - Calculate the inverse Discrete Fourier Transform
"""
import numpy as np
from scipy import constants as cnt


def T_L6mt(kx, ky, kz, Eg, m, Pl, Pt, diag_band="CB1"):
    """
    Determine the transformation elements of T that transform the
    diagonalized hamiltonian eigenfunction into the L6-↑ envolute
    Args:
        - kx, ky, kz: 1D arrays with the coordinates
        - Eg, m, Pl, Pt: Constant values dependent on the structure
        - diag_band: Band of the diagonalized Hamiltonian (CB1, CB2, VB1, VB2)
    """
    # Add a dimensional constant to convert nm into m
    dim = 1e9
    # Consants (Plancks constant is in eV)
    h_bar_m = cnt.hbar / (cnt.m_e * m)
    h_bar2_m = cnt.hbar**2 / (cnt.m_e * m)
    Kx, Ky, Kz = np.meshgrid(kx, ky, kz)
    K2 = Kx**2 + Ky**2 + Kz**2
    # Necessary constant values
    # alpha = Eg/2 + h_bar**2 k**2/(2m)
    alpha = Eg * cnt.eV / 2 + h_bar2_m * K2 * dim**2 / 2
    # A = h_bar / m * (Pl * Kz)
    A = h_bar_m * Pl * Kz * dim
    # B = h_bar / m * (Pt(Kx - iKy))
    B = h_bar_m * Pt * (Kx * dim - 1j * Ky * dim)
    # Beta = alpha - sqrt(alpha**2 + A**2 + |B|**2)
    beta = alpha - np.sqrt(alpha**2 + A**2 + np.abs(B)**2)
    # C_alpha = sqrt(beta**2 + A**2 + |B|**2)
    C_alpha = np.sqrt(beta**2 + A**2 + np.abs(B)**2)
    if diag_band == "CB1":
        return -np.conjugate(B) / C_alpha
    elif diag_band == "CB2":
        return A / C_alpha
    elif diag_band == "VB1":
        return np.zeros_like(K2)
    elif diag_band == "VB2":
        return beta / C_alpha
    else:
        raise Exception("Invalid diag_band value (valid - CB1, CB2, VB1, VB2)")


def T_L6mb(kx, ky, kz, Eg, m, Pl, Pt, diag_band="CB1"):
    """
    Determine the transformation elements of T that transform the
    diagonalized hamiltonian eigenfunction into the L6-↓ envolute
    Args:
        - kx, ky, kz: 1D arrays with the coordinates
        - Eg, m, Pl, Pt: Constant values dependent on the structure
        - diag_band: Band of the diagonalized Hamiltonian (CB1, CB2, VB1, VB2)
    """
    # Add a dimensional constant to convert nm into m
    dim = 1e9
    # Consants (Plancks constant is in eV)
    h_bar_m = cnt.hbar / (cnt.m_e * m)
    h_bar2_m = cnt.hbar**2 / (cnt.m_e * m)
    Kx, Ky, Kz = np.meshgrid(kx, ky, kz)
    K2 = Kx**2 + Ky**2 + Kz**2
    # Necessary constant values
    # alpha = Eg/2 + h_bar**2 k**2/(2m)
    alpha = Eg * cnt.eV / 2 + h_bar2_m * K2 * dim**2 / 2
    # A = h_bar / m * (Pl * Kz)
    A = h_bar_m * Pl * Kz * dim
    # B = h_bar / m * (Pt(Kx - iKy))
    B = h_bar_m * Pt * (Kx * dim - 1j * Ky * dim)
    # Beta = alpha - sqrt(alpha**2 + A**2 + |B|**2)
    beta = alpha - np.sqrt(alpha**2 + A**2 + np.abs(B)**2)
    # C_alpha = sqrt(beta**2 + A**2 + |B|**2)
    C_alpha = np.sqrt(beta**2 + A**2 + np.abs(B)**2)
    if diag_band == "CB1":
        return A / C_alpha
    elif diag_band == "CB2":
        return B / C_alpha
    elif diag_band == "VB1":
        return - beta / C_alpha
    elif diag_band == "VB2":
        return np.zeros_like(K2)
    else:
        raise Exception("Invalid diag_band value (valid - CB1, CB2, VB1, VB2)")


def T_L6pt(kx, ky, kz, Eg, m, Pl, Pt, diag_band="CB1"):
    """
    Determine the transformation elements of T that transform the
    diagonalized hamiltonian eigenfunction into the L6+↑ envolute
    Args:
        - kx, ky, kz: 1D arrays with the coordinates
        - Eg, m, Pl, Pt: Constant values dependent on the structure
        - diag_band: Band of the diagonalized Hamiltonian (CB1, CB2, VB1, VB2)
    """
    # Add a dimensional constant to convert nm into m
    dim = 1e9
    # Consants (Plancks constant is in eV)
    h_bar_m = cnt.hbar / (cnt.m_e * m)
    h_bar2_m = cnt.hbar**2 / (cnt.m_e * m)
    Kx, Ky, Kz = np.meshgrid(kx, ky, kz)
    K2 = Kx**2 + Ky**2 + Kz**2
    # Necessary constant values
    # alpha = Eg/2 + h_bar**2 k**2/(2m)
    alpha = Eg * cnt.eV / 2 + h_bar2_m * K2 * dim**2 / 2
    # A = h_bar / m * (Pl * Kz)
    A = h_bar_m * Pl * Kz * dim
    # B = h_bar / m * (Pt(Kx - iKy))
    B = h_bar_m * Pt * (Kx * dim - 1j * Ky * dim)
    # Beta = alpha - sqrt(alpha**2 + A**2 + |B|**2)
    beta = alpha - np.sqrt(alpha**2 + A**2 + np.abs(B)**2)
    # C_alpha = sqrt(beta**2 + A**2 + |B|**2)
    C_alpha = np.sqrt(beta**2 + A**2 + np.abs(B)**2)
    if diag_band == "CB1":
        return np.zeros_like(K2)
    elif diag_band == "CB2":
        return beta / C_alpha
    elif diag_band == "VB1":
        return np.conjugate(B) / C_alpha
    elif diag_band == "VB2":
        return - A / C_alpha
    else:
        raise Exception("Invalid diag_band value (valid - CB1, CB2, VB1, VB2)")


def T_L6pb(kx, ky, kz, Eg, m, Pl, Pt, diag_band="CB1"):
    """
    Determine the transformation elements of T that transform the
    diagonalized hamiltonian eigenfunction into the L6+↓ envolute
    Args:
        - kx, ky, kz: 1D arrays with the coordinates
        - Eg, m, Pl, Pt: Constant values dependent on the structure
        - diag_band: Band of the diagonalized Hamiltonian (CB1, CB2, VB1, VB2)
    """
    # Add a dimensional constant to convert nm into m
    dim = 1e9
    # Consants (Plancks constant is in eV)
    h_bar_m = cnt.hbar / (cnt.m_e * m)
    h_bar2_m = cnt.hbar**2 / (cnt.m_e * m)
    Kx, Ky, Kz = np.meshgrid(kx, ky, kz)
    K2 = Kx**2 + Ky**2 + Kz**2
    # Necessary constant values
    # alpha = Eg/2 + h_bar**2 k**2/(2m)
    alpha = Eg * cnt.eV / 2 + h_bar2_m * K2 * dim**2 / 2
    # A = h_bar / m * (Pl * Kz)
    A = h_bar_m * Pl * Kz * dim
    # B = h_bar / m * (Pt(Kx - iKy))
    B = h_bar_m * Pt * (Kx * dim - 1j * Ky * dim)
    # Beta = alpha - sqrt(alpha**2 + A**2 + |B|**2)
    beta = alpha - np.sqrt(alpha**2 + A**2 + np.abs(B)**2)
    # C_alpha = sqrt(beta**2 + A**2 + |B|**2)
    C_alpha = np.sqrt(beta**2 + A**2 + np.abs(B)**2)
    if diag_band == "CB1":
        return beta / C_alpha
    elif diag_band == "CB2":
        return np.zeros_like(K2)
    elif diag_band == "VB1":
        return A / C_alpha
    elif diag_band == "VB2":
        return np.conjugate(B) / C_alpha
    else:
        raise Exception("Invalid diag_band value (valid - CB1, CB2, VB1, VB2)")


if __name__ == "__main__":
    # Testing Block
    # Variable initialization
    Eg = 1.5
    kx = np.linspace(-2, 2, 100, dtype="complex128")
    ky = np.linspace(-2, 2, 100, dtype="complex128")
    kz = np.linspace(-2, 2, 100, dtype="complex128")

    # Calculation of the L6mt envolute
    ts = T_L6mt(kx, ky, kz, Eg, 0.08, 1e-25, 1e-25, "CB1")
    ts_adj = np.conj(ts)
    tx = T_L6mb(kx, ky, kz, Eg, 0.08, 1e-25, 1e-25, "CB1")
    tx_adj = np.conj(tx)
    ty = T_L6pt(kx, ky, kz, Eg, 0.08, 1e-25, 1e-25, "CB1")
    ty_adj = np.conj(ty)
    tz = T_L6pb(kx, ky, kz, Eg, 0.08, 1e-25, 1e-25, "CB1")
    tz_adj = np.conj(tz)
    # Checks normalization of the matrices
    # Last test does check everything out
