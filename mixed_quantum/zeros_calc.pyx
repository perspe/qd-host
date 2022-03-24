"""
Cython Module to Calculate the zeros
"""

import cython
from scipy.optimize import brentq
import inspect
# import numpy as np
# cimport numpy as cnp

"""
####################################################
# Base functions
####################################################
"""

# Avoid boundary check, and check for negative indexing
@cython.binding(True)
@cython.boundscheck(False)
def zeros_f(func, x, **kwargs):
    """Calculate the zeros of func in the given range defined by x
    Args:
        func (function): Function to calculate the zeros
        x (array): Array defining the range of values to calculate the zeros
    **kwargs:
        args = Optional arguments for func
    Returns:
        zeros: Array with the several zeros of the function
    """
    # Define datatypes
    cdef:
        list f_variables = str(inspect.signature(func))[1:-1].split(", ")[1:]
        dict f_args = dict()
        int x_size = len(x) - 1, x_i
        list zeros = list()
        double diff
        double[::1] func_data

    # Relate function variables with input args
    for (i, j) in zip(f_variables, range(len(f_variables))):
        f_args[i] = kwargs.get('args')[j]

    # Estimate the 0's location
    func_data = func(x, **f_args)
    for x_i in range(x_size):
        diff = abs(func_data[x_i + 1] - func_data[x_i])
        if (diff > <double > max(abs(func_data[x_i + 1]),
                                 abs(func_data[x_i]))) and (diff < 1):
            zeros.append(brentq(
                func, x[x_i], x[x_i + 1], args=kwargs.get("args")))
        else:
            continue
    return zeros
