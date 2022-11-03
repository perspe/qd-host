'''
Basic setup to compile with distutils to create c file
'''
from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy
import os

PY_SMM = os.path.join("smm", "py_smm_base.pyx")
CPP_SMM = os.path.join("smm", "smm_base.cpp")
ZEROS_CALC = os.path.join("cython_utils", "zeros_calc.pyx")

extensions = [
    Extension("smm.py_smm_base", [PY_SMM, CPP_SMM], include_dirs=["smm"]),
    Extension("cython_utils.zeros_calc", [ZEROS_CALC], include_dirs=[numpy.get_include()])
]
extensions = cythonize(extensions)

setup(ext_modules=extensions)
