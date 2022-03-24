'''
Basic setup to compile with distutils to create c file
'''
from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy

extensions = [
    Extension("cython_utils.zeros_calc",
              ["cython_utils/zeros_calc.pyx"])
]
extensions = cythonize(extensions)

setup(ext_modules=extensions,
      include_dirs=[numpy.get_include()])
