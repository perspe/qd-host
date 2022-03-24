'''
Basic setup to compile with distutils to create c file
'''
from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(name='zeros_calc',
      ext_modules=cythonize('zeros_calc.pyx'),
      include_dirs=[numpy.get_include()]
      )
