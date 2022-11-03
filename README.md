# 4-band k.p-based Method to Solve QD@Host Problems

This repository provides a python interface to solve QD@Host problems (Quantum
Dot superlattice structure in an arbitrary host material).

The Band_Model.ipynb provides a detailed description on how to use the program.

## Installation

Since most of the program is written in python it should work in any OS with
little effort. There are, however, 2 parts of the program that do require some
managing for everything to work properly.

First, the main QD class does depend on a cython module, that was necessary to
improve performance. This part is required, as it is paramount for the core function
of the module. Secondly, there is also the Scattering Matrix Method module that
has a cpp component. It is not necessary, however the following method should
compile both modules together.

Inside the band_model folder there is a setup.py that can be run to build the
abovementioned modules. It can be run as follows.

```python
python setup.py build_ext -if
```

__Note__: For the above to run it is necessary to have properly installed C and
CPP compilers (such as gcc and g++) and reacheable in the PATH system variable.

