"""
Provide python-space access to the functions exposed in numpy/__init__.pxd
for testing.
"""

import numpy as np
from distutils.core import setup
from Cython.Build import cythonize
from setuptools.extension import Extension
import os

from hpy.devel import HPyDevel
hpy_include_dirs = HPyDevel().get_extra_include_dirs()

macros = [("NPY_NO_DEPRECATED_API", 0)]

checks = Extension(
    "checks",
    sources=[os.path.join('.', "checks.pyx")],
    include_dirs=[np.get_include()] + hpy_include_dirs,
    define_macros=macros,
)

extensions = [checks]

setup(
    ext_modules=cythonize(extensions)
)
