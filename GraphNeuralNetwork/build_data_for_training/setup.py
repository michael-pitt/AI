#!/usr/bin/env python
# coding: utf-8

########################################################################
# ======================  TrackML CHALLENGE MODEL  =====================
########################################################################
# Author: Isabelle Guyon, Victor Estrade
# Date: Apr 10, 2018

# ALL INFORMATION, SOFTWARE, DOCUMENTATION, AND DATA ARE PROVIDED "AS-IS".
# PARIS-SUD UNIVERSITY, THE ORGANIZERS OR CODE AUTHORS DISCLAIM
# ANY EXPRESSED OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR ANY PARTICULAR PURPOSE, AND THE
# WARRANTY OF NON-INFRIGEMENT OF ANY THIRD PARTY'S INTELLECTUAL PROPERTY RIGHTS.
# IN NO EVENT SHALL PARIS-SUD UNIVERSITY AND/OR OTHER ORGANIZERS BE LIABLE FOR ANY SPECIAL,
# INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER ARISING OUT OF OR IN
# CONNECTION WITH THE USE OR PERFORMANCE OF SOFTWARE, DOCUMENTS, MATERIALS,
# PUBLICATIONS, OR INFORMATION MADE AVAILABLE FOR THE CHALLENGE.
from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension
import numpy as np

ext_modules = [
    Extension(
        "preprocess",
        ["preprocess.pyx"],
        extra_compile_args=['-ffast-math']
#        extra_link_args=['']
    )
]

setup(
    name='preprocess-parallel',
    ext_modules=cythonize(ext_modules),
    include_dirs=[np.get_include()]
)
