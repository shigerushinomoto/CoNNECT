#
# setup.py
# Copyright (c) 2020 Daisuke Endo
# This software is released under the MIT License.
# http://opensource.org/licenses/mit-license.php
#
#--------------------------------
# 作成者：Daisuke Endo
# 連絡先:daisuke.endo96@gmail.com
# 最終更新日　2020/5/17
#--------------------------------
from distutils.core import setup, Extension
from Cython.Build import cythonize
from numpy import get_include # cimport numpy を使うため

ext = Extension("crosscorrelogram",
	 sources=["sources/crosscorrelogram.pyx"],
	 include_dirs=['sources', get_include()])
setup(name="crosscorrelogram", ext_modules=cythonize([ext]))
