#from setuptools import setup, find_packages
from distutils.core import setup
import os, sys

__version__ = '0.1'

setup(name = 'LetsTalkAboutQuench',
      version = __version__,
      description = 'fitting the SFMS',
      author='ChangHoon Hahn',
      author_email='changhoonhahn@lbl.gov',
      url='',
      platforms=['*nix'],
      license='GPL',
      requires = ['numpy','matplotlib','scipy', 'sklearn', 'extreme_deconvolution'],
      provides = ['LetsTalkAboutQuench'],
      packages = ['letstalkaboutquench'],
      scripts=['letstalkaboutquench/catalogs.py', 'letstalkaboutquench/fstarforms.py', 'letstalkaboutquench/util.py']
)
