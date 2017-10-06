from setuptools import setup, find_packages
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
      requires = ['numpy','matplotlib','scipy', 'sklearn'],
      provides = ['LetsTalkAboutQuench'],
      packages = ['code'],
      scripts=['code/catalogs.py', 'code/fstarforms.py', 'code/util.py']
)
