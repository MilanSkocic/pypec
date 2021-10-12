r"""Setup"""
import os
from setuptools import setup, find_packages
import pec


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname), 'r', encoding='utf-8').read()


setup(name=pec.__package_name__,
      version=pec.__version__,
      maintainer=pec.__maintainer__,
      maintainer_email=pec.__maintainer_email__,
      author=pec.__author__,
      author_email=pec.__author_email__,
      description=pec.__package_name__,
      long_description=read('README.md'),
#      url='https://milanskocic.github.io/pec/index.html',
      download_url='https://github.com/MilanSkocic/PyPEC3/',
      packages=find_packages(),
      include_package_data=True,
      python_requires='>=3.6',
      install_requires=read('./requirements.txt').split('\n'),
      classifiers=["Development Status :: 4 - Beta",
                   "Intended Audience :: Science/Research",
                   "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
                   "Programming Language :: Python",
                   "Programming Language :: Python :: 3 :: Only",
                   "Programming Language :: Python :: 3.6",
                   "Programming Language :: Python :: 3.7",
                   "Programming Language :: Python :: 3.8",
                   "Programming Language :: Python :: 3.9",
                   "Topic :: Scientific/Engineering",
                   "Operating System :: OS Independent"]
      )
