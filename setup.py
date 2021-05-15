# @Author: Arthur Voronin <arthur>
# @Date:   11.05.2021
# @Filename: setup.py
# @Last modified by:   arthur
# @Last modified time: 11.05.2021

from setuptools import setup, find_packages

setup(name='pyrexMD',
      version='1.0',
      description='workflow orientated setup and analyses of contact-guided Replica Exchange Molecular Dynamics (requires GROMACS for MD)',
      license="MIT",
      author='Arthur Voronin',
      email='arthur.voronin@kit.edu',
      packages=find_packages(),
      install_requires=[
          'biopython>=1.78',
          'duecredit>=0.9.1',
          'future>=0.18.2',
          'GromacsWrapper==0.8.0',
          'h5py==2.10.0',
          'heat==1.0.0',
          'ipywidgets>=7.4.2',
          'jupyter>=1.0.0',
          'nglview==2.0',
          'MDAnalysis>=1.1.1',
          'numpy>=1.18.2',
          'pdf2image>=1.10.0',
          'Pillow>=8.2.0',
          'pyperclip>=1.8.2',
          'seaborn>=0.11.1',
          'termcolor>=1.1.0',
          'tqdm>=4.60.0',
          'widgetsnbextension>=3.5.1'
      ]
      )
