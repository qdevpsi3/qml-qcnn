from setuptools import find_packages, setup

setup(name='qcnn',
      version='0.0.1',
      install_requires=['torch'],
      packages=find_packages('src'),
      package_dir={'': 'src'})
