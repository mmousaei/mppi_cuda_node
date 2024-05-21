from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

d = generate_distutils_setup(
  packages=['mppi_cuda_node'],
  package_dir={'': 'scripts'}
)

setup(**d)
