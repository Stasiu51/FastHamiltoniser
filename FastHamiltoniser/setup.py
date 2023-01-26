import os
from setuptools import setup, Extension
import numpy as np

os.environ["CC"] = "g++"
os.environ["CXX"] = "g++"

if 'CUDA_PATH' in os.environ:
   CUDA_PATH = os.environ['CUDA_PATH']
else:
   raise Exception("Could not find CUDA_PATH in environment variables. Set it and try again.")


setup(name = 'FastHamiltoniser', version = '1.0',  \
   ext_modules = [
      Extension('FastHamiltoniser', ['FastHamiltoniser.c'], 
      include_dirs=[np.get_include(), os.path.join(CUDA_PATH, "include")],
      libraries=[".\\CUDA\\build\\cuda_lib", "cudart"],
      library_dirs = [".",os.path.join(CUDA_PATH, "lib","x64")],
      extra_compile_args = []
    )],
)