from setuptools import setup, Extension
from Cython.Build import cythonize
import os, platform

MODULE_NAME = "sparse_sim"

system = platform.system() 

sources = [
    "cython/sparse_sim.pyx",
    "src/wavefunction.c",
    "src/pauli.c",
]

include_dirs = ["include", "src", "cython"]
extra_compile_args = ["-O3", "-ffast-math", "-funroll-loops", "-flto", "-fopenmp"]
extra_link_args    = ["-fopenmp"]

if system == "Darwin":
    extra_compile_args += ["-mcpu=apple-m1", "-fvectorize", "-Xpreprocessor"]
    omp_root = "/opt/homebrew/opt/libomp"
    include_dirs.append(os.path.join(omp_root, "include"))
    extra_link_args  += [f"-L{os.path.join(omp_root, 'lib')}", "-lomp"]

elif system == "Linux":
    extra_compile_args.append("-march=native")

# OpenMP paths
OMP_INCLUDE = "/opt/homebrew/opt/libomp/include"
OMP_LIB = "/opt/homebrew/opt/libomp/lib"

extensions = [
    Extension(
        MODULE_NAME,
        sources=sources,
        include_dirs=include_dirs,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    )
]

setup(
    ext_modules=cythonize(extensions, language_level="3")
)
