from setuptools import setup, Extension
from Cython.Build import cythonize

MODULE_NAME = "sparse_sim"

# OpenMP paths
OMP_INCLUDE = "/opt/homebrew/opt/libomp/include"
OMP_LIB = "/opt/homebrew/opt/libomp/lib"

extensions = [
    Extension(
        MODULE_NAME,
        sources=[
            "cython/sparse_sim.pyx",
            "src/wavefunction.c",
            "src/pauli.c"
        ],
        include_dirs=["include", "src", "cython", OMP_INCLUDE],
        extra_compile_args=[
            "-O3", "-mcpu=apple-m1", "-flto",
            "-ffast-math", "-funroll-loops", "-fvectorize",
            "-Xpreprocessor", "-fopenmp", f"-I{OMP_INCLUDE}"
        ],
        extra_link_args=[f"-L{OMP_LIB}", "-lomp"],
    )
]

setup(
    ext_modules=cythonize(extensions),
)
