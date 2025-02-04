from setuptools import setup, Extension
from Cython.Build import cythonize

MODULE_NAME = "sparse_sim"

extensions = [
    Extension(
        MODULE_NAME,
        sources=[
            "cython/sparse_sim.pyx",
            "src/wavefunction.c",
            "src/pauli.c"
        ],
        include_dirs=["include", "src", "cython"],
        extra_compile_args=["-O3", "-mcpu=apple-m1", "-flto",
                            "-ffast-math", "-funroll-loops", "-fvectorize"],
    )
]

setup(
    ext_modules=cythonize(extensions),
)
