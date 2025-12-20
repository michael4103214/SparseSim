from pathlib import Path
from setuptools import setup, find_packages, Extension
import os
import platform

root = Path(__file__).parent
pkg = root / "src" / "sparse_sim"
cy = pkg / "cython"
csrc = pkg / "src"
inc = pkg / "include"


def rel(p: Path) -> str:
    return os.path.relpath(p, start=root)


use_c = (cy / "core.c").exists()

shared_csrc = [
    rel(csrc / "pauli.c"),
    rel(csrc / "wavefunction.c"),
    rel(csrc / "density_matrix.c"),
]

pauli_sources = [
    rel(cy / ("pauli.c" if use_c else "pauli.pyx")),
    *shared_csrc,
]

wavefunction_sources = [
    rel(cy / ("wavefunction.c" if use_c else "wavefunction.pyx")),
    *shared_csrc,
]

density_matrix_sources = [
    rel(cy / ("density_matrix.c" if use_c else "density_matrix.pyx")),
    *shared_csrc,
]

include_dirs = [rel(inc), rel(csrc), rel(cy)]
extra_compile_args, extra_link_args = [
    "-O3", "-ffast-math", "-funroll-loops"], []

sys = platform.system()
if sys == "Linux":
    extra_compile_args += ["-fopenmp"]
    extra_link_args += ["-fopenmp"]
elif sys == "Darwin":
    # enable OpenMP only if Homebrew libomp is present
    omp = Path("/opt/homebrew/opt/libomp")
    if (omp / "include").exists() and (omp / "lib").exists():
        include_dirs.append(str(omp / "include"))
        extra_link_args += [f"-L{omp/'lib'}", "-lomp"]
        extra_compile_args += ["-Xpreprocessor", "-fopenmp"]

kwargs = dict(
    name="sparse-sim",
    version="0.2.0",
    package_dir={"": "src"},
    packages=find_packages(where="src", include=[
                           "sparse_sim", "sparse_sim.*"]),
    include_package_data=True,
    zip_safe=False,
    python_requires=">=3.9",
    package_data={
        "sparse_sim": ["cython/*.pyi", "__init__.pyi", "py.typed"],
    }
)

extensions = [
    Extension(
        name="sparse_sim.cython.pauli",
        sources=pauli_sources,
        include_dirs=include_dirs,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        language="c",
    ),
    Extension(
        name="sparse_sim.cython.wavefunction",
        sources=wavefunction_sources,
        include_dirs=include_dirs,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        language="c",
    ),
    Extension(
        name="sparse_sim.cython.density_matrix",
        sources=density_matrix_sources,
        include_dirs=include_dirs,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        language="c",
    ),
]

if use_c:
    setup(ext_modules=extensions, **kwargs)
else:
    from Cython.Build import cythonize
    setup(
        ext_modules=cythonize(
            extensions,
            compiler_directives={"language_level": 3},
        ),
        **kwargs
    )
