# Determine Platform
UNAME_S := $(shell uname -s)
$(info >>> Make loaded $(MAKEFILE_LIST); UNAME = $(shell uname -s))

# Compiler and flags
CC      ?= gcc          
CFLAGS  := -O3 -ffast-math -funroll-loops \
           -Wall -Wextra -Wshadow -Wformat=2 -Wconversion \
           -g -I./include

# Platform specific flags
ifeq ($(UNAME_S),Darwin)          # macOS / Apple Silicon
    CC      := clang
    CFLAGS += -mcpu=apple-m1 -fvectorize -flto -Xpreprocessor -fopenmp
else                              # Linux / GCC (default)
    CFLAGS += -march=native -flto -fopenmp
endif

# Disabled C Flags
CFLAGS += -Wno-padded -Wno-c++98-compat -Wno-disabled-macro-expansion \
          -Wno-poison-system-directories -Wno-declaration-after-statement

PYTHON = python3

# Directories
SRC_DIR = src
INCLUDE_DIR = include
BUILD_DIR = build
CYTHON_DIR = cython
TESTS_DIR = tests

# Targets
TEST_PAULI = $(BUILD_DIR)/test_pauli
TEST_WAVEFUNCTION = $(BUILD_DIR)/test_wavefunction
CYTHON_BUILD = $(CYTHON_DIR)/sparse_sim.so

# Default target
all: $(BUILD_DIR) $(TEST_PAULI) $(TEST_WAVEFUNCTION) cython

# Ensure build directory exists
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# Compile test_pauli
$(TEST_PAULI): $(SRC_DIR)/test_pauli.c $(SRC_DIR)/pauli.c $(SRC_DIR)/wavefunction.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# Compile test_wavefunction
$(TEST_WAVEFUNCTION): $(SRC_DIR)/test_wavefunction.c $(SRC_DIR)/wavefunction.c $(SRC_DIR)/pauli.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# Cython build (ensures pauli.c and wavefunction.c are compiled first)
cython: $(CYTHON_BUILD)

$(CYTHON_BUILD): $(CYTHON_DIR)/sparse_sim.pyx $(SRC_DIR)/wavefunction.c $(SRC_DIR)/pauli.c
	$(PYTHON) setup.py build_ext --inplace

# Run Python tests
test-pauli:
	PYTHONPATH=$(shell pwd) $(PYTHON) tests/test_pauli.py

test-wavefunction:
	PYTHONPATH=$(shell pwd) $(PYTHON) tests/test_wavefunction.py

test-fermion:
	PYTHONPATH=$(shell pwd):$(shell pwd)/fermion $(PYTHON) tests/test_fermion.py

test-hamiltonian:
	PYTHONPATH=$(shell pwd):$(shell pwd)/fermion $(PYTHON) tests/test_hamiltonian.py

test-rdm:
	PYTHONPATH=$(shell pwd):$(shell pwd)/fermion $(PYTHON) tests/test_rdm.py

test-qiskit-wrapper:
	PYTHONPATH=$(shell pwd):$(shell pwd)/fermion $(PYTHON) tests/test_qiskit_wrapper.py

test-projector:
	PYTHONPATH=$(shell pwd):$(shell pwd)/fermion $(PYTHON) tests/test_projector.py

test-nuclear:
	PYTHONPATH=$(shell pwd):$(shell pwd)/fermion $(PYTHON) tests/test_nuclear.py

# Clean up generated files
clean:
	rm -rf $(BUILD_DIR) *.so *.o
	rm -rf cython/__pycache__