from libc.stdint cimport uintptr_t, uint64_t
from libc.stdlib cimport malloc, free

import numbers
from typing import Set

from .pauli cimport PauliString, PauliSum
from .pauli cimport PauliStringC, PauliSumC
from .wavefunction cimport Wavefunction
from .wavefunction cimport WavefunctionC

cdef extern from "khash.h":
    
    ctypedef int khiter_t

    # Hash table type for DensityMatrixC
    ctypedef struct kh_outer_product_hash_t:
        int n_buckets
        int size
        int n_occupied
        int upper_bound
        uint64_t *keys 
        char *flags     
        OuterProductC **vals 

    # Declare khash functions
    kh_outer_product_hash_t *kh_init_outer_product_hash()
    void kh_destroy_outer_product_hash(kh_outer_product_hash_t *h)
    khiter_t kh_get_outer_product_hash(kh_outer_product_hash_t *h, uint64_t key)
    khiter_t kh_put_outer_product_hash(kh_outer_product_hash_t *h, uint64_t key, int *ret)
    int kh_exist_outer_product_hash(kh_outer_product_hash_t *h, khiter_t k)
    void kh_del_outer_product_hash(kh_outer_product_hash_t *h, khiter_t k)
    OuterProductC *kh_value_outer_product_hash(kh_outer_product_hash_t *h, khiter_t k)

    # Iteration functions
    khiter_t kh_begin_outer_product_hash(kh_outer_product_hash_t *h)
    khiter_t kh_end_outer_product_hash(kh_outer_product_hash_t *h)

cdef extern from "density_matrix.h":

    cdef struct OuterProductC:
        unsigned int N;
        double complex coef;
        unsigned int *ket_orbitals;
        unsigned int *bra_orbitals;
        uint64_t encoding;

    OuterProductC *outer_product_init_c(unsigned int N, double complex coef, unsigned int ket_orbitals[], unsigned int bra_orbitals[])
    void free_outer_product_c(OuterProductC *oprod)
    char *outer_product_to_string_c(OuterProductC *oprod)
    OuterProductC *outer_product_scalar_multiplication_c(OuterProductC *oprod, double complex scalar)
    OuterProductC *outer_product_multiplication(OuterProductC *oprod_left, OuterProductC *oprod_right)
    OuterProductC *outer_product_pauli_string_left_multiplication_c(PauliStringC *pString, OuterProductC *oprod)
    OuterProductC *outer_product_pauli_string_right_multiplication_c(OuterProductC *oprod, PauliStringC *pString)

    cdef struct DensityMatrixC:
        unsigned int N;
        unsigned int o;
        kh_outer_product_hash_t *outer_products;
        double cutoff;

    DensityMatrixC *density_matrix_init_c(unsigned int N);
    DensityMatrixC *density_matrix_init_with_specified_cutoff_c(unsigned int N, double cutoff);
    DensityMatrixC *density_matrix_from_wavefunction_c(WavefunctionC *wfn);
    void free_density_matrix_c(DensityMatrixC *dm);
    char *density_matrix_to_string_c(DensityMatrixC *dm);
    double density_matrix_trace_c(DensityMatrixC *dm);
    DensityMatrixC *density_matrix_scalar_multiplication_c(DensityMatrixC *dm, double complex scalar);
    DensityMatrixC *density_matrix_multiplication_c(DensityMatrixC *dm_left, DensityMatrixC *dm_right);
    void density_matrix_append_outer_product_c(DensityMatrixC *dm, OuterProductC *oprod);
    DensityMatrixC *density_matrix_pauli_string_left_multiplication_c(PauliStringC *pString, DensityMatrixC *dm);
    DensityMatrixC *density_matrix_pauli_string_right_multiplication_c(DensityMatrixC *dm, PauliStringC *pString);
    DensityMatrixC *density_matrix_pauli_sum_left_multiplication_c(PauliSumC *pSum, DensityMatrixC *dm);
    DensityMatrixC *density_matrix_pauli_sum_right_multiplication_c(DensityMatrixC *dm, PauliSumC *pSum);
    DensityMatrixC *density_matrix_pauli_string_evolution_c(PauliStringC *pString, DensityMatrixC *dm, double complex epsilon);
    DensityMatrixC *density_matrix_pauli_sum_evolution_c(PauliSumC *pSum, DensityMatrixC *dm, double complex epsilon);
    DensityMatrixC *density_matrix_remove_global_phase_c(DensityMatrixC *dm);
    DensityMatrixC *density_matrix_remove_near_zero_terms_c(DensityMatrixC *dm, double cutoff);
    DensityMatrixC *density_matrix_CPTP_evolution_c(PauliSumC *H, PauliSumC **Ls, unsigned int num_L, DensityMatrixC *dm, double complex t);


cdef class OuterProduct:

    def __cinit__(self):
        pass

    def __init__(self, unsigned int N, double complex coef, list ket_orbitals, list bra_orbitals):
        cdef unsigned int *c_ket_orbitals
        cdef unsigned int *c_bra_orbitals

        c_ket_orbitals = <unsigned int *> malloc(N * sizeof(unsigned int))
        if not c_ket_orbitals:
            raise MemoryError("Failed to allocate memory for ket orbitals array")

        c_bra_orbitals = <unsigned int *> malloc(N * sizeof(unsigned int))
        if not c_bra_orbitals:
            raise MemoryError("Failed to allocate memory for bra orbitals array")

        cdef int i 
        for i in range(int(N)):
            c_ket_orbitals[i] = <unsigned int> ket_orbitals[i]
            c_bra_orbitals[i] = <unsigned int> bra_orbitals[i]

        cdef OuterProductC *c_oprod = outer_product_init_c(N, coef, c_ket_orbitals, c_bra_orbitals)
        if not c_oprod:
            raise MemoryError("Failed to allocate OuterProductC")
        self._c_oprod = <uintptr_t> c_oprod
        self._in_dm = False

    @staticmethod
    cdef OuterProduct _init_from_c(OuterProductC* ptr):
        """
        Create a new OuterProduct object wrapping the existing OuterProductC pointer.
        This bypasses the usual __cinit__ so we don't re-allocate or re-initialize.
        """
        cdef OuterProduct oprod = OuterProduct.__new__(OuterProduct)
        oprod._c_oprod = <uintptr_t> ptr
        oprod._in_dm = False
        return oprod

    def __dealloc__(self):
        if not self._in_dm:
            free_outer_product_c(<OuterProductC *> self._c_oprod)
    
    def __str__(self):
        cdef char *c_str = outer_product_to_string_c(<OuterProductC *> self._c_oprod)
        py_str = c_str.decode('utf-8')
        free(c_str) 
        return py_str

    def __repr__(self): 
        return self.__str__()

    @property
    def N(self):
        return (<OuterProductC *> self._c_oprod).N

    @property
    def coef(self):
        return (<OuterProductC *> self._c_oprod).coef

    @property
    def ket_orbitals(self):
        cdef unsigned int *c_ket_orbitals = (<OuterProductC *> self._c_oprod).ket_orbitals
        return [c_ket_orbitals[i] for i in range(self.N)]

    @property
    def bra_orbitals(self):
        cdef unsigned int *c_bra_orbitals = (<OuterProductC *> self._c_oprod).bra_orbitals
        return [c_bra_orbitals[i] for i in range(self.N)]

    @property
    def encoding(self):
        return (<OuterProductC *> self._c_oprod).encoding

    def __add__(self, right):
        if isinstance(right, OuterProduct):
            dm = DensityMatrix(self.N)
            dm.append_outer_product(self)
            dm.append_outer_product(right)
            return dm
        return NotImplemented

    def __rmul__(self, left):
        if isinstance(left, numbers.Number):
            return outer_product_scalar_multiplication(self, left)
        return NotImplemented

    def __mul__(self, right):
        if isinstance(right, PauliString):
            return outer_product_pauli_string_right_multiplication(self, right)
        return NotImplemented

cdef class DensityMatrix:

    def __cinit__(self):
        pass

    def __init__(self, N, cutoff=1e-16):
        cdef DensityMatrixC *c_dm = density_matrix_init_with_specified_cutoff_c(N, cutoff)
        if not c_dm:
            raise MemoryError("Failed to allocate DensityMatrixC")
        self._c_dm = <uintptr_t> c_dm

    @staticmethod
    cdef DensityMatrix _init_from_c(DensityMatrixC* ptr):
        """
        Create a new DensityMatrix object wrapping the existing DensityMatrixC pointer.
        This bypasses the usual __cinit__ so we don't re-allocate or re-initialize.
        """
        cdef DensityMatrix dm = DensityMatrix.__new__(DensityMatrix)
        dm._c_dm = <uintptr_t> ptr
        return dm

    def __dealloc__(self):
        free_density_matrix_c(<DensityMatrixC *> self._c_dm)

    def __str__(self):
        cdef char *c_str = density_matrix_to_string_c(<DensityMatrixC *> self._c_dm)
        py_str = c_str.decode('utf-8')
        free(c_str) 
        return py_str

    def __repr__(self): 
        return self.__str__()

    def trace(self):
        return density_matrix_trace_c(<DensityMatrixC *> self._c_dm)

    def append_outer_product(self, OuterProduct oprod):
        assert not oprod._in_dm, "OuterProduct is already part of a DensityMatrix"
        oprod._in_dm = True
        density_matrix_append_outer_product_c(<DensityMatrixC *> self._c_dm, <OuterProductC *> oprod._c_oprod)
        return self

    def remove_global_phase(self):
        cdef DensityMatrixC *new_dm = density_matrix_remove_global_phase_c(<DensityMatrixC *> self._c_dm)
        if not new_dm:
            raise MemoryError("density_matrix_remove_global_phase_c returned NULL")
        return DensityMatrix._init_from_c(new_dm)

    def remove_near_zero_terms(self, double cutoff):
        cdef DensityMatrixC *new_dm = density_matrix_remove_near_zero_terms_c(<DensityMatrixC *> self._c_dm, cutoff)
        if not new_dm:
            raise MemoryError("density_matrix_remove_near_zero_terms_c returned NULL")
        return DensityMatrix._init_from_c(new_dm)

    @property
    def o(self):
        return (<DensityMatrixC *> self._c_dm).o

    @property
    def N(self):
        return (<DensityMatrixC *> self._c_dm).N

    def __mul__(self, right):
        if isinstance(right, DensityMatrix):
            return density_matrix_multiplication(self, right)
        elif isinstance(right, PauliString):
            return density_matrix_pauli_string_right_multiplication(self, right)
        elif isinstance(right, PauliSum):
            return density_matrix_pauli_sum_right_multiplication(self, right)
        return NotImplemented

    def __rmul__(self, left):
        if isinstance(left, numbers.Number):
            return density_matrix_scalar_multiplication(self, left)
        elif isinstance(left, PauliString):
            return density_matrix_pauli_string_left_multiplication(left, self)
        elif isinstance(left, PauliSum):
            return density_matrix_pauli_sum_left_multiplication(left, self)
        return NotImplemented

    def __add__(self, right):
        if isinstance(right, OuterProduct):
            return self.append_outer_product(right)
        return NotImplemented

def density_matrix_from_wavefunction(Wavefunction wfn):
    """Create a DensityMatrix from a Wavefunction."""
    cdef DensityMatrixC *c_dm = density_matrix_from_wavefunction_c(<WavefunctionC *> wfn._c_wfn)
    if not c_dm:
        raise MemoryError("density_matrix_from_wavefunction_c returned NULL")
    return DensityMatrix._init_from_c(c_dm)

def outer_product_scalar_multiplication(OuterProduct oprod, complex scalar):
    """Multiply an OuterProduct by a scalar."""
    cdef OuterProductC *new_oprod = outer_product_scalar_multiplication_c(<OuterProductC *> oprod._c_oprod, scalar)
    if not new_oprod:
        raise MemoryError("outer_product_scalar_multiplication_c returned NULL")
    return OuterProduct._init_from_c(new_oprod)

def density_matrix_scalar_multiplication(DensityMatrix dm, complex scalar):
    """Multiply a DensityMatrix by a scalar."""
    cdef DensityMatrixC *new_dm = density_matrix_scalar_multiplication_c(<DensityMatrixC *> dm._c_dm, scalar)
    if not new_dm:
        raise MemoryError("density_matrix_scalar_multiplication_c returned NULL")
    return DensityMatrix._init_from_c(new_dm)

def density_matrix_multiplication(DensityMatrix left, DensityMatrix right):
    """Multiply two DensityMatrices."""
    cdef DensityMatrixC *new_dm = density_matrix_multiplication_c(<DensityMatrixC *> left._c_dm, <DensityMatrixC *> right._c_dm)
    if not new_dm:
        raise MemoryError("density_matrix_multiplication_c returned NULL")
    return DensityMatrix._init_from_c(new_dm)

def outer_product_pauli_string_left_multiplication(PauliString pString, OuterProduct oprod):
    """Multiply a Pauli string on the left of an OuterProduct."""
    cdef OuterProductC *new_oprod = outer_product_pauli_string_left_multiplication_c(<PauliStringC *> pString._c_pString, <OuterProductC *> oprod._c_oprod)
    if not new_oprod:
        raise MemoryError("outer_product_pauli_string_left_multiplication_c returned NULL")
    return OuterProduct._init_from_c(new_oprod)

def outer_product_pauli_string_right_multiplication(OuterProduct oprod, PauliString pString):
    """Multiply a Pauli string on the right of an OuterProduct."""
    cdef OuterProductC *new_oprod = outer_product_pauli_string_right_multiplication_c(<OuterProductC *> oprod._c_oprod, <PauliStringC *> pString._c_pString)
    if not new_oprod:
        raise MemoryError("outer_product_pauli_string_right_multiplication_c returned NULL")
    return OuterProduct._init_from_c(new_oprod)

def density_matrix_pauli_string_left_multiplication(PauliString pString, DensityMatrix dm):
    """Multiply a Pauli string on the left of a DensityMatrix."""
    cdef DensityMatrixC *new_dm = density_matrix_pauli_string_left_multiplication_c(<PauliStringC *> pString._c_pString, <DensityMatrixC *> dm._c_dm)
    if not new_dm:
        raise MemoryError("density_matrix_pauli_string_left_multiplication_c returned NULL")
    return DensityMatrix._init_from_c(new_dm)

def density_matrix_pauli_string_right_multiplication(DensityMatrix dm, PauliString pString):
    """Multiply a Pauli string on the right of a DensityMatrix."""
    cdef DensityMatrixC *new_dm = density_matrix_pauli_string_right_multiplication_c(<DensityMatrixC *> dm._c_dm, <PauliStringC *> pString._c_pString)
    if not new_dm:
        raise MemoryError("density_matrix_pauli_string_right_multiplication_c returned NULL")
    return DensityMatrix._init_from_c(new_dm)

def density_matrix_pauli_sum_left_multiplication(PauliSum pSum, DensityMatrix dm):
    """Multiply a Pauli sum on the left of a DensityMatrix."""
    cdef DensityMatrixC *new_dm = density_matrix_pauli_sum_left_multiplication_c(<PauliSumC *> pSum._c_pSum, <DensityMatrixC *> dm._c_dm)
    if not new_dm:
        raise MemoryError("density_matrix_pauli_sum_left_multiplication_c returned NULL")
    return DensityMatrix._init_from_c(new_dm)

def density_matrix_pauli_sum_right_multiplication(DensityMatrix dm, PauliSum pSum):
    """Multiply a Pauli sum on the right of a DensityMatrix."""
    cdef DensityMatrixC *new_dm = density_matrix_pauli_sum_right_multiplication_c(<DensityMatrixC *> dm._c_dm, <PauliSumC *> pSum._c_pSum)
    if not new_dm:
        raise MemoryError("density_matrix_pauli_sum_right_multiplication_c returned NULL")
    return DensityMatrix._init_from_c(new_dm)

def density_matrix_pauli_string_evolution(PauliString pString, DensityMatrix dm, double complex epsilon):
    """Evolve a density matrix under a Pauli string using exponentiation."""
    cdef DensityMatrixC *new_dm = density_matrix_pauli_string_evolution_c(<PauliStringC *> pString._c_pString, <DensityMatrixC *> dm._c_dm, epsilon)
    if not new_dm:
        raise MemoryError("density_matrix_pauli_string_evolution_c returned NULL")
    return DensityMatrix._init_from_c(new_dm)

def density_matrix_pauli_sum_evolution(PauliSum pSum, DensityMatrix dm, double complex epsilon):
    """Evolve a density matrix under a Pauli sum using exponentiation."""
    cdef DensityMatrixC *new_dm = density_matrix_pauli_sum_evolution_c(<PauliSumC *> pSum._c_pSum, <DensityMatrixC *> dm._c_dm, epsilon)
    if not new_dm:
        raise MemoryError("density_matrix_pauli_sum_evolution_c returned NULL")
    return DensityMatrix._init_from_c(new_dm)

def density_matrix_CPTP_evolution(PauliSum H, list[PauliSum] Ls, DensityMatrix dm, double complex t):
    """Evolve a density matrix under a Hamiltonian and Lindblad operators using CPTP evolution."""
    cdef int num_Ls = len(Ls)

    cdef PauliSumC **c_Ls = <PauliSumC **> malloc(num_Ls * sizeof(PauliSumC *))
    if not c_Ls:
        raise MemoryError("Failed to allocate memory for Lindblad operators array")
    cdef int i
    for i in range(num_Ls):
        Li = <PauliSum> Ls[i]
        c_Ls[i] = <PauliSumC *> Li._c_pSum

    cdef DensityMatrixC *new_dm = density_matrix_CPTP_evolution_c(<PauliSumC *> H._c_pSum, c_Ls, num_Ls, <DensityMatrixC *> dm._c_dm, t)
    if not new_dm:
        raise MemoryError("density_matrix_CPTP_evolution_c returned NULL")

    free(c_Ls)
    return DensityMatrix._init_from_c(new_dm)

def density_matrix_perform_tomography(DensityMatrix dm, Set[str] measurements):
    
    tomography = {}

    for measurement in measurements:
        s = list(measurement)
        pString = PauliString(len(s), 1, s)
        new_dm = density_matrix_pauli_string_left_multiplication(pString, dm)
        exp_value = new_dm.trace()
        tomography[measurement] = exp_value

    return tomography

def density_matrix_to_probability_distribution(DensityMatrix dm):
    cdef DensityMatrixC* c_dm = <DensityMatrixC *> dm._c_dm
    cdef khiter_t k
    cdef OuterProductC* c_oprod
    
    trace = dm.trace()

    if trace == 0:
        return {}

    probabilities = {}

    for k in range(kh_begin_outer_product_hash(c_dm.outer_products), kh_end_outer_product_hash(c_dm.outer_products)):
        if kh_exist_outer_product_hash(c_dm.outer_products, k):
            c_oprod = kh_value_outer_product_hash(c_dm.outer_products, k)
            ket = c_oprod.ket_orbitals
            bra = c_oprod.bra_orbitals

            diagonal = True
            for i in range(dm.N):
                if ket[i] != bra[i]:
                    diagonal = False
                    break
            
            if diagonal:
                bit_string = ''
                for i in range(dm.N):
                    bit_string += str(ket[i])
                prob = c_oprod.coef / trace
                probabilities[bit_string] = prob

    return probabilities
