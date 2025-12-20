from libc.stdint cimport uintptr_t, uint64_t
from libc.stdlib cimport malloc, free

import numbers
from typing import Set

from .pauli cimport PauliString, PauliSum
from .pauli cimport PauliStringC, PauliSumC

cdef extern from "khash.h":
    
    ctypedef int khiter_t

    # Hash table struct type for SlaterDeterminantC
    ctypedef struct kh_slater_hash_t:
        int n_buckets
        int size
        int n_occupied
        int upper_bound
        uint64_t *keys 
        char *flags     
        SlaterDeterminantC **vals 

    # Declare khash functions
    kh_slater_hash_t *kh_init_slater_hash()
    void kh_destroy_slater_hash(kh_slater_hash_t *h)
    khiter_t kh_get_slater_hash(kh_slater_hash_t *h, uint64_t key)
    khiter_t kh_put_slater_hash(kh_slater_hash_t *h, uint64_t key, int *ret)
    int kh_exist_slater_hash(kh_slater_hash_t *h, khiter_t k)
    void kh_del_slater_hash(kh_slater_hash_t *h, khiter_t k)
    SlaterDeterminantC *kh_value_slater_hash(kh_slater_hash_t *h, khiter_t k)

    # Iteration functions
    khiter_t kh_begin_slater_hash(kh_slater_hash_t *h)
    khiter_t kh_end_slater_hash(kh_slater_hash_t *h)

cdef extern from "wavefunction.h":

    cdef struct SlaterDeterminantC:
            unsigned int N
            double complex coef
            unsigned int *orbitals
            uint64_t encoding

    SlaterDeterminantC *slater_determinant_init_c(unsigned int N, double complex coef, unsigned int orbitals[])
    void free_slater_determinant_c(SlaterDeterminantC *sdet)
    char *slater_determinant_to_string_c(SlaterDeterminantC *sdet, char bra_or_ket)
    SlaterDeterminantC *slater_determinant_scalar_multiplication_c(SlaterDeterminantC *sdet, double complex scalar)
    SlaterDeterminantC *slater_determinant_adjoint_c(SlaterDeterminantC *sdet)
    double slater_determinant_comparison_c(SlaterDeterminantC *bra, SlaterDeterminantC *ket)
    double complex slater_determininant_multiplication_c(SlaterDeterminantC *bra, SlaterDeterminantC *ket)
    SlaterDeterminantC *slater_determinant_pauli_string_multiplication_c(PauliStringC *pString, SlaterDeterminantC *sdet)

    cdef struct WavefunctionC:
        unsigned int N
        unsigned int s
        kh_slater_hash_t *slater_determinants
        double cutoff

    WavefunctionC *wavefunction_init_c(unsigned int N);
    WavefunctionC *wavefunction_init_with_specified_cutoff_c(unsigned int N, double cutoff);
    void free_wavefunction_c(WavefunctionC *wfn);
    char *wavefunction_to_string_c(WavefunctionC *wfn, char bra_or_ket);
    double wavefunction_norm_c(WavefunctionC *wfn);
    WavefunctionC *wavefunction_scalar_multiplication_c(WavefunctionC *wfn, double complex scalar);
    WavefunctionC *wavefunction_adjoint_c(WavefunctionC *wfn);
    double complex wavefunction_multiplication_c(WavefunctionC *bra, WavefunctionC *ket);
    void wavefunction_append_slater_determinant_c(WavefunctionC *wfn, SlaterDeterminantC *sdet);
    WavefunctionC *wavefunction_pauli_string_multiplication_c(PauliStringC *pString, WavefunctionC *wfn);
    WavefunctionC *wavefunction_pauli_sum_multiplication_c(PauliSumC *pSum, WavefunctionC *wfn);
    WavefunctionC *wavefunction_pauli_string_evolution_c(PauliStringC *pString, WavefunctionC *wfn, double complex epsilon);
    WavefunctionC *wavefunction_pauli_sum_evolution_c(PauliSumC *pSum, WavefunctionC *wfn, double complex epsilon);
    WavefunctionC *wavefunction_remove_global_phase_c(WavefunctionC *wfn);
    WavefunctionC *wavefunction_remove_near_zero_terms_c(WavefunctionC *wfn, double cutoff);

cdef class SlaterDeterminant:

    def __cinit__(self):
        pass

    def __init__(self, unsigned int N, double complex coef, list orbitals):
        cdef unsigned int *c_orbitals
        
        c_orbitals = <unsigned int *> malloc(N * sizeof(unsigned int))
        if not c_orbitals:
            raise MemoryError("Failed to allocate memory for orbitals array")

        cdef int i
        for i in range(int(N)):
            c_orbitals[i] = orbitals[i]

        cdef SlaterDeterminantC *c_sd = slater_determinant_init_c(N, coef, c_orbitals)
        if not c_sd:
            raise MemoryError("Failed to allocate SlaterDeterminantC")
        self._c_sdet = <uintptr_t> c_sd
        self._in_wfn = False

    def __reduce__(self):
        return (SlaterDeterminant,
                (self.N, self.coef, self.orbitals))

    @staticmethod
    cdef SlaterDeterminant _init_from_c(SlaterDeterminantC* ptr):
        """
        Create a new SlaterDeterminant object wrapping the existing SlaterDeterminantC pointer.
        This bypasses the usual __cinit__ so we don't re-allocate or re-initialize.
        """
        cdef SlaterDeterminant sdet = SlaterDeterminant.__new__(SlaterDeterminant)
        sdet._c_sdet = <uintptr_t> ptr
        sdet._in_wfn = False
        return sdet

    def __dealloc__(self):
        if not self._in_wfn:
            free_slater_determinant_c(<SlaterDeterminantC *> self._c_sdet)

    def __str__(self):
        cdef char *c_str = slater_determinant_to_string_c(<SlaterDeterminantC *> self._c_sdet, b'k')
        py_str = c_str.decode('utf-8')
        free(c_str) 
        return py_str
    
    def __repr__(self): 
        return self.__str__()

    def adjoint(self):
        cdef SlaterDeterminantC *new_sdet = slater_determinant_adjoint_c(<SlaterDeterminantC *> self._c_sdet)
        if not new_sdet:
            raise MemoryError("slater_determinant_adjoint_c returned NULL")
        return SlaterDeterminant._init_from_c(new_sdet)

    @property
    def N(self):
        return (<SlaterDeterminantC *> self._c_sdet).N

    @property
    def coef(self):
        return (<SlaterDeterminantC *> self._c_sdet).coef

    @property
    def orbitals(self):
        cdef unsigned int *c_orbitals = (<SlaterDeterminantC *> self._c_sdet).orbitals
        return [c_orbitals[i] for i in range(self.N)]

    @property
    def encoding(self):
        return (<SlaterDeterminantC *> self._c_sdet).encoding

    def __add__(self, right):
        if isinstance(right, SlaterDeterminant):
            wfn = Wavefunction(self.N)
            wfn.append_slater_determinant(self)
            wfn.append_slater_determinant(right)
            return wfn
        return NotImplemented
    
    def __rmul__(self, left):
        if isinstance(left, numbers.Number):
            return slater_determinant_scalar_multiplication(self, left)
        return NotImplemented


cdef class Wavefunction:
    
    def __cinit__(self):
        pass

    def __init__(self, N, cutoff=1e-8):
        cdef WavefunctionC *c_wfn = wavefunction_init_with_specified_cutoff_c(N, cutoff)
        if not c_wfn:
            raise MemoryError("Failed to allocate WavefunctionC")
        self._c_wfn = <uintptr_t> c_wfn
        self.bra_or_ket = b'k'[0]

    @staticmethod
    cdef Wavefunction _init_from_c(WavefunctionC* ptr):
        """
        Create a new Wavefunction object wrapping the existing WavefunctionC pointer.
        This bypasses the usual __cinit__ so we don't re-allocate or re-initialize.
        """
        cdef Wavefunction wfn = Wavefunction.__new__(Wavefunction)
        wfn._c_wfn = <uintptr_t> ptr
        wfn.bra_or_ket = b'k'[0]
        return wfn

    def __dealloc__(self): 
        free_wavefunction_c(<WavefunctionC *> self._c_wfn)

    def __str__(self):
        cdef char bra_or_ket = self.bra_or_ket
        if not self._c_wfn:
            return "Wavefunction is empty"
        cdef char* c_str = wavefunction_to_string_c(<WavefunctionC *> self._c_wfn, bra_or_ket)
        py_str = c_str.decode('utf-8')
        free(c_str) 
        return py_str

    def __repr__(self): 
        return self.__str__()

    def norm(self):
        return wavefunction_norm_c(<WavefunctionC *> self._c_wfn)

    def append_slater_determinant(self, SlaterDeterminant sdet):
        assert not sdet._in_wfn, "SlaterDeterminant is already part of a Wavefunction"
        sdet._in_wfn = True
        wavefunction_append_slater_determinant_c(<WavefunctionC *> self._c_wfn, <SlaterDeterminantC *> sdet._c_sdet)
        return self

    def adjoint(self):
        cdef WavefunctionC *new_wfn = wavefunction_adjoint_c(<WavefunctionC *> self._c_wfn)
        if not new_wfn:
            raise MemoryError("wavefunction_adjoint_c returned NULL")
        py_wfn = Wavefunction._init_from_c(new_wfn)

        if self.bra_or_ket == b'b'[0]:
            py_wfn.bra_or_ket = b'k'[0]
        else:
            py_wfn.bra_or_ket = b'b'[0]
        return py_wfn

    def remove_global_phase(self):
        cdef WavefunctionC *new_wfn = wavefunction_remove_global_phase_c(<WavefunctionC *> self._c_wfn)
        if not new_wfn:
            raise MemoryError("wavefunction_remove_global_phase_c returned NULL")
        return Wavefunction._init_from_c(new_wfn)

    def remove_near_zero_terms(self, double cutoff):
        cdef WavefunctionC *new_wfn = wavefunction_remove_near_zero_terms_c(<WavefunctionC *> self._c_wfn, cutoff)
        if not new_wfn:
            raise MemoryError("wavefunction_remove_near_zero_terms_c returned NULL")
        return Wavefunction._init_from_c(new_wfn)

    @property
    def s(self):
        return (<WavefunctionC *> self._c_wfn).s

    @property
    def N(self):
        return (<WavefunctionC *> self._c_wfn).N

    def __mul__(self, right):
        if isinstance(right, Wavefunction):
            return wavefunction_multiplication(self, right)
        elif isinstance(right, PauliSum):
            raise TypeError(f"Wavefunction * PauliSum is not defined. Change order of operations.")
        return NotImplemented

    def __rmul__(self, left):
        if isinstance(left, numbers.Number):
            return wavefunction_scalar_multiplication(self, left)
        elif isinstance(left, PauliString):
            return wavefunction_pauli_string_multiplication(left, self)
        elif isinstance(left, PauliSum):
            return wavefunction_pauli_sum_multiplication(left, self)
        return NotImplemented

    def __add__(self, right):
        if isinstance(right, SlaterDeterminant):
            return self.append_slater_determinant(right)
        return NotImplemented
        
def slater_determinant_scalar_multiplication(SlaterDeterminant sdet, complex scalar):
    """Multiply a Slater determinant by a scalar."""
    cdef SlaterDeterminantC *new_sd = slater_determinant_scalar_multiplication_c(<SlaterDeterminantC *> sdet._c_sdet, scalar)
    if not new_sd:
        raise MemoryError("slater_determinant_scalar_multiplication_c returned NULL")
    return SlaterDeterminant._init_from_c(new_sd)

def wavefunction_scalar_multiplication(Wavefunction wfn, complex scalar):
    """Multiply a wavefunction by a scalar."""
    cdef WavefunctionC *new_wfn = wavefunction_scalar_multiplication_c(<WavefunctionC *> wfn._c_wfn, scalar)
    if not new_wfn:
        raise MemoryError("wavefunction_scalar_multiplication_c returned NULL")
    return Wavefunction._init_from_c(new_wfn)  

def wavefunction_multiplication(Wavefunction bra, Wavefunction ket):
    """Compute the inner product of two wavefunctions."""
    return wavefunction_multiplication_c(<WavefunctionC *> bra._c_wfn, <WavefunctionC *> ket._c_wfn)  

def wavefunction_pauli_string_multiplication(PauliString pString, Wavefunction wfn):
    """Apply a Pauli string to a wavefunction."""
    cdef WavefunctionC *new_wfn = wavefunction_pauli_string_multiplication_c(<PauliStringC *> pString._c_pString, <WavefunctionC *> wfn._c_wfn)
    if not new_wfn:
        raise MemoryError("wavefunction_scalar_multiplication_c returned NULL")
    return Wavefunction._init_from_c(new_wfn) 

def wavefunction_pauli_sum_multiplication(PauliSum pSum, Wavefunction wfn):
    """Apply a Pauli sum to a wavefunction."""
    cdef WavefunctionC *new_wfn = wavefunction_pauli_sum_multiplication_c(<PauliSumC *> pSum._c_pSum, <WavefunctionC *> wfn._c_wfn)
    if not new_wfn:
        raise MemoryError("wavefunction_scalar_multiplication_c returned NULL")
    return Wavefunction._init_from_c(new_wfn)  

def wavefunction_pauli_string_evolution(PauliString pString, Wavefunction wfn, double complex epsilon):
    """Evolve a wavefunction under a Pauli string using exponentiation."""
    cdef WavefunctionC *new_wfn = wavefunction_pauli_string_evolution_c(<PauliStringC *> pString._c_pString, <WavefunctionC *> wfn._c_wfn, epsilon)
    if not new_wfn:
        raise MemoryError("wavefunction_scalar_multiplication_c returned NULL")
    return Wavefunction._init_from_c(new_wfn)  

def wavefunction_pauli_sum_evolution(PauliSum pSum, Wavefunction wfn, double complex epsilon):
    """Evolve a wavefunction under a Pauli sum using exponentiation."""
    cdef WavefunctionC *new_wfn = wavefunction_pauli_sum_evolution_c(<PauliSumC *> pSum._c_pSum, <WavefunctionC *> wfn._c_wfn, epsilon)
    if not new_wfn:
        raise MemoryError("wavefunction_scalar_multiplication_c returned NULL")
    return Wavefunction._init_from_c(new_wfn)  

def wavefunction_perform_tomography(Wavefunction wfn, Set[str] measurements):

    tomography = {}

    bra = wfn.adjoint()

    for measurement in measurements:
        s = list(measurement)
        pString = PauliString(len(s), 1, s)
        ket = wavefunction_pauli_string_multiplication(pString, wfn)
        exp_value = wavefunction_multiplication(bra, ket)
        tomography[measurement] = exp_value

    return tomography

def wavefunction_to_probability_distribution(Wavefunction wfn):
    cdef WavefunctionC* c_wfn = <WavefunctionC *> wfn._c_wfn
    cdef khiter_t k
    cdef SlaterDeterminantC* c_sdet
    
    norm = wfn.norm()
    
    if norm == 0:
        return {}

    probabilities = {}
    
    for k in range(kh_begin_slater_hash(c_wfn.slater_determinants), kh_end_slater_hash(c_wfn.slater_determinants)):
        if kh_exist_slater_hash(c_wfn.slater_determinants, k):
            c_sdet = kh_value_slater_hash(c_wfn.slater_determinants, k)
            sdet = SlaterDeterminant._init_from_c(c_sdet)
            sdet._in_wfn = True 
            bit_string = ''.join([str(i) for i in sdet.orbitals])
            prob = abs(c_sdet.coef)**2 / norm**2
            probabilities[bit_string] = prob

    return probabilities