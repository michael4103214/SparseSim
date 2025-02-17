from libc.stdlib cimport malloc, free
from libc.stdint cimport uintptr_t
import numbers
from typing import Set

cdef extern from "khash.h":
    
    ctypedef int khiter_t

    # Hash table struct type for SlaterDeterminantC
    ctypedef struct kh_slater_hash_t:
        int n_buckets
        int size
        int n_occupied
        int upper_bound
        unsigned int *keys 
        char *flags     
        SlaterDeterminantC **vals 

    # Declare khash functions
    kh_slater_hash_t *kh_init_slater_hash()
    void kh_destroy_slater_hash(kh_slater_hash_t *h)
    khiter_t kh_get_slater_hash(kh_slater_hash_t *h, unsigned int key)
    khiter_t kh_put_slater_hash(kh_slater_hash_t *h, unsigned int key, int *ret)
    int kh_exist(kh_slater_hash_t *h, khiter_t k)
    void kh_del_slater_hash(kh_slater_hash_t *h, khiter_t k)
    SlaterDeterminantC *kh_value(kh_slater_hash_t *h, khiter_t k)

    # Iteration functions
    khiter_t kh_begin(kh_slater_hash_t *h)
    khiter_t kh_end(kh_slater_hash_t *h)

    # Hash table type for PauliSumC 
    ctypedef struct kh_pauli_hash_t:
        int n_buckets
        int size
        int n_occupied
        int upper_bound
        unsigned int *keys 
        char *flags         
        PauliStringC **vals 

    # Function declarations for handling khash
    kh_pauli_hash_t *kh_init_pauli_hash()
    void kh_destroy_pauli_hash(kh_pauli_hash_t *h)
    khiter_t kh_get_pauli_hash(kh_pauli_hash_t *h, unsigned int key)
    khiter_t kh_put_pauli_hash(kh_pauli_hash_t *h, unsigned int key, int *ret)
    int kh_exist(kh_pauli_hash_t *h, khiter_t k)
    void kh_del_pauli_hash(kh_pauli_hash_t *h, khiter_t k)
    PauliStringC *kh_value(kh_pauli_hash_t *h, khiter_t k)

    # Declare hash iteration functions
    khiter_t kh_begin(kh_pauli_hash_t *h)
    khiter_t kh_end(kh_pauli_hash_t *h)

cdef extern from "wavefunction.h":

    cdef struct SlaterDeterminantC:
            unsigned int N
            double complex coef
            unsigned int *orbitals
            unsigned int encoding

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

    WavefunctionC *wavefunction_init_c(unsigned int N);
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

cdef extern from "pauli.h":

    cdef struct PauliStringC :
        unsigned int N
        double complex coef
        unsigned int *paulis

    PauliStringC *pauli_string_init_as_chars_c(unsigned int N, double complex coef, char paulis[])
    PauliStringC *pauli_string_init_as_ints_c(unsigned int N, double complex coef, unsigned int paulis[])
    void free_pauli_string_c(PauliStringC *pString)
    char *pauli_string_to_string_no_coef_c(PauliStringC *pString)
    char *pauli_string_to_string_c(PauliStringC *pString)
    PauliStringC *pauli_string_scalar_multiplication_c(PauliStringC *pString, double complex scalar)
    PauliStringC *pauli_string_adjoint_c(PauliStringC *pString)
    double pauli_string_comparison_c(PauliStringC *left, PauliStringC *right)
    PauliStringC *pauli_string_multiplication_c(PauliStringC *left, PauliStringC *right)

    cdef struct PauliSumC:
        unsigned int N
        unsigned int p
        kh_pauli_hash_t *pauli_strings

    PauliSumC *pauli_sum_init_c(unsigned int N)
    void free_pauli_sum_c(PauliSumC *pSum)
    char *pauli_sum_to_string_c(PauliSumC *pSum)
    void pauli_sum_append_pauli_string_c(PauliSumC *pSum, PauliStringC *pString)
    PauliSumC *pauli_sum_scalar_multiplication_c(PauliSumC *pSum, double complex scalar)
    PauliSumC *pauli_sum_adjoint_c(PauliSumC *pSum)
    PauliSumC *pauli_sum_multiplication_c(PauliSumC *left, PauliSumC *right)
    PauliSumC *pauli_sum_addition_c(PauliSumC *left, PauliSumC *right)

cdef class SlaterDeterminant:
    cdef uintptr_t _c_sd
    cdef bint _in_wfn

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
        self._c_sd = <uintptr_t> c_sd
        self._in_wfn = False

    @staticmethod
    cdef SlaterDeterminant _init_from_c(SlaterDeterminantC* ptr):
        """
        Create a new SlaterDeterminant object wrapping the existing SlaterDeterminantC pointer.
        This bypasses the usual __cinit__ so we don't re-allocate or re-initialize.
        """
        cdef SlaterDeterminant sDet = SlaterDeterminant.__new__(SlaterDeterminant)
        sDet._c_sDet = <uintptr_t> ptr
        sDet._in_wfn = False
        return sDet

    def __dealloc__(self):
        if not self._in_wfn:
            free_slater_determinant_c(<SlaterDeterminantC *> self._c_sd)

    def __str__(self):
        cdef char *c_str = slater_determinant_to_string_c(<SlaterDeterminantC *> self._c_sd, b'k')
        py_str = c_str.decode('utf-8')
        free(c_str) 
        return py_str

    @property
    def N(self):
        return (<SlaterDeterminantC *> self._c_sd).N

cdef class Wavefunction:
    cdef uintptr_t _c_wfn
    cdef char bra_or_ket

    def __cinit__(self):
        pass

    def __init__(self, N):
        cdef WavefunctionC *c_wfn = wavefunction_init_c(N)
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

    def norm(self):
        return wavefunction_norm_c(<WavefunctionC *> self._c_wfn)

    def append_slater_determinant(self, SlaterDeterminant sdet):
        sdet._in_wfn = True
        wavefunction_append_slater_determinant_c(<WavefunctionC *> self._c_wfn, <SlaterDeterminantC *> sdet._c_sd)

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

    @property
    def s(self):
        return (<WavefunctionC *> self._c_wfn).s

    def __mul__(self, right):
        if isinstance(right, Wavefunction):
            return wavefunction_multiplication(self, right)
        else:
            raise TypeError(f"Wavefunction * {type(right)} is not defined")

    def __rmul__(self, left):
        if isinstance(left, numbers.Number):
            return wavefunction_scalar_multiplication(self, left)
        else:
            raise TypeError(f"{type(left)} * Wavefunction is not defined")

    @property
    def N(self):
        return (<WavefunctionC *> self._c_wfn).N
        
cdef class PauliString:
    cdef uintptr_t _c_pString
    cdef bint _in_sum

    def __cinit__(self):
        pass

    def __init__(self, unsigned int N, double complex coef, list paulis):
        if any(len(p) != 1 for p in paulis):
            raise ValueError("Each Pauli character must be a single letter (e.g., ['X', 'Y', 'Z'])")

        pauli_bytes = "".join(paulis).encode("utf-8") 
        cdef char *c_paulis = pauli_bytes 

        cdef PauliStringC *c_pString = pauli_string_init_as_chars_c(N, coef, c_paulis)
        if not c_pString:
            raise MemoryError("Failed to allocate PauliStringC")

        self._c_pString = <uintptr_t> c_pString
        self._in_sum = False

    @staticmethod
    cdef PauliString _init_from_c(PauliStringC* ptr):
        """
        Create a new PauliString object wrapping the existing PauliStringC pointer.
        This bypasses the usual __cinit__ so we don't re-allocate or re-initialize.
        """
        cdef PauliString pString = PauliString.__new__(PauliString)
        pString._c_pString = <uintptr_t> ptr
        pString._in_sum = False
        return pString

    def __dealloc__(self):
        if not self._in_sum:
            free_pauli_string_c(<PauliStringC *> self._c_pString)

    def __str__(self):
        cdef char *c_str = pauli_string_to_string_c(<PauliStringC *> self._c_pString)
        py_str = c_str.decode('utf-8')
        free(c_str) 
        return py_str

    def adjoint(self):
        cdef PauliStringC *new_pString = pauli_string_adjoint_c(<PauliStringC *> self._c_pString)
        py_pString = PauliString._init_from_c(new_pString)
        py_pString._in_sum = False
        return py_pString

    def __mul__(self, right):
        if isinstance(right, Wavefunction):
            return wavefunction_pauli_string_multiplication(self, right)
        elif isinstance(right, PauliString):
            return pauli_string_multiplication(self, right)
        else:
            raise TypeError(f"pString * {type(right)} is not defined")

    def __rmul__(self, left):
        if isinstance(left, numbers.Number):
            return pauli_string_scalar_multiplication(self, left)
        else:
            raise TypeError(f"{type(left)} * pString is not defined")

    @property
    def N(self):
        return (<PauliStringC *> self._c_pString).N

cdef class PauliSum:
    cdef uintptr_t _c_pSum

    def __cinit__(self):
        pass

    def __init__(self, N):
        cdef PauliSumC *c_pSum = pauli_sum_init_c(N)
        if not c_pSum:
            raise MemoryError("Failed to allocate PauliSum")
        self._c_pSum = <uintptr_t> c_pSum

    @staticmethod
    cdef PauliSum _init_from_c(PauliSumC* ptr):
        """
        Create a new PauliSum object wrapping the existing PauliSumC pointer.
        This bypasses the usual __cinit__ so we don't re-allocate or re-initialize.
        """
        cdef PauliSum pSum = PauliSum.__new__(PauliSum)
        pSum._c_pSum = <uintptr_t> ptr
        return pSum

    def __dealloc__(self):
        free_pauli_sum_c(<PauliSumC *> self._c_pSum)

    def __str__(self):
        cdef char *c_str = pauli_sum_to_string_c(<PauliSumC *> self._c_pSum)
        py_str = c_str.decode('utf-8')
        free(c_str) 
        return py_str

    def append_pauli_string(self, PauliString pString):
        pString._in_sum = True
        pauli_sum_append_pauli_string_c(<PauliSumC *> self._c_pSum, <PauliStringC *> pString._c_pString)

    def adjoint(self):
        cdef PauliSumC *new_pSum = pauli_sum_adjoint_c(<PauliSumC *> self._c_pSum)
        pSum = PauliSum._init_from_c(new_pSum)
        return pSum

    @property
    def p(self):
        return (<PauliSumC *> self._c_pSum).p

    def __mul__(self, right):
        if isinstance(right, Wavefunction):
            return wavefunction_pauli_sum_multiplication(self, right)
        elif isinstance(right, PauliSum):
            return pauli_sum_multiplication(self, right)
        else:
            raise TypeError(f"pSum * {type(right)} is not defined")

    def __rmul__(self, left):
        if isinstance(left, numbers.Number):
            return pauli_sum_scalar_multiplication(self, left)
        else:
            raise TypeError(f"{type(left)} * pSum is not defined")

    def __add__(self, right):
        if isinstance(right, PauliSum):
            return pauli_sum_addition(self, right)
        else:
            raise TypeError(f"pSum + {type(right)} is not defined")

    @property
    def N(self):
        return (<PauliSumC *> self._c_pSum).N

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

def pauli_string_scalar_multiplication(PauliString pString, complex scalar):
    """Multiply a Pauli string by a scalar."""
    cdef PauliStringC *new_pString = pauli_string_scalar_multiplication_c(<PauliStringC *> pString._c_pString, scalar)
    if not new_pString:
        raise MemoryError("pauli_string_scalar_multiplication_c returned NULL")
    return PauliString._init_from_c(new_pString)  

def pauli_string_multiplication(PauliString left, PauliString right):
    """Multiply two Pauli strings."""
    cdef PauliStringC *new_pString = pauli_string_multiplication_c(<PauliStringC *> left._c_pString, <PauliStringC *> right._c_pString)
    return PauliString._init_from_c(new_pString)  

def pauli_sum_scalar_multiplication(PauliSum pSum, complex scalar):
    """Multiply a Pauli sum by a scalar."""
    cdef PauliSumC *new_pSum = pauli_sum_scalar_multiplication_c(<PauliSumC *> pSum._c_pSum, scalar)
    if not new_pSum:
        raise MemoryError("pauli_sum_scalar_multiplication_c returned NULL")
    return PauliSum._init_from_c(new_pSum)  
    
def pauli_sum_multiplication(PauliSum left, PauliSum right):
    """Multiply two Pauli sums."""
    cdef PauliSumC *new_pSum = pauli_sum_multiplication_c(<PauliSumC *> left._c_pSum, <PauliSumC *> right._c_pSum)
    if not new_pSum:
        raise MemoryError("pauli_sum_multiplication_c returned NULL")
    return PauliSum._init_from_c(new_pSum) 

def pauli_sum_collect_measurements(PauliSum pSum):
    cdef PauliSumC* c_pSum = <PauliSumC *> pSum._c_pSum
    cdef PauliStringC* c_pString
    cdef char* c_str
    cdef khiter_t k

    unique_strings = set()

    for k in range(kh_begin(c_pSum.pauli_strings), kh_end(c_pSum.pauli_strings)):
        if kh_exist(c_pSum.pauli_strings, k):
            c_pString = kh_value(c_pSum.pauli_strings, k)

            c_str = pauli_string_to_string_no_coef_c(c_pString)
            py_str = c_str.decode('utf-8')
            free(c_str)

            unique_strings.add(py_str)

    return unique_strings

def measurements_calculate_tomography(Set[str] measurements, Wavefunction wfn):
    cdef WavefunctionC* c_wfn = <WavefunctionC *> wfn._c_wfn
    cdef PauliStringC* c_pString
    cdef char* c_str
    cdef khiter_t k

    tomography = {}

    bra = wfn.adjoint()

    for measurement in measurements:
        s = list(measurement)
        pString = PauliString(len(s), 1, s)
        ket = wavefunction_pauli_string_multiplication(pString, wfn)
        exp_value = wavefunction_multiplication(bra, ket)
        tomography[measurement] = exp_value

    return tomography

def pauli_sum_evaluate_expectation(PauliSum pSum, dict tomography):
    cdef PauliSumC* c_pSum = <PauliSumC *> pSum._c_pSum
    cdef PauliStringC* c_pString
    cdef char* c_str
    cdef khiter_t k

    exp = 0 + 0j

    for k in range(kh_begin(c_pSum.pauli_strings), kh_end(c_pSum.pauli_strings)):
        if kh_exist(c_pSum.pauli_strings, k):
            c_pString = kh_value(c_pSum.pauli_strings, k)

            c_str = pauli_string_to_string_no_coef_c(c_pString)
            py_str = c_str.decode('utf-8')
            free(c_str)

            if py_str in tomography:
                coef = c_pString.coef
                exp += coef * tomography[py_str]  
            else:
                print(f"Error: PauliString '{py_str}' not found in tomography data")

    return exp

def pauli_sum_addition(PauliSum left, PauliSum right):
    """Add two Pauli sums."""
    cdef PauliSumC *new_pSum = pauli_sum_addition_c(<PauliSumC *> left._c_pSum, <PauliSumC *> right._c_pSum)
    if not new_pSum:
        raise MemoryError("pauli_sum_addition_c returned NULL")
    return PauliSum._init_from_c(new_pSum) 



