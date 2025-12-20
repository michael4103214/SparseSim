from libc.stdint cimport uintptr_t, uint64_t
from libc.stdlib cimport free

import numbers
import numpy as np
from typing import Set

cdef extern from "khash.h":
    
    ctypedef int khiter_t

    # Hash table type for PauliSumC 
    ctypedef struct kh_pauli_hash_t:
        int n_buckets
        int size
        int n_occupied
        int upper_bound
        uint64_t *keys 
        char *flags         
        PauliStringC **vals 

    # Function declarations for handling khash
    kh_pauli_hash_t *kh_init_pauli_hash()
    void kh_destroy_pauli_hash(kh_pauli_hash_t *h)
    khiter_t kh_get_pauli_hash(kh_pauli_hash_t *h, uint64_t key)
    khiter_t kh_put_pauli_hash(kh_pauli_hash_t *h, uint64_t key, int *ret)
    int kh_exist_pauli_hash(kh_pauli_hash_t *h, khiter_t k)
    void kh_del_pauli_hash(kh_pauli_hash_t *h, khiter_t k)
    PauliStringC *kh_value_pauli_hash(kh_pauli_hash_t *h, khiter_t k)

    # Declare hash iteration functions
    khiter_t kh_begin_pauli_hash(kh_pauli_hash_t *h)
    khiter_t kh_end_pauli_hash(kh_pauli_hash_t *h)

cdef extern from "pauli.h":

    cdef struct PauliStringC :
        unsigned int N
        double complex coef
        unsigned int *paulis
        uint64_t encoding

    PauliStringC *pauli_string_init_as_chars_c(unsigned int N, double complex coef, char paulis[])
    PauliStringC *pauli_string_init_as_ints_c(unsigned int N, double complex coef, unsigned int paulis[])
    void free_pauli_string_c(PauliStringC *pString)
    char *pauli_string_to_string_no_coef_c(PauliStringC *pString)
    char *pauli_string_to_string_c(PauliStringC *pString)
    PauliStringC *pauli_string_scalar_multiplication_c(PauliStringC *pString, double complex scalar)
    PauliStringC *pauli_string_adjoint_c(PauliStringC *pString)
    double pauli_string_comparison_c(PauliStringC *left, PauliStringC *right)
    PauliStringC *pauli_string_multiplication_c(PauliStringC *left, PauliStringC *right)
    PauliStringC **get_pauli_strings_c(PauliSumC *pSum)

    cdef struct PauliSumC:
        unsigned int N
        unsigned int p
        kh_pauli_hash_t *pauli_strings
        double cutoff

    PauliSumC *pauli_sum_init_c(unsigned int N)
    PauliSumC *pauli_sum_init_with_specified_cutoff_c(unsigned int N, double cutoff)
    void free_pauli_sum_c(PauliSumC *pSum)
    char *pauli_sum_to_string_c(PauliSumC *pSum)
    void pauli_sum_append_pauli_string_c(PauliSumC *pSum, PauliStringC *pString)
    PauliSumC *pauli_sum_scalar_multiplication_c(PauliSumC *pSum, double complex scalar)
    PauliSumC *pauli_sum_adjoint_c(PauliSumC *pSum)
    PauliSumC *pauli_sum_multiplication_c(PauliSumC *left, PauliSumC *right)
    PauliSumC *pauli_sum_addition_c(PauliSumC *left, PauliSumC *right)

cdef class PauliString:

    def __cinit__(self):
        pass

    def __init__(self, unsigned int N, double complex coef, list paulis):
        if any(len(p) != 1 for p in paulis):
            raise ValueError("Each Pauli character must be a single letter (e.g., ['X', 'Y', 'Z'])")

        pauli_bytes = "".join(paulis).encode("utf-8") 
        cdef const char *c_paulis = pauli_bytes 

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

    def __repr__(self): 
        return self.__str__()

    def adjoint(self):
        cdef PauliStringC *new_pString = pauli_string_adjoint_c(<PauliStringC *> self._c_pString)
        py_pString = PauliString._init_from_c(new_pString)
        py_pString._in_sum = False
        return py_pString

    @property
    def N(self):
        return (<PauliStringC *> self._c_pString).N

    @property
    def coef(self):
        return (<PauliStringC *> self._c_pString).coef

    @property
    def string(self):
        cdef char *c_str = pauli_string_to_string_no_coef_c(<PauliStringC *> self._c_pString)
        py_str = c_str.decode('utf-8')
        free(c_str) 
        return py_str
    
    @property
    def matrix(self):
        I = np.eye(2)
        X = np.array([[0, 1], [1, 0]])
        Y = np.array([[0, -1j], [1j, 0]])
        Z = np.array([[1, 0], [0, -1]])

        py_str = self.string
        as_matrix = np.array([[self.coef]])
        for i in range(self.N):
            if py_str[i] == 'I':
                as_matrix = np.kron(as_matrix, I)
            elif py_str[i] == 'X':
                as_matrix = np.kron(as_matrix, X)
            elif py_str[i] == 'Y':
                as_matrix = np.kron(as_matrix, Y)
            elif py_str[i] == 'Z':
                as_matrix = np.kron(as_matrix, Z)
            else:
                raise ValueError(f"Invalid Pauli character: {py_str[i]}") 

        return as_matrix

    def __mul__(self, right):
        if isinstance(right, PauliString):
            return pauli_string_multiplication(self, right)
        return NotImplemented

    def __rmul__(self, left):
        if isinstance(left, numbers.Number):
            return pauli_string_scalar_multiplication(self, left)
        return NotImplemented

    def __add__(self, right):
        if isinstance(right, PauliString):
            pSum = PauliSum(self.N)
            pSum.append_pauli_string(self)
            pSum.append_pauli_string(right)
            return pSum
        return NotImplemented


cdef class PauliSum:

    def __cinit__(self):
        pass

    def __init__(self, N, cutoff=1e-12):
        cdef PauliSumC *c_pSum = pauli_sum_init_with_specified_cutoff_c(N, cutoff)
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

    def __repr__(self):
        return self.__str__()

    def append_pauli_string(self, PauliString pString):
        assert not pString._in_sum, "PauliString is already part of a PauliSum"
        pString._in_sum = True
        pauli_sum_append_pauli_string_c(<PauliSumC *> self._c_pSum, <PauliStringC *> pString._c_pString)
        return self

    def adjoint(self):
        cdef PauliSumC *new_pSum = pauli_sum_adjoint_c(<PauliSumC *> self._c_pSum)
        pSum = PauliSum._init_from_c(new_pSum)
        return pSum

    def get_pauli_strings(self):
        cdef PauliStringC **c_pStrings = get_pauli_strings_c(<PauliSumC *> self._c_pSum)
        py_pStrings = []
        for i in range(self.p) :
            py_pStrings.append(PauliString._init_from_c(c_pStrings[i]))
        free(c_pStrings)
        return py_pStrings

    @property
    def p(self):
        return (<PauliSumC *> self._c_pSum).p

    @property
    def N(self):
        return (<PauliSumC *> self._c_pSum).N

    @property
    def matrix(self):
        py_pStrings = self.get_pauli_strings()

        as_matrix = np.zeros((2**self.N, 2**self.N), dtype=complex)
        for pString in py_pStrings:
            as_matrix += pString.matrix
        return as_matrix

    def __mul__(self, right):
        if isinstance(right, PauliSum):
            return pauli_sum_multiplication(self, right)
        return NotImplemented

    def __rmul__(self, left):
        if isinstance(left, numbers.Number):
            return pauli_sum_scalar_multiplication(self, left)
        return NotImplemented

    def __add__(self, right):
        if isinstance(right, PauliSum):
            return pauli_sum_addition(self, right)
        return NotImplemented


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

    for k in range(kh_begin_pauli_hash(c_pSum.pauli_strings), kh_end_pauli_hash(c_pSum.pauli_strings)):
        if kh_exist_pauli_hash(c_pSum.pauli_strings, k):
            c_pString = kh_value_pauli_hash(c_pSum.pauli_strings, k)

            c_str = pauli_string_to_string_no_coef_c(c_pString)
            py_str = c_str.decode('utf-8')
            free(c_str)

            unique_strings.add(py_str)

    return unique_strings

def pauli_sum_evaluate_expectation(PauliSum pSum, dict tomography):
    cdef PauliSumC* c_pSum = <PauliSumC *> pSum._c_pSum
    cdef PauliStringC* c_pString
    cdef char* c_str
    cdef khiter_t k

    exp = 0 + 0j

    for k in range(kh_begin_pauli_hash(c_pSum.pauli_strings), kh_end_pauli_hash(c_pSum.pauli_strings)):
        if kh_exist_pauli_hash(c_pSum.pauli_strings, k):
            c_pString = kh_value_pauli_hash(c_pSum.pauli_strings, k)

            c_str = pauli_string_to_string_no_coef_c(c_pString)
            py_str = c_str.decode('utf-8')
            free(c_str)

            if py_str in tomography:
                coef = c_pString.coef
                exp += coef * tomography[py_str] 
                #print(f"{py_str}: {coef} * {tomography[py_str]}") 
            else:
                print(f"Error: PauliString '{py_str}' not found in tomography data")

    return exp

def pauli_sum_addition(PauliSum left, PauliSum right):
    """Add two Pauli sums."""
    cdef PauliSumC *new_pSum = pauli_sum_addition_c(<PauliSumC *> left._c_pSum, <PauliSumC *> right._c_pSum)
    if not new_pSum:
        raise MemoryError("pauli_sum_addition_c returned NULL")
    return PauliSum._init_from_c(new_pSum) 