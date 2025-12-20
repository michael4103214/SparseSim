from libc.stdint cimport uintptr_t, uint64_t

cdef extern from "khash.h":
    ctypedef int khiter_t

    ctypedef struct kh_pauli_hash_t:
        int n_buckets
        int size
        int n_occupied
        int upper_bound
        uint64_t *keys
        char *flags
        PauliStringC **vals

    khiter_t kh_begin_pauli_hash(kh_pauli_hash_t *h)
    khiter_t kh_end_pauli_hash(kh_pauli_hash_t *h)
    int kh_exist_pauli_hash(kh_pauli_hash_t *h, khiter_t k)
    PauliStringC *kh_value_pauli_hash(kh_pauli_hash_t *h, khiter_t k)

cdef extern from "pauli.h":
    cdef struct PauliStringC:
        unsigned int N
        double complex coef
        unsigned int *paulis
        uint64_t encoding

    cdef struct PauliSumC:
        unsigned int N
        unsigned int p
        kh_pauli_hash_t *pauli_strings
        double cutoff 

cdef class PauliString:
    cdef uintptr_t _c_pString
    cdef bint _in_sum
    @staticmethod
    cdef PauliString _init_from_c(PauliStringC* ptr)
    

cdef class PauliSum:
    cdef uintptr_t _c_pSum
    @staticmethod
    cdef PauliSum _init_from_c(PauliSumC* ptr)