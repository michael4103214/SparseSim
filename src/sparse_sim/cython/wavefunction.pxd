from libc.stdint cimport uintptr_t, uint64_t

cdef extern from "khash.h":
    ctypedef int khiter_t

    ctypedef struct kh_slater_hash_t:
        int n_buckets
        int size
        int n_occupied
        int upper_bound
        uint64_t *keys
        char *flags
        SlaterDeterminantC **vals

    khiter_t kh_begin_slater_hash(kh_slater_hash_t *h)
    khiter_t kh_end_slater_hash(kh_slater_hash_t *h)
    int kh_exist_slater_hash(kh_slater_hash_t *h, khiter_t k)
    SlaterDeterminantC *kh_value_slater_hash(kh_slater_hash_t *h, khiter_t k)

cdef extern from "wavefunction.h":
    cdef struct SlaterDeterminantC:
        unsigned int N
        double complex coef
        unsigned int *orbitals
        uint64_t encoding

    cdef struct WavefunctionC:
        unsigned int N
        unsigned int s
        kh_slater_hash_t *slater_determinants
        double cutoff

cdef class SlaterDeterminant:
    cdef uintptr_t _c_sdet
    cdef bint _in_wfn
    @staticmethod
    cdef SlaterDeterminant _init_from_c(SlaterDeterminantC* ptr)

cdef class Wavefunction:
    cdef uintptr_t _c_wfn
    cdef char bra_or_ket
    @staticmethod
    cdef Wavefunction _init_from_c(WavefunctionC* ptr)

