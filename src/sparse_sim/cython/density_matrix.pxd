from libc.stdint cimport uintptr_t, uint64_t

cdef extern from "khash.h":
    ctypedef int khiter_t

    ctypedef struct kh_outer_product_hash_t:
        int n_buckets
        int size
        int n_occupied
        int upper_bound
        unsigned int *keys 
        char *flags     
        OuterProductC **vals 

        khiter_t kh_begin_outer_product_hash(kh_outer_product_hash_t *h)
        khiter_t kh_end_outer_product_hash(kh_outer_product_hash_t *h)
        int kh_exist_outer_product_hash(kh_outer_product_hash_t *h, khiter_t k)
        OuterProductC *kh_value_outer_product_hash(kh_outer_product_hash_t *h, khiter_t k)

cdef extern from "density_matrix.h":
    cdef struct OuterProductC:
        unsigned int N
        double complex coef
        unsigned int *ket_orbitals
        unsigned int *bra_orbitals
        uint64_t encoding

    cdef struct DensityMatrixC:
        unsigned int N
        unsigned int o
        kh_outer_product_hash_t *outer_products
        double cutoff


cdef class OuterProduct:
    cdef uintptr_t _c_oprod
    cdef bint _in_dm
    @staticmethod
    cdef OuterProduct _init_from_c(OuterProductC* ptr)

cdef class DensityMatrix:
    cdef uintptr_t _c_dm

    @staticmethod
    cdef DensityMatrix _init_from_c(DensityMatrixC* ptr)