#ifndef DENSITY_MATRIX_H
#define DENSITY_MATRIX_H

#include <complex.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef klib_unused
#undef klib_unused
#endif
#define klib_unused

#include "khash.h"
#include "pauli.h"
#include "wavefunction.h"

typedef struct OuterProductC {
  unsigned int N;      // Number of orbitals
  double complex coef; // Complex coefficient in front of the outer product
  unsigned int
      *ket_orbitals; // Dynamically allocated array for ket orbital occupations
  unsigned int
      *bra_orbitals; // Dynamically allocated array for bra orbital occupations
  uint64_t encoding; // Decimal encoding of bra and ket orbital occupations
                     // for hashing purposes
} OuterProductC;

// Outer product function declarations
OuterProductC *outer_product_init_c(unsigned int N, double complex coef,
                                    unsigned int ket_orbitals[],
                                    unsigned int bra_orbitals[]);

void free_outer_product_c(OuterProductC *oprod);
char *outer_product_to_string_c(OuterProductC *oprod);
OuterProductC *outer_product_scalar_multiplication_c(OuterProductC *oprod,
                                                     double complex scalar);
OuterProductC *outer_product_multiplication(OuterProductC *oprod_left,
                                            OuterProductC *oprod_right);
OuterProductC *
outer_product_pauli_string_left_multiplication_c(PauliStringC *pString,
                                                 OuterProductC *oprod);
OuterProductC *
outer_product_pauli_string_right_multiplication_c(OuterProductC *oprod,
                                                  PauliStringC *pString);

// Struct for outer product khash hashmap
KHASH_MAP_INIT_INT64(outer_product_hash, OuterProductC *)

extern double DENSITYMATRIX_CUTOFF_DEFAULT;

// Struct for a density matrix containing multiple outer products
typedef struct DensityMatrixC {
  unsigned int N; // Number of orbitals
  unsigned int o; // Current number of outer products
  khash_t(outer_product_hash) *
      outer_products; // Hashmap of outer product elements
  double cutoff;      // Cutoff for small coefficients
} DensityMatrixC;

// Density matrix function declarations
DensityMatrixC *density_matrix_init_c(unsigned int N);
DensityMatrixC *density_matrix_init_with_specified_cutoff_c(unsigned int N,
                                                            double cutoff);
DensityMatrixC *density_matrix_from_wavefunction_c(WavefunctionC *wfn);
void free_density_matrix_c(DensityMatrixC *dm);
char *density_matrix_to_string_c(DensityMatrixC *dm);
double density_matrix_trace_c(DensityMatrixC *dm);
DensityMatrixC *density_matrix_scalar_multiplication_c(DensityMatrixC *dm,
                                                       double complex scalar);
DensityMatrixC *density_matrix_multiplication_c(DensityMatrixC *dm_left,
                                                DensityMatrixC *dm_right);
void density_matrix_append_outer_product_c(DensityMatrixC *dm,
                                           OuterProductC *oprod);
DensityMatrixC *
density_matrix_pauli_string_left_multiplication_c(PauliStringC *pString,
                                                  DensityMatrixC *dm);
DensityMatrixC *
density_matrix_pauli_string_right_multiplication_c(DensityMatrixC *dm,
                                                   PauliStringC *pString);
DensityMatrixC *
density_matrix_pauli_sum_left_multiplication_c(PauliSumC *pSum,
                                               DensityMatrixC *dm);
DensityMatrixC *
density_matrix_pauli_sum_right_multiplication_c(DensityMatrixC *dm,
                                                PauliSumC *pSum);
DensityMatrixC *density_matrix_pauli_string_evolution_c(PauliStringC *pString,
                                                        DensityMatrixC *dm,
                                                        double complex epsilon);
DensityMatrixC *density_matrix_pauli_sum_evolution_c(PauliSumC *pSum,
                                                     DensityMatrixC *dm,
                                                     double complex epsilon);
DensityMatrixC *density_matrix_remove_global_phase_c(DensityMatrixC *dm);
DensityMatrixC *density_matrix_remove_near_zero_terms_c(DensityMatrixC *dm,
                                                        double cutoff);
DensityMatrixC *density_matrix_CPTP_evolution_c(PauliSumC *H, PauliSumC **Ls,
                                                unsigned int num_L,
                                                DensityMatrixC *dm, double t);

static inline khiter_t kh_begin_outer_product_hash(khash_t(outer_product_hash) *
                                                   h) {
  (void)h;
  return kh_begin(h);
}

static inline khiter_t kh_end_outer_product_hash(khash_t(outer_product_hash) *
                                                 h) {
  (void)h;
  return kh_end(h);
}

static inline int kh_exist_outer_product_hash(khash_t(outer_product_hash) * h,
                                              khiter_t k) {
  return kh_exist(h, k);
}

static inline OuterProductC *
kh_value_outer_product_hash(khash_t(outer_product_hash) * h, khiter_t k) {
  return kh_value(h, k);
}

#endif // DENSITY_MATRIX_H