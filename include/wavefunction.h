#ifndef WAVEFUNCTION_H
#define WAVEFUNCTION_H

#include <complex.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "pauli.h"

// Struct for a single SlaterDeterminant
typedef struct SlaterDeterminantC {
    unsigned int N; // Number of orbitals
    double complex coef; // Complex coefficient in front of the slater determinant
    unsigned int *orbitals; // Dynamically allocated array for orbital occupations
    unsigned int encoding; // Decimal encoding of orbital occupations for indexing purposes
} SlaterDeterminantC;

// Function declarations
SlaterDeterminantC *slater_determinant_init_c(unsigned int N, double complex coef, unsigned int orbitals[]);
void free_slater_determinant_c(SlaterDeterminantC *sdet);
char *slater_determinant_to_string_c(SlaterDeterminantC *sdet, char bra_or_ket);
SlaterDeterminantC *slater_determinant_scalar_multiplication_c(SlaterDeterminantC *sdet, double complex scalar);
SlaterDeterminantC *slater_determinant_adjoint_c(SlaterDeterminantC *sdet);
double slater_determinant_comparison_c(SlaterDeterminantC *bra, SlaterDeterminantC *ket);
double complex slater_dermininant_multiplication_c(SlaterDeterminantC *bra, SlaterDeterminantC *ket);
SlaterDeterminantC *slater_determinant_pauli_string_multiplication_c(PauliStringC *pString, SlaterDeterminantC *sdet);

// Struct for a Wavefunction containing multiple SlaterDeterminants
typedef struct WavefunctionC {
    unsigned int s_max; // Maximum number of slater determinants
    unsigned int s; // Current number of slater determinants
    SlaterDeterminantC **slater_determinants; // Array of Slater Determinant elements
} WavefunctionC;

WavefunctionC *wavefunction_init_c(unsigned int s_max);
void free_wavefunction_c(WavefunctionC *wfn);
char *wavefunction_to_string_c(WavefunctionC *wfn, char bra_or_ket);
double wavefunction_norm_c(WavefunctionC *wfn);
WavefunctionC *wavefunction_scalar_multiplication_c(WavefunctionC *wfn, double complex scalar);
WavefunctionC *wavefunction_adjoint_c(WavefunctionC *wfn);
double complex wavefunction_multiplication_c(WavefunctionC *bra, WavefunctionC *ket);
WavefunctionC *wavefunction_realloc_c(WavefunctionC *wfn, unsigned int new_s_max);
WavefunctionC *wavefunction_append_slater_determinant_c(WavefunctionC *wfn, SlaterDeterminantC *sdet);
WavefunctionC *wavefunction_pauli_string_multiplication_c(PauliStringC *pString, WavefunctionC *wfn);
WavefunctionC *wavefunction_pauli_sum_multiplication_c(PauliSumC *pSum, WavefunctionC *wfn);
WavefunctionC *wavefunction_pauli_string_evolution_c(PauliStringC *pString, WavefunctionC *wfn, double epsilon);
WavefunctionC *wavefunction_pauli_sum_evolution_c(PauliSumC *pSum, WavefunctionC *wfn, double epsilon);

#endif // WAVEFUNCTION_H
