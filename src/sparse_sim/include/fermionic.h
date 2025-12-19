#ifndef FERMIONIC_H
#define FERMIONIC_H

#include <complex.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "wavefunction.h"

typedef struct FermionicOperatorC {
  unsigned int N;         // number of orbitals spinless
  char *opstring;         // string representation of the fermionic operator
  unsigned int *ordering; // notation ordering of the fermionic operators
  double complex *coefs;  // complex coefficients for each term
} FermionicOperatorC;

FermionicOperatorC *fermionic_operator_init_c(unsigned int N, char opstring[],
                                              unsigned int ordering[],
                                              double complex *coefs);

void free_fermionic_operator_c(FermionicOperatorC *fop);

char *fermionic_operator_to_string_c(FermionicOperatorC *fop);

FermionicOperatorC *
fermionic_operator_scalar_multiplication_c(FermionicOperatorC *fop,
                                           double complex scalar);

FermionicOperatorC *fermionic_operator_adjoint_c(FermionicOperatorC *fop);

SlaterDeterminantC *
slater_determinant_single_term_fermionic_operator_multiplication_c(
    SlaterDeterminantC *sdet, char opstring[], unsigned int unordered_idx[],
    double complex coef);

WavefunctionC *wavefunction_single_term_fermionic_operator_multiplication_c(
    WavefunctionC *wfn, char opstring[], unsigned int unordered_idx[],
    double complex coef);

FermionicOperatorC *
wavefunction_termwise_fermionic_operator_expectation_c(WavefunctionC *wfn,
                                                       FermionicOperatorC *fop);

double complex wavefunction_fermionic_operator_expectation_c(
    WavefunctionC *wfn, FermionicOperatorC *fop);
#endif // FERMIONIC_H