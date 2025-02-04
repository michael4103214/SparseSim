#ifndef PAULI_H
#define PAULI_H

#include <complex.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct PauliStringC {
  unsigned int N;       // Number of qubits
  double complex coef;  // Coefficient in front of the Pauli string
  unsigned int *paulis; // Dynamically allocated array for Pauli operators
} PauliStringC;

PauliStringC *pauli_string_init_as_chars_c(unsigned int N, double complex coef,
                                           char paulis[]);
PauliStringC *pauli_string_init_as_ints_c(unsigned int N, double complex coef,
                                          unsigned int paulis[]);
void free_pauli_string_c(PauliStringC *pString);
char *pauli_string_to_string_c(PauliStringC *pString);
PauliStringC *pauli_string_scalar_multiplication_c(PauliStringC *pString,
                                                   double complex scalar);
PauliStringC *pauli_string_adjoint_c(PauliStringC *pString);
double pauli_string_comparison_c(PauliStringC *left, PauliStringC *right);
PauliStringC *pauli_string_multiplication_c(PauliStringC *left,
                                            PauliStringC *right);

typedef struct PauliSumC {
  unsigned int p_max;           // Maximum number of Pauli strings
  unsigned int p;               // Current number of Pauli strings
  PauliStringC **pauli_strings; // Array of Pauli strings
} PauliSumC;

PauliSumC *pauli_sum_init_c(unsigned int p_max);
void free_pauli_sum_c(PauliSumC *pSum);
PauliSumC *pauli_sum_realloc_c(PauliSumC *pSum, unsigned int new_p_max);
char *pauli_sum_to_string_c(PauliSumC *pSum);
PauliSumC *pauli_sum_append_pauli_string_c(PauliSumC *pSum,
                                           PauliStringC *pString);
PauliSumC *pauli_sum_scalar_multiplication_c(PauliSumC *pSum,
                                             complex double scalar);
PauliSumC *pauli_sum_adjoint_c(PauliSumC *pSum);
PauliSumC *pauli_sum_multiplication_c(PauliSumC *left, PauliSumC *right);

#endif // PAULI_H
