#include <complex.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "wavefunction.h"

void test_initialization_scaling_freeing(void) {

  SlaterDeterminantC *sdet0 =
      slater_determinant_init_c(4, 1, (unsigned int[]){0, 1, 0, 1});
  SlaterDeterminantC *sdet1 = slater_determinant_init_c(
      4, (double complex)I, (unsigned int[]){1, 0, 1, 0});

  WavefunctionC *wfn = wavefunction_init_c(4);
  printf("Wavefunction initialized at %p with sdets: %u, %u\n", (void *)wfn,
         sdet0->encoding, sdet1->encoding);

  wavefunction_append_slater_determinant_c(wfn, sdet0);
  wavefunction_append_slater_determinant_c(wfn, sdet1);

  char *wfn_str = wavefunction_to_string_c(wfn, 'k');
  printf("%s\n", wfn_str);
  free(wfn_str);

  double norm = wavefunction_norm_c(wfn);
  printf("Norm: %f\n", norm);

  WavefunctionC *normalized_wfn =
      wavefunction_scalar_multiplication_c(wfn, 1 / norm);
  char *nwfn_str = wavefunction_to_string_c(normalized_wfn, 'k');
  printf("Normalized:\n");
  printf("%s\n", nwfn_str);
  free(nwfn_str);

  WavefunctionC *adjoint_wfn = wavefunction_adjoint_c(normalized_wfn);
  char *awfn_str = wavefunction_to_string_c(adjoint_wfn, 'b');
  printf("Adjoint:\n");
  printf("%s\n", awfn_str);
  free(awfn_str);

  free_wavefunction_c(wfn);
  free_wavefunction_c(normalized_wfn);
  free_wavefunction_c(adjoint_wfn);
}

void test_inner_product(void) {

  SlaterDeterminantC *sdet0 =
      slater_determinant_init_c(4, 1, (unsigned int[]){0, 1, 0, 1});
  SlaterDeterminantC *sdet1 = slater_determinant_init_c(
      4, (double complex)I, (unsigned int[]){1, 0, 1, 0});

  WavefunctionC *ket = wavefunction_init_c(4);
  wavefunction_append_slater_determinant_c(ket, sdet0);
  wavefunction_append_slater_determinant_c(ket, sdet1);

  WavefunctionC *bra = wavefunction_adjoint_c(ket);
  double complex inner_prod = wavefunction_multiplication_c(bra, ket);

  char *bra_str = wavefunction_to_string_c(bra, 'b');
  char *ket_str = wavefunction_to_string_c(ket, 'k');

  printf("(%s) * (%s) = %lf + %lfi\n", bra_str, ket_str, creal(inner_prod),
         cimag(inner_prod));

  free(bra_str);
  free(ket_str);
  free_wavefunction_c(bra);
  free_wavefunction_c(ket);
}

void test_appending_slater_determinants(void) {

  printf("Order of Adding:\n");

  WavefunctionC *wfn1 = wavefunction_init_c(4);

  unsigned int orbitals[4];
  for (unsigned int i = 0; i < 2; i++) {
    for (unsigned int j = 0; j < 2; j++) {
      for (unsigned int k = 0; k < 2; k++) {
        for (unsigned int l = 0; l < 2; l++) {
          orbitals[0] = i;
          orbitals[1] = j;
          orbitals[2] = k;
          orbitals[3] = l;

          SlaterDeterminantC *sdet = slater_determinant_init_c(4, 1, orbitals);
          wavefunction_append_slater_determinant_c(wfn1, sdet);
        }
      }
    }
  }

  WavefunctionC *wfn2 = wavefunction_init_c(4);

  for (unsigned int i = 0; i < 2; i++) {
    for (unsigned int j = 0; j < 2; j++) {
      for (unsigned int k = 0; k < 2; k++) {
        for (unsigned int l = 0; l < 2; l++) {
          orbitals[0] = l;
          orbitals[1] = k;
          orbitals[2] = j;
          orbitals[3] = i;

          SlaterDeterminantC *sdet = slater_determinant_init_c(4, 1, orbitals);
          wavefunction_append_slater_determinant_c(wfn2, sdet);
        }
      }
    }
  }

  char *wfn1_str = wavefunction_to_string_c(wfn1, 'b');
  printf("%s\n\n", wfn1_str);
  free(wfn1_str);

  char *wfn2_str = wavefunction_to_string_c(wfn2, 'k');
  printf("%s\n\n", wfn2_str);
  free(wfn2_str);

  double complex product = wavefunction_multiplication_c(wfn1, wfn2);
  printf("Product of %u orbitals with %u orbitals : %lf + %lfi\n\n", wfn1->s,
         wfn2->s, creal(product), cimag(product));

  free_wavefunction_c(wfn1);
  free_wavefunction_c(wfn2);

  WavefunctionC *wfn3 = wavefunction_init_c(4);
  printf("Repeated addition of the same orbitals:\n");
  printf("s before addition: %u\n", wfn3->s);

  for (unsigned int i = 0; i < 4; i++) {
    unsigned int small_orbitals[] = {0, 1};
    SlaterDeterminantC *sdet = slater_determinant_init_c(2, 1, small_orbitals);
    wavefunction_append_slater_determinant_c(wfn3, sdet);
  }

  printf("s after addition: %u\n", wfn3->s);

  char *wfn3_str = wavefunction_to_string_c(wfn3, 'k');
  printf("%s\n", wfn3_str);
  free(wfn3_str);

  free_wavefunction_c(wfn3);
}

void test_wavefunction_pauli_sum_multiplication(void) {

  SlaterDeterminantC *sdet0 =
      slater_determinant_init_c(4, 1, (unsigned int[]){0, 1, 0, 1});
  SlaterDeterminantC *sdet1 =
      slater_determinant_init_c(4, 1, (unsigned int[]){1, 0, 1, 0});
  WavefunctionC *wfn = wavefunction_init_c(4);

  wavefunction_append_slater_determinant_c(wfn, sdet0);
  wavefunction_append_slater_determinant_c(wfn, sdet1);

  PauliStringC *pString0 =
      pauli_string_init_as_ints_c(4, 1, (unsigned int[]){0, 2, 0, 1});
  PauliStringC *pString1 =
      pauli_string_init_as_ints_c(4, 1, (unsigned int[]){1, 0, 1, 0});

  PauliSumC *pSum = pauli_sum_init_c(4);
  pauli_sum_append_pauli_string_c(pSum, pString0);
  pauli_sum_append_pauli_string_c(pSum, pString1);

  WavefunctionC *new_wfn = wavefunction_pauli_sum_multiplication_c(pSum, wfn);

  char *wfn_str = wavefunction_to_string_c(wfn, 'k');
  char *pSum_str = pauli_sum_to_string_c(pSum);
  char *new_wfn_str = wavefunction_to_string_c(new_wfn, 'k');

  printf("(%s) * (%s) = %s\n", pSum_str, wfn_str, new_wfn_str);

  free(wfn_str);
  free(pSum_str);
  free(new_wfn_str);

  free_wavefunction_c(wfn);
  free_wavefunction_c(new_wfn);
  free_pauli_sum_c(pSum);
}

void test_wavefunction_pauli_string_evolution(void) {

  SlaterDeterminantC *sdet0 =
      slater_determinant_init_c(4, 1, (unsigned int[]){0, 1, 0, 1});
  WavefunctionC *wfn = wavefunction_init_c(4);
  wavefunction_append_slater_determinant_c(wfn, sdet0);

  PauliStringC *pString = pauli_string_init_as_ints_c(
      4, (double complex)I, (unsigned int[]){1, 1, 1, 1});

  char *wfn_str = wavefunction_to_string_c(wfn, 'k');

  for (unsigned int t = 0; t < 100000; t++) {
    WavefunctionC *new_wfn =
        wavefunction_pauli_string_evolution_c(pString, wfn, 0.01);
    free_wavefunction_c(wfn);
    wfn = new_wfn;
  }

  char *pString_str = pauli_string_to_string_c(pString);
  char *new_wfn_str = wavefunction_to_string_c(wfn, 'k');
  double norm = wavefunction_norm_c(wfn);

  printf("exp(1000 * (%s)) * (%s) = %s\n", pString_str, wfn_str, new_wfn_str);
  printf("Norm: %lf\n", norm);

  free(wfn_str);
  free(pString_str);
  free(new_wfn_str);

  free_wavefunction_c(wfn);
  free_pauli_string_c(pString);
}

void test_wavefunction_pauli_sum_evolution(void) {

  SlaterDeterminantC *sdet0 =
      slater_determinant_init_c(4, 1, (unsigned int[]){0, 1, 0, 1});
  WavefunctionC *wfn = wavefunction_init_c(4);
  wavefunction_append_slater_determinant_c(wfn, sdet0);

  PauliStringC *pString0 = pauli_string_init_as_ints_c(
      4, (0.5 * (double complex)I), (unsigned int[]){1, 1, 1, 1});
  PauliStringC *pString1 = pauli_string_init_as_ints_c(
      4, (0.5 * (double complex)I), (unsigned int[]){1, 1, 1, 1});

  PauliSumC *pSum = pauli_sum_init_c(4);
  pauli_sum_append_pauli_string_c(pSum, pString0);
  pauli_sum_append_pauli_string_c(pSum, pString1);

  char *wfn_str = wavefunction_to_string_c(wfn, 'k');

  for (unsigned int t = 0; t < 100000; t++) {
    WavefunctionC *new_wfn =
        wavefunction_pauli_sum_evolution_c(pSum, wfn, 0.01);
    free_wavefunction_c(wfn);
    wfn = new_wfn;
  }

  char *pSum_str = pauli_sum_to_string_c(pSum);
  char *new_wfn_str = wavefunction_to_string_c(wfn, 'k');
  double norm = wavefunction_norm_c(wfn);

  printf("exp(1000 * (%s)) * (%s) = %s\n", pSum_str, wfn_str, new_wfn_str);
  printf("Norm: %lf\n", norm);

  free(wfn_str);
  free(pSum_str);
  free(new_wfn_str);

  free_wavefunction_c(wfn);
  free_pauli_sum_c(pSum);
}

void test_wavefunction_speed(void) {
  const unsigned int N = 20;
  unsigned int num_sdets = 2;
  const unsigned int num_operations = 20;

  printf("\nRunning Wavefunction Speed Test with %u qubits, %u Slater "
         "determinants, %u Pauli sum operations.\n",
         N, num_sdets, num_operations);

  WavefunctionC *wfn = wavefunction_init_c(N);

  unsigned int orbitals[N];
  for (unsigned int i = 0; i < num_sdets; i++) {
    for (unsigned int j = 0; j < N; j++) {
      orbitals[j] = (unsigned int)(rand() % 2);
    }
    double complex coef = (rand() / (double)RAND_MAX) +
                          (rand() / (double)RAND_MAX) * (double complex)I;
    SlaterDeterminantC *sdet = slater_determinant_init_c(N, coef, orbitals);
    wavefunction_append_slater_determinant_c(wfn, sdet);
  }

  double norm = wavefunction_norm_c(wfn);
  wfn = wavefunction_scalar_multiplication_c(wfn, 1 / norm);

  printf("Wavefunction initialized.\n");

  PauliSumC *pauli_sums[num_operations];
  unsigned int paulis[N];
  for (unsigned int i = 0; i < num_operations; i++) {
    pauli_sums[i] = pauli_sum_init_c(N);

    for (unsigned int j = 0; j < 3; j++) {
      for (unsigned int k = 0; k < N; k++) {
        unsigned int pauli_type = (unsigned int)(rand() % 4);
        paulis[k] = pauli_type;
      }
      double complex coef = (rand() / (double)RAND_MAX) * (double complex)I;
      PauliStringC *pString = pauli_string_init_as_ints_c(N, coef, paulis);
      pauli_sum_append_pauli_string_c(pauli_sums[i], pString);
    }
  }

  printf("Pauli sums generated.\n");

  WavefunctionC **old_wfns =
      (WavefunctionC **)malloc(num_operations * sizeof(WavefunctionC *));

  clock_t start_time = clock();

  for (unsigned int i = 0; i < num_operations; i++) {
    char *pSum_str = pauli_sum_to_string_c(pauli_sums[i]);
    printf("%u: %s\n", i, pSum_str);
    free(pSum_str);

    WavefunctionC *new_wfn =
        wavefunction_pauli_sum_evolution_c(pauli_sums[i], wfn, 0.01);
    old_wfns[i] = wfn;
    wfn = new_wfn;
  }

  clock_t end_time = clock();
  double elapsed_time = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;

  printf("Completed %u Pauli sum evolutions in %.4f seconds.\n", num_operations,
         elapsed_time);
  printf("Final wavefunction norm: %lf\n", wavefunction_norm_c(wfn));
  printf(
      "Final number of Slater determinants is %u showing a 2^%.2f increase.\n",
      wfn->s, log2((double)wfn->s / num_sdets));

  free_wavefunction_c(wfn);
  for (unsigned int i = 0; i < num_operations; i++) {
    free_wavefunction_c(old_wfns[i]);
    free_pauli_sum_c(pauli_sums[i]);
  }
  free(old_wfns);
}
int main(void) {
  printf("\nTesting Initialization, Scaling, and Freeing\n");
  test_initialization_scaling_freeing();
  printf("\n\nTesting Inner Products\n");
  test_inner_product();
  printf("\n\nTesting Appending Slater Determinants\n");
  test_appending_slater_determinants();
  printf("\n\nTesting Wavefunction Multiplication by Pauli Sum\n");
  test_wavefunction_pauli_sum_multiplication();
  printf("\n\nTesting Wavefunction Evolution by Pauli String\n");
  test_wavefunction_pauli_string_evolution();
  printf("\n\nTesting Wavefunction Evolution by Pauli Sum\n");
  test_wavefunction_pauli_sum_evolution();
  printf("\n\nTesting Wavefunction Speed");
  test_wavefunction_speed();
}
