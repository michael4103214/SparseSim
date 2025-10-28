#include <complex.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "density_matrix.h"
#include "wavefunction.h"

void test_initialization_scaling_freeing(void);
void test_density_matrix_from_wavefunction(void);
void test_appending_outer_products(void);
void test_density_matrix_pauli_sum_multiplication(void);
void test_density_matrix_pauli_string_evolution(void);
void test_density_matrix_pauli_sum_evolution(void);
void test_density_matrix_speed(void);

void test_initialization_scaling_freeing(void) {
  DensityMatrixC *dm;
  DensityMatrixC *normalized_dm;

  char *dm_str;
  char *ndm_str;

  double trace;

  unsigned int orbitals0[] = {0, 1, 0, 1};
  unsigned int orbitals1[] = {1, 0, 1, 0};

  OuterProductC *oprod0;
  OuterProductC *oprod1;

  oprod0 = outer_product_init_c(4, 1, orbitals0, orbitals0);
  oprod1 = outer_product_init_c(4, 1, orbitals1, orbitals1);

  dm = density_matrix_init_c(4);
  printf("Density matrix initialized at %p with orbitals: %u, %u\n", (void *)dm,
         oprod0->encoding, oprod1->encoding);

  density_matrix_append_outer_product_c(dm, oprod0);
  density_matrix_append_outer_product_c(dm, oprod1);

  dm_str = density_matrix_to_string_c(dm);
  printf("%s\n", dm_str);
  free(dm_str);

  trace = density_matrix_trace_c(dm);
  printf("Trace: %f\n", trace);

  normalized_dm = density_matrix_scalar_multiplication_c(dm, 1 / trace);
  ndm_str = density_matrix_to_string_c(normalized_dm);
  printf("Normalized:\n");
  printf("%s\n", ndm_str);
  free(ndm_str);

  free_density_matrix_c(dm);
  free_density_matrix_c(normalized_dm);
}

void test_density_matrix_from_wavefunction(void) {
  WavefunctionC *wfn;
  WavefunctionC *adjoint_wfn;
  DensityMatrixC *dm;

  char *wfn_str;
  char *awfn_str;
  char *dm_str;

  double norm;
  double trace;

  unsigned int orbitals0[] = {0, 1, 0, 1};
  unsigned int orbitals1[] = {1, 0, 1, 0};

  SlaterDeterminantC *sdet0;
  SlaterDeterminantC *sdet1;

  sdet0 = slater_determinant_init_c(4, 1 / sqrt(2), orbitals0);
  sdet1 = slater_determinant_init_c(4, (double complex)I / sqrt(2), orbitals1);

  wfn = wavefunction_init_c(4);
  wavefunction_append_slater_determinant_c(wfn, sdet0);
  wavefunction_append_slater_determinant_c(wfn, sdet1);

  adjoint_wfn = wavefunction_adjoint_c(wfn);

  norm = wavefunction_norm_c(wfn);

  dm = density_matrix_from_wavefunction_c(wfn);

  trace = density_matrix_trace_c(dm);

  wfn_str = wavefunction_to_string_c(wfn, 'k');
  awfn_str = wavefunction_to_string_c(adjoint_wfn, 'b');
  dm_str = density_matrix_to_string_c(dm);

  printf("(%s) (%s)\n = %s\n", wfn_str, awfn_str, dm_str);
  printf("Wavefunction norm: %f\n", norm);
  printf("Density matrix trace: %f\n", trace);

  free(wfn_str);
  free(awfn_str);
  free(dm_str);

  free_wavefunction_c(wfn);
  free_wavefunction_c(adjoint_wfn);
  free_density_matrix_c(dm);
}

void test_appending_outer_products(void) {
  DensityMatrixC *dm1;
  DensityMatrixC *dm2;
  DensityMatrixC *dm3;

  unsigned int i, j, k, l;
  unsigned int orbitals[4];

  OuterProductC *oprod;

  char *dm1_str;
  char *dm2_str;
  char *dm3_str;

  printf("Order of Adding:\n");

  dm1 = density_matrix_init_c(4);

  for (i = 0; i < 2; i++) {
    for (j = 0; j < 2; j++) {
      for (k = 0; k < 2; k++) {
        for (l = 0; l < 2; l++) {
          orbitals[0] = i;
          orbitals[1] = j;
          orbitals[2] = k;
          orbitals[3] = l;

          oprod = outer_product_init_c(4, 1, orbitals, orbitals);
          density_matrix_append_outer_product_c(dm1, oprod);
        }
      }
    }
  }

  dm2 = density_matrix_init_c(4);

  for (i = 0; i < 2; i++) {
    for (j = 0; j < 2; j++) {
      for (k = 0; k < 2; k++) {
        for (l = 0; l < 2; l++) {
          orbitals[0] = l;
          orbitals[1] = k;
          orbitals[2] = j;
          orbitals[3] = i;

          oprod = outer_product_init_c(4, 1, orbitals, orbitals);
          density_matrix_append_outer_product_c(dm2, oprod);
        }
      }
    }
  }

  dm1_str = density_matrix_to_string_c(dm1);
  printf("%s\n\n", dm1_str);
  free(dm1_str);

  dm2_str = density_matrix_to_string_c(dm2);
  printf("%s\n\n", dm2_str);
  free(dm2_str);

  free_density_matrix_c(dm1);
  free_density_matrix_c(dm2);

  dm3 = density_matrix_init_c(4);
  printf("Repeated addition of the same orbitals:\n");
  printf("o before addition: %u\n", dm3->o);

  for (i = 0; i < 4; i++) {
    unsigned int small_orbitals[] = {0, 1};
    oprod = outer_product_init_c(2, 1, small_orbitals, small_orbitals);
    density_matrix_append_outer_product_c(dm3, oprod);
  }

  printf("o after addition: %u\n", dm3->o);

  dm3_str = density_matrix_to_string_c(dm3);
  printf("%s\n", dm3_str);
  free(dm3_str);

  free_density_matrix_c(dm3);
}

void test_density_matrix_pauli_sum_multiplication(void) {
  unsigned int orbitals0[] = {0, 1, 0, 1};
  unsigned int orbitals1[] = {1, 0, 1, 0};
  unsigned int paulis0[] = {0, 2, 0, 1};
  unsigned int paulis1[] = {1, 0, 1, 0};

  OuterProductC *oprod0;
  OuterProductC *oprod1;
  DensityMatrixC *dm;
  DensityMatrixC *new_dm_left;
  DensityMatrixC *new_dm_right;

  PauliStringC *pString0;
  PauliStringC *pString1;
  PauliSumC *pSum;

  char *dm_str;
  char *pSum_str;
  char *new_dm_left_str;
  char *new_dm_right_str;

  oprod0 = outer_product_init_c(4, 1, orbitals0, orbitals0);
  oprod1 = outer_product_init_c(4, 1, orbitals1, orbitals1);
  dm = density_matrix_init_c(4);

  density_matrix_append_outer_product_c(dm, oprod0);
  density_matrix_append_outer_product_c(dm, oprod1);

  pString0 = pauli_string_init_as_ints_c(4, 1, paulis0);
  pString1 = pauli_string_init_as_ints_c(4, 1, paulis1);

  pSum = pauli_sum_init_c(4);
  pauli_sum_append_pauli_string_c(pSum, pString0);
  pauli_sum_append_pauli_string_c(pSum, pString1);

  new_dm_left = density_matrix_pauli_sum_left_multiplication_c(pSum, dm);
  new_dm_right = density_matrix_pauli_sum_right_multiplication_c(dm, pSum);

  dm_str = density_matrix_to_string_c(dm);
  pSum_str = pauli_sum_to_string_c(pSum);
  new_dm_left_str = density_matrix_to_string_c(new_dm_left);
  new_dm_right_str = density_matrix_to_string_c(new_dm_right);

  printf("(%s) * (%s)\n = %s\n\n", pSum_str, dm_str, new_dm_left_str);
  printf("(%s) * (%s)\n = %s\n", dm_str, pSum_str, new_dm_right_str);

  free(dm_str);
  free(pSum_str);
  free(new_dm_left_str);
  free(new_dm_right_str);

  free_density_matrix_c(dm);
  free_density_matrix_c(new_dm_left);
  free_density_matrix_c(new_dm_right);
  free_pauli_sum_c(pSum);
}

void test_density_matrix_pauli_string_evolution(void) {
  unsigned int orbitals0[] = {0, 1, 0, 1};
  unsigned int paulis0[] = {1, 1, 1, 1};

  OuterProductC *oprod0;
  DensityMatrixC *dm;
  DensityMatrixC *new_dm;

  PauliStringC *pString;

  char *dm_str;
  char *pString_str;
  char *new_dm_str;

  double trace;
  unsigned int t;

  oprod0 = outer_product_init_c(4, 1, orbitals0, orbitals0);
  dm = density_matrix_init_c(4);
  density_matrix_append_outer_product_c(dm, oprod0);

  pString = pauli_string_init_as_ints_c(4, (double complex)I, paulis0);

  dm_str = density_matrix_to_string_c(dm);

  for (t = 0; t < 100000; t++) {
    new_dm = density_matrix_pauli_string_evolution_c(pString, dm, 0.01);
    free_density_matrix_c(dm);
    dm = new_dm;
  }

  pString_str = pauli_string_to_string_c(pString);
  new_dm_str = density_matrix_to_string_c(dm);
  trace = density_matrix_trace_c(dm);

  printf("exp(1000 * (%s)) * [(%s)]\n = %s\n", pString_str, dm_str, new_dm_str);
  printf("Trace: %lf\n", trace);

  free(dm_str);
  free(pString_str);
  free(new_dm_str);

  free_density_matrix_c(dm);
  free_pauli_string_c(pString);
}

void test_density_matrix_pauli_sum_evolution(void) {
  unsigned int orbitals0[] = {0, 1, 0, 1};
  unsigned int paulis0[] = {1, 1, 1, 1};

  OuterProductC *oprod0;
  DensityMatrixC *dm;
  DensityMatrixC *new_dm;

  PauliStringC *pString0;
  PauliStringC *pString1;
  PauliSumC *pSum;

  char *dm_str;
  char *pSum_str;
  char *new_dm_str;

  double trace;
  unsigned int t;

  oprod0 = outer_product_init_c(4, 1, orbitals0, orbitals0);
  dm = density_matrix_init_c(4);
  density_matrix_append_outer_product_c(dm, oprod0);

  pString0 = pauli_string_init_as_ints_c(4, (0.5 * (double complex)I), paulis0);
  pString1 = pauli_string_init_as_ints_c(4, (0.5 * (double complex)I), paulis0);

  pSum = pauli_sum_init_c(4);
  pauli_sum_append_pauli_string_c(pSum, pString0);
  pauli_sum_append_pauli_string_c(pSum, pString1);

  dm_str = density_matrix_to_string_c(dm);

  for (t = 0; t < 100000; t++) {
    new_dm = density_matrix_pauli_sum_evolution_c(pSum, dm, 0.01);
    free_density_matrix_c(dm);
    dm = new_dm;
  }

  pSum_str = pauli_sum_to_string_c(pSum);
  new_dm_str = density_matrix_to_string_c(dm);
  trace = density_matrix_trace_c(dm);

  printf("exp(1000 * (%s)) [(%s)]\n = %s\n", pSum_str, dm_str, new_dm_str);
  printf("Trace: %lf\n", trace);

  free(dm_str);
  free(pSum_str);
  free(new_dm_str);

  free_density_matrix_c(dm);
  free_pauli_sum_c(pSum);
}

void test_density_matrix_CPTP_evolution(void) {
  const unsigned int N = 4;
  const unsigned int num_H_ops = 2;
  const unsigned int num_L_ops = 2;

  unsigned int paulis[N];
  unsigned int pauli_type;

  PauliStringC *pString;
  PauliSumC *H;
  PauliSumC *Ls[num_L_ops];

  unsigned int i;
  unsigned int j;
  unsigned int k;

  H = pauli_sum_init_c(N);

  for (j = 0; j < num_H_ops; j++) {
    for (k = 0; k < N; k++) {
      pauli_type = (unsigned int)(rand() % 4);
      paulis[k] = pauli_type;
    }
    pString = pauli_string_init_as_ints_c(N, 1.0, paulis);
    pauli_sum_append_pauli_string_c(H, pString);
  }

  char *H_str = pauli_sum_to_string_c(H);
  printf("Hamiltonian:\n%s\n", H_str);
  free(H_str);

  for (i = 0; i < num_L_ops; i++) {
    Ls[i] = pauli_sum_init_c(N);

    for (j = 0; j < 1; j++) {
      for (k = 0; k < N; k++) {
        pauli_type = (unsigned int)(rand() % 4);
        paulis[k] = pauli_type;
      }
      pString = pauli_string_init_as_ints_c(N, 1.0, paulis);
      pauli_sum_append_pauli_string_c(Ls[i], pString);
    }

    char *L_str = pauli_sum_to_string_c(Ls[i]);
    printf("Lindblad Operator %u:\n%s\n", i, L_str);
    free(L_str);
  }

  WavefunctionC *wfn;
  DensityMatrixC *dm;
  double trace;

  unsigned int orbitals0[] = {0, 1, 0, 1};
  unsigned int orbitals1[] = {1, 0, 1, 0};

  SlaterDeterminantC *sdet0;
  SlaterDeterminantC *sdet1;

  sdet0 = slater_determinant_init_c(4, 1 / sqrt(2), orbitals0);
  sdet1 = slater_determinant_init_c(4, (double complex)I / sqrt(2), orbitals1);

  wfn = wavefunction_init_c(4);
  wavefunction_append_slater_determinant_c(wfn, sdet0);
  wavefunction_append_slater_determinant_c(wfn, sdet1);

  dm = density_matrix_from_wavefunction_c(wfn);
  trace = density_matrix_trace_c(dm);

  char *dm_str;

  dm_str = density_matrix_to_string_c(dm);

  printf("Initial density matrix \n%s\n", dm_str);
  printf("Initial density matrix trace: %lf\n\n", trace);

  free(dm_str);

  DensityMatrixC *new_dm =
      density_matrix_CPTP_evolution_c(H, Ls, num_L_ops, dm, 100);

  dm_str = density_matrix_to_string_c(new_dm);
  trace = density_matrix_trace_c(new_dm);

  printf("Density matrix after CPTP evolution \n%s\n", dm_str);
  printf("Density matrix trace after CPTP evolution: %lf\n", trace);

  free(dm_str);

  free_density_matrix_c(dm);
  free_density_matrix_c(new_dm);
  free_wavefunction_c(wfn);
  free_pauli_sum_c(H);
  for (i = 0; i < num_L_ops; i++) {
    free_pauli_sum_c(Ls[i]);
  }
}

void test_density_matrix_speed(void) {
  const unsigned int N = 10;
  unsigned int num_oprods = 2;
  const unsigned int num_operations = 20;

  unsigned int i, j, k;
  unsigned int orbitals[N];
  unsigned int paulis[N];
  unsigned int pauli_type;

  OuterProductC *oprod;
  DensityMatrixC *dm;
  DensityMatrixC *new_dm;

  PauliSumC *pauli_sums[num_operations];
  PauliStringC *pString;

  DensityMatrixC **old_dms;
  char *pSum_str;

  double complex coef;
  double trace;
  clock_t start_time, end_time;
  double elapsed_time;

  printf("\nRunning Density Matrix Speed Test with %u qubits, %u Outer "
         "Products, %u Pauli sum operations.\n",
         N, num_oprods, num_operations);

  dm = density_matrix_init_c(N);

  for (i = 0; i < num_oprods; i++) {
    for (j = 0; j < N; j++) {
      orbitals[j] = (unsigned int)(rand() % 2);
    }
    coef = (rand() / (double)RAND_MAX);
    oprod = outer_product_init_c(N, 1, orbitals, orbitals);
    density_matrix_append_outer_product_c(dm, oprod);
  }

  trace = density_matrix_trace_c(dm);
  dm = density_matrix_scalar_multiplication_c(dm, 1 / trace);

  printf("Density Matrix initialized.\n");

  for (i = 0; i < num_operations; i++) {
    pauli_sums[i] = pauli_sum_init_c(N);

    for (j = 0; j < 3; j++) {
      for (k = 0; k < N; k++) {
        pauli_type = (unsigned int)(rand() % 4);
        paulis[k] = pauli_type;
      }
      coef = (rand() / (double)RAND_MAX) * (double complex)I;
      pString = pauli_string_init_as_ints_c(N, coef, paulis);
      pauli_sum_append_pauli_string_c(pauli_sums[i], pString);
    }
  }

  printf("Pauli sums generated.\n");

  old_dms =
      (DensityMatrixC **)malloc(num_operations * sizeof(DensityMatrixC *));

  start_time = clock();

  for (i = 0; i < num_operations; i++) {
    pSum_str = pauli_sum_to_string_c(pauli_sums[i]);
    printf("%u: %s\n", i, pSum_str);
    free(pSum_str);

    new_dm = density_matrix_pauli_sum_evolution_c(pauli_sums[i], dm, 0.01);
    old_dms[i] = dm;
    dm = new_dm;
  }

  end_time = clock();
  elapsed_time = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;

  printf("Completed %u Pauli sum evolutions in %.4f seconds.\n", num_operations,
         elapsed_time);
  printf("Final density matrix trace: %lf\n", density_matrix_trace_c(dm));
  printf("Final number of outer products is %u showing a 2^%.2f increase.\n",
         dm->o, log2((double)dm->o / num_oprods));

  free_density_matrix_c(dm);
  for (i = 0; i < num_operations; i++) {
    free_density_matrix_c(old_dms[i]);
    free_pauli_sum_c(pauli_sums[i]);
  }
  free(old_dms);
}
int main(void) {
  printf("\nTesting Initialization, Scaling, and Freeing\n");
  test_initialization_scaling_freeing();
  printf("\n\nTesting Density Matrix from Wavefunction\n");
  test_density_matrix_from_wavefunction();
  printf("\n\nTesting Appending outer products\n");
  test_appending_outer_products();
  printf("\n\nTesting Density Matrix Multiplication by Pauli Sum\n");
  test_density_matrix_pauli_sum_multiplication();
  printf("\n\nTesting Density Matrix Evolution by Pauli String\n");
  test_density_matrix_pauli_string_evolution();
  printf("\n\nTesting Density Matrix Evolution by Pauli Sum\n");
  test_density_matrix_pauli_sum_evolution();
  printf("\n\nTesting Density Matrix CPTP Evolution\n");
  test_density_matrix_CPTP_evolution();
  printf("\n\nTesting Density Matrix Speed");
  test_density_matrix_speed();
}