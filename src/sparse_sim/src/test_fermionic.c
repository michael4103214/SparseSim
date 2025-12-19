#include "fermionic.h"

void test_initialization_scaling_freeing(void) {
  unsigned int total_terms = 9;
  double complex *coefs =
      (double complex *)malloc(total_terms * sizeof(double complex));
  for (int i = 0; i < total_terms; i++) {
    coefs[i] = (double complex)(i);
  }

  FermionicOperatorC *fop =
      fermionic_operator_init_c(3, "-+", (unsigned int[]){0, 1}, coefs);
  char *fop_str = fermionic_operator_to_string_c(fop);

  FermionicOperatorC *scaled_fop =
      fermionic_operator_scalar_multiplication_c(fop, 2.0);
  char *scaled_fop_str = fermionic_operator_to_string_c(scaled_fop);

  printf("2 * \n%s\n=\n%s\n", fop_str, scaled_fop_str);
  free(fop_str);
  free(scaled_fop_str);

  free_fermionic_operator_c(fop);
  free_fermionic_operator_c(scaled_fop);
}

void test_adjoint(void) {
  unsigned int total_terms = 16;
  double complex *coefs =
      (double complex *)malloc(total_terms * sizeof(double complex));
  for (int i = 0; i < total_terms; i++) {
    coefs[i] = (double complex)((i) + (total_terms - i - 1) * I);
  }

  FermionicOperatorC *fop =
      fermionic_operator_init_c(2, "+--+", (unsigned int[]){0, 1, 2, 3}, coefs);
  char *fop_str = fermionic_operator_to_string_c(fop);

  FermionicOperatorC *adjoint_fop = fermionic_operator_adjoint_c(fop);
  char *adjoint_fop_str = fermionic_operator_to_string_c(adjoint_fop);

  printf("Adjoint of \n%s\n=\n%s\n", fop_str, adjoint_fop_str);
  free(fop_str);
  free(adjoint_fop_str);

  free_fermionic_operator_c(fop);
  free_fermionic_operator_c(adjoint_fop);
}

void test_slater_determinant_single_term_multiplication(void) {
  SlaterDeterminantC *sdet =
      slater_determinant_init_c(6, 1.0, (unsigned int[]){1, 0, 0, 0, 0, 1});

  char opstring[] = "----++++";
  unsigned int unordered_idx[] = {1, 2, 4, 3, 1, 2, 3, 4};
  double complex coef = 2.0 + 0.0 * I;

  SlaterDeterminantC *new_sdet =
      slater_determinant_single_term_fermionic_operator_multiplication_c(
          sdet, opstring, unordered_idx, coef);

  char *sdet_str = slater_determinant_to_string_c(sdet, 'k');
  if (new_sdet) {
    char *new_sdet_str = slater_determinant_to_string_c(new_sdet, 'k');

    printf("Applying operator on %s gives %s\n", sdet_str, new_sdet_str);

    free(new_sdet_str);
    free_slater_determinant_c(new_sdet);
  } else {
    printf("Applying operator on %s gives NULL (annihilated)\n", sdet_str);
  }

  free(sdet_str);
  free_slater_determinant_c(sdet);
}

void test_wavefunction_single_term_multiplication(void) {
  SlaterDeterminantC *sdet0 =
      slater_determinant_init_c(6, -1 * I, (unsigned int[]){0, 1, 0, 1, 0, 1});
  SlaterDeterminantC *sdet1 =
      slater_determinant_init_c(6, 1, (unsigned int[]){1, 0, 1, 0, 1, 0});
  SlaterDeterminantC *sdet2 =
      slater_determinant_init_c(6, 0.5, (unsigned int[]){0, 1, 0, 1, 0, 0});

  WavefunctionC *wfn = wavefunction_init_c(6);
  wavefunction_append_slater_determinant_c(wfn, sdet0);
  wavefunction_append_slater_determinant_c(wfn, sdet1);
  wavefunction_append_slater_determinant_c(wfn, sdet2);

  char opstring[] = "++--";
  unsigned int unordered_idx[] = {0, 2, 3, 1};
  double complex coef = 1.0 + 0.0 * I;

  WavefunctionC *new_wfn =
      wavefunction_single_term_fermionic_operator_multiplication_c(
          wfn, opstring, unordered_idx, coef);

  char *wfn_str = wavefunction_to_string_c(wfn, 'k');
  if (new_wfn) {
    char *new_wfn_str = wavefunction_to_string_c(new_wfn, 'k');

    printf("Applying operator on \n%s\n gives\n%s\n", wfn_str, new_wfn_str);

    free(new_wfn_str);
    free_wavefunction_c(new_wfn);
  } else {
    printf("Applying operator on %s gives NULL (annihilated)\n", wfn_str);
  }

  free(wfn_str);
  free_wavefunction_c(wfn);
}

void test_termwise_expectation(void) {
  SlaterDeterminantC *sdet0 =
      slater_determinant_init_c(4, 1, (unsigned int[]){0, 1, 0, 1});
  SlaterDeterminantC *sdet1 =
      slater_determinant_init_c(4, 2 * I, (unsigned int[]){1, 0, 1, 0});

  WavefunctionC *wfn = wavefunction_init_c(4);

  wavefunction_append_slater_determinant_c(wfn, sdet0);
  wavefunction_append_slater_determinant_c(wfn, sdet1);

  char *wfn_str = wavefunction_to_string_c(wfn, 'k');
  printf("|wfn> =%s\n", wfn_str);
  free(wfn_str);

  double complex *format_coefs =
      (double complex *)malloc(256 * sizeof(double complex));
  for (int i = 0; i < 256; i++) {
    format_coefs[i] = 1;
  }

  FermionicOperatorC *trdm_format = fermionic_operator_init_c(
      4, "++--", (unsigned int[]){0, 1, 3, 2}, format_coefs);

  FermionicOperatorC *trdm =
      wavefunction_termwise_fermionic_operator_expectation_c(wfn, trdm_format);
  char *trdm_str = fermionic_operator_to_string_c(trdm);
  printf("TRDM:\n%s\n", trdm_str);

  free(trdm_str);
  free_fermionic_operator_c(trdm);
  free_fermionic_operator_c(trdm_format);
  free_wavefunction_c(wfn);
}

void test_expectation(void) {
  SlaterDeterminantC *sdet0 =
      slater_determinant_init_c(4, 1, (unsigned int[]){0, 1, 0, 1});
  SlaterDeterminantC *sdet1 =
      slater_determinant_init_c(4, 1, (unsigned int[]){1, 0, 1, 0});

  WavefunctionC *wfn = wavefunction_init_c(4);

  wavefunction_append_slater_determinant_c(wfn, sdet0);
  wavefunction_append_slater_determinant_c(wfn, sdet1);

  char *wfn_str = wavefunction_to_string_c(wfn, 'k');
  printf("|wfn> = %s\n", wfn_str);
  free(wfn_str);

  double complex *coefs = (double complex *)calloc(256, sizeof(double complex));
  unsigned int n3 = 4 * 4 * 4;
  unsigned int n2 = 4 * 4;
  unsigned int n1 = 4;
  unsigned int n0 = 1;
  coefs[0 * n3 + 1 * n2 + 1 * n1 + 0 * n0] = 1 * I;
  coefs[1 * n3 + 2 * n2 + 2 * n1 + 1 * n0] = 1 * I;
  coefs[2 * n3 + 3 * n2 + 3 * n1 + 2 * n0] = 1 * I;
  coefs[0 * n3 + 2 * n2 + 2 * n1 + 0 * n0] = -1;
  coefs[0 * n3 + 2 * n2 + 0 * n1 + 2 * n0] = 1;
  coefs[2 * n3 + 0 * n2 + 0 * n1 + 2 * n0] = -1;
  coefs[2 * n3 + 0 * n2 + 2 * n1 + 0 * n0] = 1;
  coefs[1 * n3 + 3 * n2 + 2 * n1 + 0 * n0] = -0.1;
  coefs[3 * n3 + 1 * n2 + 2 * n1 + 0 * n0] = 0.1;
  coefs[1 * n3 + 3 * n2 + 0 * n1 + 2 * n0] = 0.1;
  coefs[3 * n3 + 1 * n2 + 0 * n1 + 2 * n0] = -0.1;
  coefs[0 * n3 + 2 * n2 + 3 * n1 + 1 * n0] = -0.1;
  coefs[0 * n3 + 2 * n2 + 1 * n1 + 3 * n0] = 0.1;
  coefs[2 * n3 + 0 * n2 + 3 * n1 + 1 * n0] = 0.1;
  coefs[2 * n3 + 0 * n2 + 1 * n1 + 3 * n0] = -0.1;

  FermionicOperatorC *fop =
      fermionic_operator_init_c(4, "++--", (unsigned int[]){0, 1, 2, 3}, coefs);
  char *fop_str = fermionic_operator_to_string_c(fop);

  double complex expectation =
      wavefunction_fermionic_operator_expectation_c(wfn, fop);
  printf("Expectation value of \n%s\n = %.6f%+.6fi\n", fop_str,
         creal(expectation), cimag(expectation));

  free_fermionic_operator_c(fop);
  free_wavefunction_c(wfn);
}

int main(void) {
  printf("\nTesting Initialization, Scaling, and Freeing\n");
  test_initialization_scaling_freeing();
  printf("\nTesting Adjoint\n");
  test_adjoint();
  // printf("\nTesting Slater Determinant Single Term Multiplication\n");
  // test_slater_determinant_single_term_multiplication();
  // printf("\nTesting Wavefunction Single Term Multiplication\n");
  // test_wavefunction_single_term_multiplication();
  printf("\nTesting Termwise Expectation\n");
  test_termwise_expectation();
  printf("\nTesting Expectation\n");
  test_expectation();
}