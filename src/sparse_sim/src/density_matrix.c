#include "density_matrix.h"

void free_outer_product_c(OuterProductC *oprod) {
  free(oprod->ket_orbitals);
  free(oprod->bra_orbitals);
  free(oprod);
}

OuterProductC *outer_product_init_c(const unsigned int N, double complex coef,
                                    unsigned int ket_orbitals[],
                                    unsigned int bra_orbitals[]) {

  OuterProductC *oprod = (OuterProductC *)malloc(sizeof(OuterProductC));
  if (!oprod) {
    fprintf(stderr, "Malloc failed for OuterProduct\n");
    return NULL;
  }

  oprod->N = N;
  oprod->coef = coef;

  oprod->ket_orbitals = (unsigned int *)malloc(N * sizeof(unsigned int));
  if (!oprod->ket_orbitals) {
    fprintf(stderr, "Malloc failed for ket_orbitals\n");
    free(oprod);
    return NULL;
  }

  oprod->bra_orbitals = (unsigned int *)malloc(N * sizeof(unsigned int));
  if (!oprod->bra_orbitals) {
    fprintf(stderr, "Malloc failed for bra_orbitals\n");
    free(oprod->ket_orbitals);
    free(oprod);
    return NULL;
  }

  uint64_t encoding = 0;
  for (unsigned int i = 0; i < N; i++) {
    oprod->ket_orbitals[i] = ket_orbitals[i];
    oprod->bra_orbitals[i] = bra_orbitals[i];
    if (ket_orbitals[i]) {
      encoding += 1 << (i);
    }
    if (bra_orbitals[i]) {
      encoding += 1 << (i + N);
    }
  }
  oprod->encoding = encoding;

  return oprod;
}

char *outer_product_to_string_c(OuterProductC *oprod) {

  char orbital_char[2]; // To hold a single orbital (1 digit + null terminator)

  // Calculate buffer size
  size_t buffer_size = 40         // for coef
                       + 1        // for |
                       + oprod->N // for ket orbitals
                       + 2        // for ><
                       + oprod->N // for bra orbitals
                       + 1        // for |
                       + 1;       // for terminator

  // Allocate memory for the buffer
  char *buffer = (char *)malloc(buffer_size);
  if (!buffer) {
    fprintf(stderr, "Malloc failed for outer_product_to_string_c\n");
    return NULL;
  }

  snprintf(buffer, buffer_size, "(%.4lf + %.4lfi)", creal(oprod->coef),
           cimag(oprod->coef));

  // Append the | character
  strcat(buffer, "|");

  // Append the ket orbital occupations
  for (unsigned int i = 0; i < oprod->N; i++) {
    snprintf(orbital_char, sizeof(orbital_char), "%u", oprod->ket_orbitals[i]);
    strcat(buffer, orbital_char);
  }

  // Append the >< characters
  strcat(buffer, "><");

  // Append the bra orbital occupations
  for (unsigned int i = 0; i < oprod->N; i++) {
    snprintf(orbital_char, sizeof(orbital_char), "%u", oprod->bra_orbitals[i]);
    strcat(buffer, orbital_char);
  }

  // Append the | character
  strcat(buffer, "|");

  return buffer;
}

OuterProductC *outer_product_scalar_multiplication_c(OuterProductC *oprod,
                                                     double complex scalar) {
  OuterProductC *new_oprod = outer_product_init_c(
      oprod->N, oprod->coef * scalar, oprod->ket_orbitals, oprod->bra_orbitals);
  return new_oprod;
}
OuterProductC *outer_product_multiplication(OuterProductC *oprod_left,
                                            OuterProductC *oprod_right) {

  if (oprod_left->N != oprod_right->N) {
    fprintf(stderr, "Error: Outer products have different number of qubits.\n");
    return NULL;
  }

  unsigned int N = oprod_left->N;

  unsigned int comparison = 1;
  for (unsigned int i = 0; i < N; i++) {
    if (oprod_left->bra_orbitals[i] != oprod_right->ket_orbitals[i]) {
      comparison = 0;
      break;
    }
  }

  double complex new_coef;
  if (comparison == 0) {
    new_coef = 0.0 + 0.0 * (double complex)I;
  } else {
    new_coef = oprod_left->coef * oprod_right->coef;
  }

  OuterProductC *new_oprod = outer_product_init_c(
      N, new_coef, oprod_left->ket_orbitals, oprod_right->bra_orbitals);
  return new_oprod;
}

OuterProductC *
outer_product_pauli_string_left_multiplication_c(PauliStringC *pString,
                                                 OuterProductC *oprod) {

  if (pString->N != oprod->N) {
    fprintf(stderr, "Error: Pauli string and wavefunction have different "
                    "number of qubits.\n");
    return NULL;
  }

  unsigned int N = oprod->N;
  unsigned int *new_ket_orbitals =
      (unsigned int *)malloc(oprod->N * sizeof(int));
  if (!new_ket_orbitals) {
    fprintf(stderr, "Malloc failed for new_ket_orbitals\n");
    return NULL;
  }
  double complex new_coef = pString->coef * oprod->coef;

  for (unsigned int i = 0; i < N; i++) {
    switch (pString->paulis[i]) {
    case 0: // I
      new_ket_orbitals[i] = oprod->ket_orbitals[i];
      break;
    case 1: // X
      new_ket_orbitals[i] = oprod->ket_orbitals[i] ^ 1;
      break;
    case 2: // Y
      new_ket_orbitals[i] = oprod->ket_orbitals[i] ^ 1;
      new_coef = new_coef * ((double complex)I) *
                 (1 - (int)(oprod->ket_orbitals[i] << 1));
      break;
    case 3: // Z
      new_ket_orbitals[i] = oprod->ket_orbitals[i];
      new_coef = new_coef * (1 - (int)(oprod->ket_orbitals[i] << 1));
      break;
    default:
      fprintf(stderr, "Error: Invalid Pauli operator %u at index %u\n",
              pString->paulis[i], i);
      free(new_ket_orbitals);
      return NULL;
    }
  }

  OuterProductC *new_oprod =
      outer_product_init_c(N, new_coef, new_ket_orbitals, oprod->bra_orbitals);
  free(new_ket_orbitals);
  return new_oprod;
}

OuterProductC *
outer_product_pauli_string_right_multiplication_c(OuterProductC *oprod,
                                                  PauliStringC *pString) {

  if (pString->N != oprod->N) {
    fprintf(stderr, "Error: Pauli string and wavefunction have different "
                    "number of qubits.\n");
    return NULL;
  }

  unsigned int N = oprod->N;
  unsigned int *new_bra_orbitals =
      (unsigned int *)malloc(oprod->N * sizeof(int));
  if (!new_bra_orbitals) {
    fprintf(stderr, "Malloc failed for new_bra_orbitals\n");
    return NULL;
  }
  double complex new_coef = pString->coef * oprod->coef;

  for (unsigned int i = 0; i < N; i++) {
    switch (pString->paulis[i]) {
    case 0: // I
      new_bra_orbitals[i] = oprod->bra_orbitals[i];
      break;
    case 1: // X
      new_bra_orbitals[i] = oprod->bra_orbitals[i] ^ 1;
      break;
    case 2: // Y
      new_bra_orbitals[i] = oprod->bra_orbitals[i] ^ 1;
      new_coef = new_coef * (-(double complex)I) *
                 (1 - (int)(oprod->bra_orbitals[i] << 1));
      break;
    case 3: // Z
      new_bra_orbitals[i] = oprod->bra_orbitals[i];
      new_coef = new_coef * (1 - (int)(oprod->bra_orbitals[i] << 1));
      break;
    default:
      fprintf(stderr, "Error: Invalid Pauli operator %u at index %u\n",
              pString->paulis[i], i);
      free(new_bra_orbitals);
      return NULL;
    }
  }

  OuterProductC *new_oprod =
      outer_product_init_c(N, new_coef, oprod->ket_orbitals, new_bra_orbitals);
  free(new_bra_orbitals);
  return new_oprod;
}

void free_density_matrix_c(DensityMatrixC *dm) {
  if (dm == NULL) {
    return;
  }

  for (khiter_t k = kh_begin(dm->outer_products);
       k != kh_end(dm->outer_products); ++k) {
    if (kh_exist(dm->outer_products, k)) {
      OuterProductC *oprod = (OuterProductC *)kh_value(dm->outer_products, k);
      free_outer_product_c(oprod);
    }
  }
  kh_destroy(outer_product_hash, dm->outer_products);
  free(dm);
}

double DENSITYMATRIX_CUTOFF_DEFAULT = 1e-16;

DensityMatrixC *density_matrix_init_c(unsigned int N) {
  DensityMatrixC *dm = (DensityMatrixC *)malloc(sizeof(DensityMatrixC));
  if (!dm) {
    fprintf(stderr, "Malloc failed for DensityMatrix\n");
    return NULL;
  }
  dm->o = 0;
  dm->outer_products = kh_init(outer_product_hash);
  dm->N = N;
  dm->cutoff = DENSITYMATRIX_CUTOFF_DEFAULT;
  return dm;
}

DensityMatrixC *density_matrix_init_with_specified_cutoff_c(unsigned int N,
                                                            double cutoff) {
  DensityMatrixC *dm = density_matrix_init_c(N);
  dm->cutoff = cutoff;
  return dm;
}

void density_matrix_append_outer_product_c(DensityMatrixC *dm,
                                           OuterProductC *oprod) {

  if (fabs(creal(oprod->coef)) < dm->cutoff &&
      fabs(cimag(oprod->coef)) < dm->cutoff) {
    free_outer_product_c(oprod);
    return;
  }
  int ret;
  khiter_t k =
      kh_put(outer_product_hash, dm->outer_products, oprod->encoding, &ret);

  if (ret == 0) {
    OuterProductC *existing_oprod =
        (OuterProductC *)kh_value(dm->outer_products, k);
    existing_oprod->coef += oprod->coef;

    if (fabs(creal(existing_oprod->coef)) < dm->cutoff &&
        fabs(cimag(existing_oprod->coef)) < dm->cutoff) {
      kh_del(outer_product_hash, dm->outer_products, k);
      free_outer_product_c(existing_oprod);
      dm->o--;
    }
    free_outer_product_c(oprod);
  } else {
    kh_value(dm->outer_products, k) = oprod;
    dm->o++;
  }
}

DensityMatrixC *density_matrix_from_wavefunction_c(WavefunctionC *wfn) {
  if (!wfn) {
    fprintf(stderr, "Error: Received NULL wavefunction.\n");
    return NULL;
  }

  DensityMatrixC *dm = density_matrix_init_with_specified_cutoff_c(
      wfn->N, wfn->cutoff * wfn->cutoff);
  if (!dm) {
    return NULL;
  }

  for (khiter_t k_ket = kh_begin(wfn->slater_determinants);
       k_ket != kh_end(wfn->slater_determinants); ++k_ket) {

    if (kh_exist(wfn->slater_determinants, k_ket)) {

      SlaterDeterminantC *ket_sd =
          (SlaterDeterminantC *)kh_value(wfn->slater_determinants, k_ket);

      for (khiter_t k_bra = kh_begin(wfn->slater_determinants);
           k_bra != kh_end(wfn->slater_determinants); ++k_bra) {

        if (kh_exist(wfn->slater_determinants, k_bra)) {

          SlaterDeterminantC *bra_sd =
              (SlaterDeterminantC *)kh_value(wfn->slater_determinants, k_bra);

          double complex coef = ket_sd->coef * conj(bra_sd->coef);
          OuterProductC *oprod = outer_product_init_c(
              wfn->N, coef, ket_sd->orbitals, bra_sd->orbitals);
          if (!oprod) {
            free_density_matrix_c(dm);
            return NULL;
          }

          density_matrix_append_outer_product_c(dm, oprod);
        }
      }
    }
  }

  return dm;
}

double density_matrix_trace_c(DensityMatrixC *dm) {

  if (!dm) {
    fprintf(stderr, "Error: Received NULL density matrix.\n");
    return 0.0;
  }

  double trace = 0.0;

  for (khiter_t k = kh_begin(dm->outer_products);
       k != kh_end(dm->outer_products); ++k) {
    if (kh_exist(dm->outer_products, k)) {
      OuterProductC *oprod = (OuterProductC *)kh_value(dm->outer_products, k);
      int is_diagonal = 1;
      for (unsigned int i = 0; i < dm->N; i++) {
        if (oprod->ket_orbitals[i] != oprod->bra_orbitals[i]) {
          is_diagonal = 0;
          break;
        }
      }
      if (is_diagonal) {
        trace += creal(oprod->coef);
      }
    }
  }
  return trace;
}

DensityMatrixC *density_matrix_scalar_multiplication_c(DensityMatrixC *dm,
                                                       double complex scalar) {
  if (!dm) {
    fprintf(stderr, "Error: Received NULL density matrix.\n");
    return NULL;
  }

  DensityMatrixC *new_dm =
      density_matrix_init_with_specified_cutoff_c(dm->N, dm->cutoff);

  for (khiter_t k = kh_begin(dm->outer_products);
       k != kh_end(dm->outer_products); ++k) {
    if (kh_exist(dm->outer_products, k)) {
      OuterProductC *oprod = (OuterProductC *)kh_value(dm->outer_products, k);
      OuterProductC *new_oprod =
          outer_product_scalar_multiplication_c(oprod, scalar);

      int ret;
      khiter_t new_k = kh_put(outer_product_hash, new_dm->outer_products,
                              new_oprod->encoding, &ret);
      kh_value(new_dm->outer_products, new_k) = new_oprod;
      new_dm->o++;
    }
  }

  return new_dm;
}

DensityMatrixC *density_matrix_multiplication_c(DensityMatrixC *dm_left,
                                                DensityMatrixC *dm_right) {
  if (dm_left->N != dm_right->N) {
    fprintf(stderr,
            "Error: Density matrices have different number of qubits.\n");
    return NULL;
  }

  DensityMatrixC *new_dm =
      density_matrix_init_with_specified_cutoff_c(dm_left->N, dm_left->cutoff);

  for (khiter_t k_left = kh_begin(dm_left->outer_products);
       k_left != kh_end(dm_left->outer_products); ++k_left) {
    if (kh_exist(dm_left->outer_products, k_left)) {
      OuterProductC *oprod_left =
          (OuterProductC *)kh_value(dm_left->outer_products, k_left);

      for (khiter_t k_right = kh_begin(dm_right->outer_products);
           k_right != kh_end(dm_right->outer_products); ++k_right) {
        if (kh_exist(dm_right->outer_products, k_right)) {
          OuterProductC *oprod_right =
              (OuterProductC *)kh_value(dm_right->outer_products, k_right);

          OuterProductC *new_oprod =
              outer_product_multiplication(oprod_left, oprod_right);
          if (new_oprod) {
            density_matrix_append_outer_product_c(new_dm, new_oprod);
          }
        }
      }
    }
  }

  return new_dm;
}

char *density_matrix_to_string_c(DensityMatrixC *dm) {
  if (!dm) {
    fprintf(stderr, "Error: Received NULL density matrix.\n");
    return NULL;
  }

  if (dm->o == 0) {
    char *empty_str = (char *)malloc(1);
    if (!empty_str)
      return NULL;
    empty_str[0] = '\0';
    return empty_str;
  }

  size_t buffer_size = 1; // For the null terminator
  char *buffer = (char *)malloc(buffer_size);
  if (!buffer) {
    fprintf(stderr, "Error: Memory allocation failed\n");
    return NULL;
  }
  buffer[0] = '\0';

  int first_entry = 1; // Flag to track first element (to avoid extra " + ")

  for (khiter_t k = kh_begin(dm->outer_products);
       k != kh_end(dm->outer_products); ++k) {
    if (kh_exist(dm->outer_products, k)) {
      OuterProductC *oprod = (OuterProductC *)kh_value(dm->outer_products, k);
      char *oprod_str = outer_product_to_string_c(oprod);
      if (!oprod_str) {
        fprintf(stderr,
                "Error: Failed to allocate memory for Outer Product string\n");
        free(buffer);
        return NULL;
      }

      // Calculate new buffer size
      size_t new_size = buffer_size + strlen(oprod_str) +
                        (first_entry ? 0 : 3); // +3 for " + "

      char *new_buffer = (char *)realloc(buffer, new_size);
      if (!new_buffer) {
        fprintf(stderr, "Error: Memory reallocation failed\n");
        free(buffer);
        free(oprod_str);
        return NULL;
      }
      buffer = new_buffer;
      buffer_size = new_size;

      // Append " + " if it's not the first entry
      if (!first_entry) {
        strcat(buffer, " + ");
      }
      strcat(buffer, oprod_str);

      free(oprod_str);
      first_entry = 0;
    }
  }

  return buffer;
}

DensityMatrixC *
density_matrix_pauli_string_left_multiplication_c(PauliStringC *pString,
                                                  DensityMatrixC *dm) {
  if (!dm || !pString) {
    fprintf(stderr, "Error: Received NULL density matrix or Pauli string.\n");
    return NULL;
  }

  if (dm->N != pString->N) {
    fprintf(stderr, "Error: Density matrix and Pauli string have different "
                    "number of qubits.\n");
    return NULL;
  }

  DensityMatrixC *new_dm =
      density_matrix_init_with_specified_cutoff_c(dm->N, dm->cutoff);

  for (khiter_t k = kh_begin(dm->outer_products);
       k != kh_end(dm->outer_products); ++k) {
    if (kh_exist(dm->outer_products, k)) {
      OuterProductC *oprod = (OuterProductC *)kh_value(dm->outer_products, k);
      OuterProductC *new_oprod =
          outer_product_pauli_string_left_multiplication_c(pString, oprod);

      if (new_oprod) {
        density_matrix_append_outer_product_c(new_dm, new_oprod);
      }
    }
  }

  return new_dm;
}

DensityMatrixC *
density_matrix_pauli_string_right_multiplication_c(DensityMatrixC *dm,
                                                   PauliStringC *pString) {
  if (!dm || !pString) {
    fprintf(stderr, "Error: Received NULL density matrix or Pauli string.\n");
    return NULL;
  }

  if (dm->N != pString->N) {
    fprintf(stderr, "Error: Density matrix and Pauli string have different "
                    "number of qubits.\n");
    return NULL;
  }

  DensityMatrixC *new_dm =
      density_matrix_init_with_specified_cutoff_c(dm->N, dm->cutoff);

  for (khiter_t k = kh_begin(dm->outer_products);
       k != kh_end(dm->outer_products); ++k) {
    if (kh_exist(dm->outer_products, k)) {
      OuterProductC *oprod = (OuterProductC *)kh_value(dm->outer_products, k);
      OuterProductC *new_oprod =
          outer_product_pauli_string_right_multiplication_c(oprod, pString);

      if (new_oprod) {
        density_matrix_append_outer_product_c(new_dm, new_oprod);
      }
    }
  }

  return new_dm;
}

DensityMatrixC *
density_matrix_pauli_sum_left_multiplication_c(PauliSumC *pSum,
                                               DensityMatrixC *dm) {
  if (!dm || !pSum) {
    fprintf(stderr, "Error: Received NULL density matrix or Pauli sum.\n");
    return NULL;
  }
  if (dm->N != pSum->N) {
    fprintf(stderr, "Error: Density matrix and Pauli sum have different number "
                    "of qubits.\n");
    return NULL;
  }

  DensityMatrixC *new_dm =
      density_matrix_init_with_specified_cutoff_c(dm->N, dm->cutoff);

  int success = 0;

  for (khiter_t kp = kh_begin(pSum->pauli_strings);
       kp != kh_end(pSum->pauli_strings); ++kp) {
    if (kh_exist(pSum->pauli_strings, kp)) {
      PauliStringC *pString = (PauliStringC *)kh_value(pSum->pauli_strings, kp);

      for (khiter_t ko = kh_begin(dm->outer_products);
           ko != kh_end(dm->outer_products); ++ko) {
        if (kh_exist(dm->outer_products, ko)) {
          OuterProductC *oprod =
              (OuterProductC *)kh_value(dm->outer_products, ko);
          OuterProductC *new_oprod =
              outer_product_pauli_string_left_multiplication_c(pString, oprod);

          if (!new_oprod) {
            fprintf(stderr,
                    "Error: Pauli string left multiplication failed.\n");
            continue;
          }

          density_matrix_append_outer_product_c(new_dm, new_oprod);
          success = 1;
        }
      }
    }
  }

  if (!success) {
    fprintf(stderr, "Warning: No valid multiplications were performed.\n");
  }

  return new_dm;
}

DensityMatrixC *
density_matrix_pauli_sum_right_multiplication_c(DensityMatrixC *dm,
                                                PauliSumC *pSum) {
  if (!dm || !pSum) {
    fprintf(stderr, "Error: Received NULL density matrix or Pauli sum.\n");
    return NULL;
  }
  if (dm->N != pSum->N) {
    fprintf(stderr, "Error: Density matrix and Pauli sum have different number "
                    "of qubits.\n");
    return NULL;
  }

  DensityMatrixC *new_dm =
      density_matrix_init_with_specified_cutoff_c(dm->N, dm->cutoff);

  int success = 0;

  for (khiter_t kp = kh_begin(pSum->pauli_strings);
       kp != kh_end(pSum->pauli_strings); ++kp) {
    if (kh_exist(pSum->pauli_strings, kp)) {
      PauliStringC *pString = (PauliStringC *)kh_value(pSum->pauli_strings, kp);

      for (khiter_t ko = kh_begin(dm->outer_products);
           ko != kh_end(dm->outer_products); ++ko) {
        if (kh_exist(dm->outer_products, ko)) {
          OuterProductC *oprod =
              (OuterProductC *)kh_value(dm->outer_products, ko);
          OuterProductC *new_oprod =
              outer_product_pauli_string_right_multiplication_c(oprod, pString);

          if (!new_oprod) {
            fprintf(stderr,
                    "Error: Pauli string right multiplication failed.\n");
            continue;
          }

          density_matrix_append_outer_product_c(new_dm, new_oprod);
          success = 1;
        }
      }
    }
  }

  if (!success) {
    fprintf(stderr, "Warning: No valid multiplications were performed.\n");
  }

  return new_dm;
}

OuterProductC *evolution_helper_cosh_cosh(PauliStringC *pString,
                                          OuterProductC *oprod,
                                          double complex epsilon) {
  unsigned int N = oprod->N;
  double complex x = pString->coef * epsilon;
  double complex new_coef = ccosh(x) * ccosh(conj(x)) * oprod->coef;
  return outer_product_init_c(N, new_coef, oprod->ket_orbitals,
                              oprod->bra_orbitals);
}

OuterProductC *evolution_helper_sinh_cosh(PauliStringC *pString,
                                          OuterProductC *oprod,
                                          double complex epsilon) {

  if (pString->N != oprod->N) {
    fprintf(stderr, "Error: Pauli string and wavefunction have different "
                    "number of indices.\n");
    return NULL;
  }

  unsigned int N = oprod->N;
  unsigned int *new_ket_orbitals =
      (unsigned int *)malloc(oprod->N * sizeof(int));
  if (!new_ket_orbitals) {
    fprintf(stderr, "Malloc failed for new_ket_orbitals\n");
    return NULL;
  }

  double complex x = pString->coef * epsilon;
  double complex new_coef = csinh(x) * ccosh(conj(x)) * oprod->coef;

  for (unsigned int i = 0; i < N; i++) {
    switch (pString->paulis[i]) {
    case 0: // I
      new_ket_orbitals[i] = oprod->ket_orbitals[i];
      break;
    case 1: // X
      new_ket_orbitals[i] = oprod->ket_orbitals[i] ^ 1;
      break;
    case 2: // Y
      new_ket_orbitals[i] = oprod->ket_orbitals[i] ^ 1;
      new_coef = new_coef * ((double complex)I) *
                 (1 - (int)(oprod->ket_orbitals[i] << 1));
      break;
    case 3: // Z
      new_ket_orbitals[i] = oprod->ket_orbitals[i];
      new_coef = new_coef * (1 - (int)(oprod->ket_orbitals[i] << 1));
      break;
    default:
      fprintf(stderr, "Error: Invalid Pauli operator %u at index %u\n",
              pString->paulis[i], i);
      free(new_ket_orbitals);
      return NULL;
    }
  }
  OuterProductC *new_oprod =
      outer_product_init_c(N, new_coef, new_ket_orbitals, oprod->bra_orbitals);
  free(new_ket_orbitals);
  return new_oprod;
}

OuterProductC *evolution_helper_cosh_sinh(PauliStringC *pString,
                                          OuterProductC *oprod,
                                          double complex epsilon) {

  if (pString->N != oprod->N) {
    fprintf(stderr, "Error: Pauli string and wavefunction have different "
                    "number of indices.\n");
    return NULL;
  }

  unsigned int N = oprod->N;
  unsigned int *new_bra_orbitals =
      (unsigned int *)malloc(oprod->N * sizeof(int));
  if (!new_bra_orbitals) {
    fprintf(stderr, "Malloc failed for new_bra_orbitals\n");
    return NULL;
  }

  double complex x = pString->coef * epsilon;
  double complex new_coef = ccosh(x) * csinh(conj(x)) * oprod->coef;

  for (unsigned int i = 0; i < N; i++) {
    switch (pString->paulis[i]) {
    case 0: // I
      new_bra_orbitals[i] = oprod->bra_orbitals[i];
      break;
    case 1: // X
      new_bra_orbitals[i] = oprod->bra_orbitals[i] ^ 1;
      break;
    case 2: // Y
      new_bra_orbitals[i] = oprod->bra_orbitals[i] ^ 1;
      new_coef = new_coef * (-(double complex)I) *
                 (1 - (int)(oprod->bra_orbitals[i] << 1));
      break;
    case 3: // Z
      new_bra_orbitals[i] = oprod->bra_orbitals[i];
      new_coef = new_coef * (1 - (int)(oprod->bra_orbitals[i] << 1));
      break;
    default:
      fprintf(stderr, "Error: Invalid Pauli operator %u at index %u\n",
              pString->paulis[i], i);
      free(new_bra_orbitals);
      return NULL;
    }
  }
  OuterProductC *new_oprod =
      outer_product_init_c(N, new_coef, oprod->ket_orbitals, new_bra_orbitals);
  free(new_bra_orbitals);
  return new_oprod;
}

OuterProductC *evolution_helper_sinh_sinh(PauliStringC *pString,
                                          OuterProductC *oprod,
                                          double complex epsilon) {

  if (pString->N != oprod->N) {
    fprintf(stderr, "Error: Pauli string and wavefunction have different "
                    "number of indices.\n");
    return NULL;
  }

  unsigned int N = oprod->N;
  unsigned int *new_ket_orbitals =
      (unsigned int *)malloc(oprod->N * sizeof(int));
  if (!new_ket_orbitals) {
    fprintf(stderr, "Malloc failed for new_ket_orbitals\n");
    return NULL;
  }
  unsigned int *new_bra_orbitals =
      (unsigned int *)malloc(oprod->N * sizeof(int));
  if (!new_bra_orbitals) {
    fprintf(stderr, "Malloc failed for new_bra_orbitals\n");
    free(new_ket_orbitals);
    return NULL;
  }

  double complex x = pString->coef * epsilon;
  double complex new_coef = csinh(x) * csinh(conj(x)) * oprod->coef;

  for (unsigned int i = 0; i < N; i++) {
    switch (pString->paulis[i]) {
    case 0: // I
      new_ket_orbitals[i] = oprod->ket_orbitals[i];
      new_bra_orbitals[i] = oprod->bra_orbitals[i];
      break;
    case 1: // X
      new_ket_orbitals[i] = oprod->ket_orbitals[i] ^ 1;
      new_bra_orbitals[i] = oprod->bra_orbitals[i] ^ 1;
      break;
    case 2: // Y
      new_ket_orbitals[i] = oprod->ket_orbitals[i] ^ 1;
      new_bra_orbitals[i] = oprod->bra_orbitals[i] ^ 1;
      new_coef = new_coef * (1 - (int)(oprod->ket_orbitals[i] << 1)) *
                 (1 - (int)(oprod->bra_orbitals[i] << 1));
      break;
    case 3: // Z
      new_ket_orbitals[i] = oprod->ket_orbitals[i];
      new_bra_orbitals[i] = oprod->bra_orbitals[i];
      new_coef = new_coef * (1 - (int)(oprod->ket_orbitals[i] << 1)) *
                 (1 - (int)(oprod->bra_orbitals[i] << 1));
      break;
    default:
      fprintf(stderr, "Error: Invalid Pauli operator %u at index %u\n",
              pString->paulis[i], i);
      free(new_bra_orbitals);
      return NULL;
    }
  }
  OuterProductC *new_oprod =
      outer_product_init_c(N, new_coef, new_ket_orbitals, new_bra_orbitals);
  free(new_ket_orbitals);
  free(new_bra_orbitals);
  return new_oprod;
}

DensityMatrixC *density_matrix_pauli_string_evolution_c(
    PauliStringC *pString, DensityMatrixC *dm, double complex epsilon) {
  if (!dm || !pString) {
    fprintf(stderr, "Error recieved NULL density matrix or Pauli string.");
  }

  if (dm->N != pString->N) {
    fprintf(stderr, "Error: Wavefunction and Pauli string have different "
                    "number of indices.\n");
  }

  DensityMatrixC *new_dm =
      density_matrix_init_with_specified_cutoff_c(dm->N, dm->cutoff);

  for (khiter_t k = kh_begin(dm->outer_products);
       k != kh_end(dm->outer_products); ++k) {
    if (kh_exist(dm->outer_products, k)) {
      OuterProductC *oprod = (OuterProductC *)kh_value(dm->outer_products, k);

      OuterProductC *new_oprod_cosh_cosh =
          evolution_helper_cosh_cosh(pString, oprod, epsilon);
      OuterProductC *new_oprod_sinh_cosh =
          evolution_helper_sinh_cosh(pString, oprod, epsilon);
      OuterProductC *new_oprod_cosh_sinh =
          evolution_helper_cosh_sinh(pString, oprod, epsilon);
      OuterProductC *new_oprod_sinh_sinh =
          evolution_helper_sinh_sinh(pString, oprod, epsilon);

      if (new_oprod_cosh_cosh) {
        density_matrix_append_outer_product_c(new_dm, new_oprod_cosh_cosh);
      }

      if (new_oprod_sinh_cosh) {
        density_matrix_append_outer_product_c(new_dm, new_oprod_sinh_cosh);
      }

      if (new_oprod_cosh_sinh) {
        density_matrix_append_outer_product_c(new_dm, new_oprod_cosh_sinh);
      }

      if (new_oprod_sinh_sinh) {
        density_matrix_append_outer_product_c(new_dm, new_oprod_sinh_sinh);
      }
    }
  }
  return new_dm;
}

DensityMatrixC *density_matrix_pauli_sum_evolution_c(PauliSumC *pSum,
                                                     DensityMatrixC *dm,
                                                     double complex epsilon) {
  if (!dm || !pSum) {
    fprintf(stderr, "Error: Received NULL density matrix or Pauli sum.\n");
    return NULL;
  }
  if (dm->N != pSum->N) {
    fprintf(stderr, "Error: Density matrix and Pauli sum have different number "
                    "of qubits.\n");
    return NULL;
  }

  DensityMatrixC *new_dm = NULL;
  int first_iteration = 1;

  for (khiter_t kp = kh_begin(pSum->pauli_strings);
       kp != kh_end(pSum->pauli_strings); ++kp) {
    if (kh_exist(pSum->pauli_strings, kp)) {
      PauliStringC *pString = (PauliStringC *)kh_value(pSum->pauli_strings, kp);
      new_dm = density_matrix_pauli_string_evolution_c(pString, dm, epsilon);

      if (!new_dm) {
        fprintf(stderr, "Error: Pauli string evolution failed.\n");
        continue;
      }

      if (!first_iteration) {
        free_density_matrix_c(dm);
      }
      dm = new_dm;
      first_iteration = 0;
    }
  }

  return dm;
}

static inline OuterProductC *
density_matrix_get_min_encoding_outer_product_c(DensityMatrixC *dm) {
  if (!dm) {
    fprintf(stderr, "Error: NULL density matrix provided.\n");
    return NULL;
  }

  OuterProductC *min_oprod = NULL;
  uint64_t min_encoding = UINT64_MAX;
  for (khiter_t k = kh_begin(dm->outer_products);
       k != kh_end(dm->outer_products); ++k) {
    if (kh_exist(dm->outer_products, k)) {
      OuterProductC *oprod = (OuterProductC *)kh_value(dm->outer_products, k);
      uint64_t encoding = oprod->encoding;
      if (encoding < min_encoding) {
        min_encoding = encoding;
        min_oprod = oprod;
      }
    }
  }
  return min_oprod;
}

DensityMatrixC *density_matrix_remove_global_phase_c(DensityMatrixC *dm) {
  if (!dm) {
    fprintf(stderr, "Error: NULL density matrix provided.\n");
    return NULL;
  }

  OuterProductC *min_oprod =
      density_matrix_get_min_encoding_outer_product_c(dm);
  double phase = carg(min_oprod->coef);
  double complex invert = cexp(-1 * (double complex)I * phase);
  return density_matrix_scalar_multiplication_c(dm, invert);
}

DensityMatrixC *density_matrix_remove_near_zero_terms_c(DensityMatrixC *dm,
                                                        double cutoff) {
  if (!dm) {
    fprintf(stderr, "Error: NULL density matrix provided.\n");
    return NULL;
  }

  DensityMatrixC *new_dm =
      density_matrix_init_with_specified_cutoff_c(dm->N, dm->cutoff);

  for (khiter_t k = kh_begin(dm->outer_products);
       k != kh_end(dm->outer_products); ++k) {
    if (kh_exist(dm->outer_products, k)) {
      OuterProductC *oprod = (OuterProductC *)kh_value(dm->outer_products, k);
      if (fabs(creal(oprod->coef)) >= cutoff ||
          fabs(cimag(oprod->coef)) >= cutoff) {
        OuterProductC *new_oprod = outer_product_init_c(
            oprod->N, oprod->coef, oprod->ket_orbitals, oprod->bra_orbitals);
        density_matrix_append_outer_product_c(new_dm, new_oprod);
      }
    }
  }

  return new_dm;
}

DensityMatrixC *
CPTP_evolution_hamiltonian_helper(PauliSumC *H, DensityMatrixC *dm, double t) {

  DensityMatrixC *Hdm = density_matrix_pauli_sum_left_multiplication_c(H, dm);
  DensityMatrixC *dmH = density_matrix_pauli_sum_right_multiplication_c(dm, H);

  DensityMatrixC *result =
      density_matrix_scalar_multiplication_c(Hdm, -(double complex)I * t);

  for (khiter_t k = kh_begin(dmH->outer_products);
       k != kh_end(dmH->outer_products); ++k) {
    if (kh_exist(dmH->outer_products, k)) {
      OuterProductC *oprod = (OuterProductC *)kh_value(dmH->outer_products, k);
      oprod =
          outer_product_scalar_multiplication_c(oprod, (double complex)I * t);
      density_matrix_append_outer_product_c(result, oprod);
    }
  }

  free_density_matrix_c(Hdm);
  free_density_matrix_c(dmH);

  return result;
}

DensityMatrixC *CPTP_evolution_lindblad_helper(PauliSumC *L, DensityMatrixC *dm,
                                               double t) {

  DensityMatrixC *result =
      density_matrix_init_with_specified_cutoff_c(dm->N, dm->cutoff);

  PauliSumC *Ldag = pauli_sum_adjoint_c(L);
  PauliSumC *LdagL = pauli_sum_multiplication_c(Ldag, L);

  DensityMatrixC *LdmLdag =
      density_matrix_pauli_sum_left_multiplication_c(L, dm);
  LdmLdag = density_matrix_pauli_sum_right_multiplication_c(LdmLdag, Ldag);

  DensityMatrixC *LdagLdm =
      density_matrix_pauli_sum_left_multiplication_c(LdagL, dm);
  DensityMatrixC *dmLadL =
      density_matrix_pauli_sum_right_multiplication_c(dm, LdagL);

  for (khiter_t k = kh_begin(LdmLdag->outer_products);
       k != kh_end(LdmLdag->outer_products); ++k) {
    if (kh_exist(LdmLdag->outer_products, k)) {
      OuterProductC *oprod =
          (OuterProductC *)kh_value(LdmLdag->outer_products, k);
      oprod = outer_product_scalar_multiplication_c(oprod, t);
      density_matrix_append_outer_product_c(result, oprod);
    }
  }

  for (khiter_t k = kh_begin(LdagLdm->outer_products);
       k != kh_end(LdagLdm->outer_products); ++k) {
    if (kh_exist(LdagLdm->outer_products, k)) {
      OuterProductC *oprod =
          (OuterProductC *)kh_value(LdagLdm->outer_products, k);
      oprod = outer_product_scalar_multiplication_c(oprod, -0.5 * t);
      density_matrix_append_outer_product_c(result, oprod);
    }
  }

  for (khiter_t k = kh_begin(dmLadL->outer_products);
       k != kh_end(dmLadL->outer_products); ++k) {
    if (kh_exist(dmLadL->outer_products, k)) {
      OuterProductC *oprod =
          (OuterProductC *)kh_value(dmLadL->outer_products, k);
      oprod = outer_product_scalar_multiplication_c(oprod, -0.5 * t);
      density_matrix_append_outer_product_c(result, oprod);
    }
  }

  free_pauli_sum_c(Ldag);
  free_pauli_sum_c(LdagL);
  free_density_matrix_c(LdmLdag);
  free_density_matrix_c(LdagLdm);
  free_density_matrix_c(dmLadL);
  return result;
}

DensityMatrixC *density_matrix_CPTP_evolution_c(PauliSumC *H, PauliSumC **Ls,
                                                unsigned int num_L,
                                                DensityMatrixC *dm, double t) {

  DensityMatrixC *result =
      density_matrix_init_with_specified_cutoff_c(dm->N, dm->cutoff);

  for (khiter_t k = kh_begin(dm->outer_products);
       k != kh_end(dm->outer_products); ++k) {
    if (kh_exist(dm->outer_products, k)) {
      OuterProductC *oprod = (OuterProductC *)kh_value(dm->outer_products, k);
      OuterProductC *new_oprod = outer_product_init_c(
          oprod->N, oprod->coef, oprod->ket_orbitals, oprod->bra_orbitals);
      density_matrix_append_outer_product_c(result, new_oprod);
    }
  }

  DensityMatrixC *hamiltonian_contribution =
      CPTP_evolution_hamiltonian_helper(H, dm, t);

  for (khiter_t k = kh_begin(hamiltonian_contribution->outer_products);
       k != kh_end(hamiltonian_contribution->outer_products); ++k) {
    if (kh_exist(hamiltonian_contribution->outer_products, k)) {
      OuterProductC *oprod = (OuterProductC *)kh_value(
          hamiltonian_contribution->outer_products, k);
      OuterProductC *new_oprod = outer_product_init_c(
          oprod->N, oprod->coef, oprod->ket_orbitals, oprod->bra_orbitals);
      density_matrix_append_outer_product_c(result, new_oprod);
    }
  }

  for (unsigned int i = 0; i < num_L; i++) {
    DensityMatrixC *lindblad_contribution =
        CPTP_evolution_lindblad_helper(Ls[i], dm, t);

    for (khiter_t k = kh_begin(lindblad_contribution->outer_products);
         k != kh_end(lindblad_contribution->outer_products); ++k) {
      if (kh_exist(lindblad_contribution->outer_products, k)) {
        OuterProductC *oprod =
            (OuterProductC *)kh_value(lindblad_contribution->outer_products, k);
        OuterProductC *new_oprod = outer_product_init_c(
            oprod->N, oprod->coef, oprod->ket_orbitals, oprod->bra_orbitals);
        density_matrix_append_outer_product_c(result, new_oprod);
      }
    }
  }

  return result;
}