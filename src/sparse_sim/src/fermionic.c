#include "fermionic.h"

void free_fermionic_operator_c(FermionicOperatorC *fop) {
  free(fop->coefs);
  free(fop->opstring);
  free(fop->ordering);
  free(fop);
}

FermionicOperatorC *fermionic_operator_init_c(unsigned int N, char opstring[],
                                              unsigned int ordering[],
                                              double complex *coefs) {

  FermionicOperatorC *fop = malloc(sizeof *fop);
  if (!fop) {
    fprintf(stderr, "Malloc failed for FermionicOperator\n");
    return NULL;
  }

  fop->N = N;

  size_t len = strlen(opstring);

  fop->opstring = malloc(len + 1);
  if (!fop->opstring) {
    fprintf(stderr, "Malloc failed for opstring\n");
    free(fop);
    return NULL;
  }
  memcpy(fop->opstring, opstring, len + 1);

  fop->ordering = malloc(len * sizeof *fop->ordering);
  if (!fop->ordering) {
    fprintf(stderr, "Malloc failed for ordering\n");
    free(fop->opstring);
    free(fop);
    return NULL;
  }
  memcpy(fop->ordering, ordering, len * sizeof(unsigned int));

  fop->coefs = coefs;

  return fop;
}

char *fermionic_operator_to_string_c(FermionicOperatorC *fop) {

  // Number of ops per term
  size_t L = strlen(fop->opstring);

  // Max total number of terms
  size_t total_terms = 1;
  for (size_t i = 0; i < L; i++) {
    total_terms *= fop->N;
  }

  // Number of digits needed to print orbital indices
  size_t digits = 1;
  if (fop->N > 1) {
    digits = (size_t)floor(log10((double)(fop->N - 1))) + 1;
  }

  size_t COEF_LEN = 40;                           // (%.6f%+.6fi) worst case
  size_t term_len = 1 +                           //'\t'
                    COEF_LEN + L * (4 + digits) + // " +a_<idx>" per operator
                    1;                            // '\n'

  // Allocate string
  size_t cap = total_terms * term_len + 1;
  char *out = malloc(cap);
  if (!out) {
    fprintf(stderr, "Malloc failed for fermionic operator string\n");
    return NULL;
  }

  size_t out_idx = 0;
  int first_term = 1;

  unsigned int idx[L];
  memset(idx, 0, L * sizeof(unsigned int));

  for (size_t flat_idx = 0; flat_idx < total_terms; flat_idx++) {
    double complex c = fop->coefs[flat_idx];

    if (cabs(c) > 1e-12) {
      if (!first_term) {
        out[out_idx++] = '\n';
        out[out_idx++] = '\t';
        out[out_idx++] = ' ';
        out[out_idx++] = '+';
        out[out_idx++] = ' ';
      } else {
        out[out_idx++] = '\t';
        out[out_idx++] = ' ';
        out[out_idx++] = ' ';
        out[out_idx++] = ' ';
      }
      first_term = 0;

      // Append coefficient
      out_idx += snprintf(out + out_idx, cap - out_idx, "(%.6f%+.6fi)",
                          creal(c), cimag(c));

      // Append operators
      for (size_t k = 0; k < L; k++) {
        unsigned int site = idx[fop->ordering[k]];
        out_idx += snprintf(out + out_idx, cap - out_idx, " %ca_%u",
                            fop->opstring[k], site);
      }
    }

    // Update indices
    int k = (int)L - 1;
    while (k >= 0) {
      idx[k]++;
      if (idx[k] < fop->N)
        break;
      idx[k] = 0;
      k--;
    }
  }

  // If empty
  if (first_term) {
    strcpy(out, "\t0");
  } else {
    out[out_idx] = '\0';
  }

  return out;
}

FermionicOperatorC *
fermionic_operator_scalar_multiplication_c(FermionicOperatorC *fop,
                                           double complex scalar) {

  size_t L = strlen(fop->opstring);
  size_t total_terms = 1;
  for (size_t i = 0; i < L; i++) {
    total_terms *= fop->N;
  }

  double complex *new_coefs = malloc(total_terms * sizeof(double complex));
  if (!new_coefs) {
    fprintf(stderr, "Malloc failed for new coefficients\n");
    return NULL;
  }
  for (size_t i = 0; i < total_terms; i++) {
    new_coefs[i] = fop->coefs[i] * scalar;
  }

  return fermionic_operator_init_c(fop->N, fop->opstring, fop->ordering,
                                   new_coefs);
}

FermionicOperatorC *fermionic_operator_adjoint_c(FermionicOperatorC *fop) {
  size_t L = strlen(fop->opstring);
  size_t total_terms = 1;
  for (size_t i = 0; i < L; i++) {
    total_terms *= fop->N;
  }

  FermionicOperatorC *adj = malloc(sizeof *adj);
  if (!adj)
    return NULL;

  adj->N = fop->N;

  char *new_opstring = malloc(L + 1);
  if (!new_opstring) {
    free(adj);
    return NULL;
  }
  for (size_t k = 0; k < L; k++) {
    char c = fop->opstring[L - 1 - k];
    new_opstring[k] = (c == '+') ? '-' : '+';
  }
  new_opstring[L] = '\0';
  adj->opstring = new_opstring;

  unsigned int *new_ordering = malloc(L * sizeof *new_ordering);
  if (!new_ordering) {
    free(new_opstring);
    free(adj);
    return NULL;
  }
  for (size_t k = 0; k < L; k++) {
    new_ordering[k] = fop->ordering[k];
  }
  adj->ordering = new_ordering;

  double complex *new_coefs = malloc(total_terms * sizeof *new_coefs);
  if (!new_coefs) {
    free(new_ordering);
    free(new_opstring);
    free(adj);
    return NULL;
  }

  unsigned int inv_ordering[L];
  for (size_t k = 0; k < L; k++) {
    inv_ordering[fop->ordering[k]] = k;
  }

  unsigned int perm[L];
  for (size_t a = 0; a < L; a++) {
    size_t k = inv_ordering[a];
    size_t k2 = L - 1 - k;
    perm[a] = fop->ordering[k2];
  }

  unsigned int idx[L];
  memset(idx, 0, L * sizeof(unsigned int));

  for (size_t flat_idx = 0; flat_idx < total_terms; flat_idx++) {

    /* compute flat index for adjoint tensor */
    size_t rflat_idx = 0;
    for (size_t a = 0; a < L; a++) {
      rflat_idx = rflat_idx * fop->N + idx[perm[a]];
    }

    new_coefs[rflat_idx] = conj(fop->coefs[flat_idx]);

    /* increment original multi-index (base-N counter) */
    int k = (int)L - 1;
    while (k >= 0) {
      idx[k]++;
      if (idx[k] < fop->N)
        break;
      idx[k] = 0;
      k--;
    }
  }

  adj->coefs = new_coefs;

  return adj;
}

SlaterDeterminantC *
slater_determinant_single_term_fermionic_operator_multiplication_c(
    SlaterDeterminantC *sdet, char opstring[], unsigned int unordered_idx[],
    double complex coef) {

  size_t L = strlen(opstring);
  double sign = 1;

  unsigned int *new_orbitals = (unsigned int *)malloc(sdet->N * sizeof(int));

  for (unsigned int i = 0; i < sdet->N; i++) {
    new_orbitals[i] = sdet->orbitals[i];
  }

  for (size_t k = 0; k < L; k++) {
    char op = opstring[L - k - 1];
    unsigned int idx = unordered_idx[L - k - 1];

    if (op == '+') {
      // Creation operator
      if (new_orbitals[idx] == 1) {
        // Occupied, results in zero
        free(new_orbitals);
        return NULL;
      } else {
        new_orbitals[idx] = 1;
        for (unsigned int j = 0; j < idx; j++) {
          if (new_orbitals[j] == 1) {
            sign *= -1;
          }
        }
      }
    } else if (op == '-') {
      // Annihilation operator
      if (new_orbitals[idx] == 0) {
        // Unoccupied, results in zero
        free(new_orbitals);
        return NULL;
      } else {
        new_orbitals[idx] = 0;
        for (unsigned int j = 0; j < idx; j++) {
          if (new_orbitals[j] == 1) {
            sign *= -1;
          }
        }
      }
    } else {
      fprintf(stderr, "Error: Invalid fermionic operator '%c'.\n", op);
      free(new_orbitals);
      return NULL;
    }
  }

  SlaterDeterminantC *new_sdet = slater_determinant_init_c(
      sdet->N, sdet->coef * coef * sign, new_orbitals);
  if (!new_sdet) {
    fprintf(stderr, "Malloc failed for new slater determinant\n");
    free(new_orbitals);
    return NULL;
  }
  return new_sdet;
}

WavefunctionC *wavefunction_single_term_fermionic_operator_multiplication_c(
    WavefunctionC *wfn, char opstring[], unsigned int unordered_idx[],
    double complex coef) {

  size_t L = strlen(opstring);
  for (size_t i = 0; i < L; i++) {
    if (unordered_idx[i] >= wfn->N) {
      fprintf(stderr, "Error: Fermionic operator index %u out of bounds.\n",
              unordered_idx[i]);
      return NULL;
    }
  }

  WavefunctionC *new_wfn =
      wavefunction_init_with_specified_cutoff_c(wfn->N, wfn->cutoff);
  if (!new_wfn) {
    fprintf(stderr, "Malloc failed for new wavefunction\n");
    return NULL;
  }

  khiter_t k;
  for (k = kh_begin(wfn->slater_determinants);
       k != kh_end(wfn->slater_determinants); ++k) {
    if (kh_exist(wfn->slater_determinants, k)) {
      SlaterDeterminantC *sdet =
          (SlaterDeterminantC *)kh_value(wfn->slater_determinants, k);
      SlaterDeterminantC *new_sdet =
          slater_determinant_single_term_fermionic_operator_multiplication_c(
              sdet, opstring, unordered_idx, coef);

      if (new_sdet) {
        wavefunction_append_slater_determinant_c(new_wfn, new_sdet);
      }
    }
  }

  return new_wfn;
}

FermionicOperatorC *wavefunction_termwise_fermionic_operator_expectation_c(
    WavefunctionC *wfn, FermionicOperatorC *fop) {

  size_t L = strlen(fop->opstring);
  size_t total_terms = 1;
  for (size_t i = 0; i < L; i++) {
    total_terms *= fop->N;
  }

  unsigned int idx[L];
  memset(idx, 0, L * sizeof(unsigned int));
  double complex *expectation_coefs =
      malloc(total_terms * sizeof(double complex));

  WavefunctionC *bra = wavefunction_adjoint_c(wfn);

  for (size_t flat_idx = 0; flat_idx < total_terms; flat_idx++) {
    unsigned int unordered_idx[L];
    for (unsigned int k = 0; k < L; k++) {
      unordered_idx[k] = idx[fop->ordering[k]];
    }

    WavefunctionC *new_wfn =
        wavefunction_single_term_fermionic_operator_multiplication_c(
            wfn, fop->opstring, unordered_idx, fop->coefs[flat_idx]);
    if (new_wfn) {
      if (new_wfn->s != 0) {
        double complex expectation =
            wavefunction_multiplication_c(bra, new_wfn);
        expectation_coefs[flat_idx] = expectation;
      } else {
        expectation_coefs[flat_idx] = 0.0 + 0.0 * I;
      }

    } else {
      free_wavefunction_c(bra);
      free(expectation_coefs);
      fprintf(stderr, "Wavefunction multiplication returned NULL\n");
      return NULL;
    }

    free_wavefunction_c(new_wfn);

    int k = (int)L - 1;
    while (k >= 0) {
      idx[k]++;
      if (idx[k] < fop->N)
        break;
      idx[k] = 0;
      k--;
    }
  }
  free_wavefunction_c(bra);

  return fermionic_operator_init_c(fop->N, fop->opstring, fop->ordering,
                                   expectation_coefs);
}

double complex wavefunction_fermionic_operator_expectation_c(
    WavefunctionC *wfn, FermionicOperatorC *fop) {

  size_t L = strlen(fop->opstring);
  size_t total_terms = 1;
  for (size_t i = 0; i < L; i++) {
    total_terms *= fop->N;
  }

  unsigned int idx[L];
  memset(idx, 0, L * sizeof(unsigned int));
  double complex expectation = 0.0 + 0.0 * I;

  WavefunctionC *bra = wavefunction_adjoint_c(wfn);

  for (size_t flat_idx = 0; flat_idx < total_terms; flat_idx++) {
    unsigned int unordered_idx[L];
    for (unsigned int k = 0; k < L; k++) {
      unordered_idx[k] = idx[fop->ordering[k]];
    }

    WavefunctionC *new_wfn =
        wavefunction_single_term_fermionic_operator_multiplication_c(
            wfn, fop->opstring, unordered_idx, fop->coefs[flat_idx]);
    if (new_wfn) {
      expectation += wavefunction_multiplication_c(bra, new_wfn);
    } else {
      free_wavefunction_c(bra);
      fprintf(stderr, "Wavefunction multiplication returned NULL\n");
      return CMPLX(NAN, NAN);
    }

    free_wavefunction_c(new_wfn);

    int k = (int)L - 1;
    while (k >= 0) {
      idx[k]++;
      if (idx[k] < fop->N)
        break;
      idx[k] = 0;
      k--;
    }
  }
  free_wavefunction_c(bra);

  return expectation;
}