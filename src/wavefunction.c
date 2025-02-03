#include "wavefunction.h"

void free_slater_determinant_c(SlaterDeterminantC *sdet) {
    free(sdet->orbitals);
    free(sdet);
}

SlaterDeterminantC *slater_determinant_init_c(const unsigned int N, double complex coef, unsigned int orbitals[]) {
    SlaterDeterminantC *sdet;
    unsigned int encoding;
    unsigned int i;

    sdet = (SlaterDeterminantC *) malloc(sizeof(SlaterDeterminantC));
    if (!sdet) {
        fprintf(stderr, "Malloc failed for SlaterDeterminant\n");
        return NULL;
    }

    sdet->N = N;
    sdet->coef = coef;

    sdet->orbitals = (unsigned int *) malloc(N * sizeof(unsigned int));
    if (!sdet->orbitals) {
        fprintf(stderr, "Malloc failed for orbitals\n");
        free(sdet);
        return NULL;
    }

    encoding = 0;
    for (i = 0; i < N; i++) {
        sdet->orbitals[i] = orbitals[i];
        if (orbitals[i]) {
            encoding += 1 << (N - 1 - i);
        }
    }
    sdet->encoding = encoding; 

    return sdet;
}

char *slater_determinant_to_string_c(SlaterDeterminantC *sdet, char bra_or_ket) {
    size_t buffer_size;
    char *buffer;
    unsigned int i;
    char orbital_char[2]; // To hold a single orbital (1 digit + null terminator)
    
    // Calculate buffer size
    buffer_size = 40  // for coef
                + 1   // for |
                + sdet->N  // for orbitals
                + 1   // for >
                + 1;  // for terminator

    // Allocate memory for the buffer
    buffer = (char *) malloc(buffer_size);
    if (!buffer) {
        fprintf(stderr, "Malloc failed for slater_determinant_to_string_c\n");
        return NULL;
    }

    // Write the coefficient part to the buffer
    snprintf(buffer, buffer_size, "(%.4lf + %.4lfi)", creal(sdet->coef), cimag(sdet->coef));

    // Append the `|` character
    if (bra_or_ket == 'b') {
        strcat(buffer, "<");
    } else if (bra_or_ket == 'k') {
        strcat(buffer, "|");
    } else {
        fprintf(stderr, "Error: Invalid bra_or_ket character.\n");
        free(buffer);
        return NULL;
    }

    // Append the orbital occupations
    for (i = 0; i < sdet->N; i++) {
        snprintf(orbital_char, sizeof(orbital_char), "%d", sdet->orbitals[i]);
        strcat(buffer, orbital_char);
    }

    // Append the `>` character
    if (bra_or_ket == 'b') {
        strcat(buffer, "|");
    } else if (bra_or_ket == 'k') {
        strcat(buffer, ">");
    }

    return buffer;
}

SlaterDeterminantC *slater_determinant_scalar_multiplication_c(SlaterDeterminantC *sdet, double complex scalar) {
    SlaterDeterminantC *new_sdet = slater_determinant_init_c(sdet->N, sdet->coef * scalar, sdet->orbitals);
    return new_sdet;
}

SlaterDeterminantC *slater_determinant_adjoint_c(SlaterDeterminantC *sdet) {
    double complex coef = sdet->coef;
    SlaterDeterminantC *new_sdet = slater_determinant_init_c(sdet->N, (double complex) (creal(coef) + -1 * cimag(coef) * (double complex) I), sdet->orbitals);
    return new_sdet;
}

double slater_determinant_comparison_c(SlaterDeterminantC *bra, SlaterDeterminantC *ket) {
    if (bra->N != ket->N) {
        fprintf(stderr, "Error: Slater determinants have different number of orbitals.\n");
        return 0;
    }
    return (double) (bra->encoding == ket->encoding);
}

double complex slater_dermininant_multiplication_c(SlaterDeterminantC *bra, SlaterDeterminantC *ket) {
    return slater_determinant_comparison_c(bra, ket) * bra->coef * ket->coef;
}

SlaterDeterminantC *slater_determinant_pauli_string_multiplication_c(PauliStringC *pString, SlaterDeterminantC *sdet) {

    unsigned int N;
    unsigned int *new_orbitals;
    double complex new_coef;
    SlaterDeterminantC *new_sdet;

    if (pString->N != sdet->N) {
        fprintf(stderr, "Error: Pauli strings have different number of qubits.\n");
        return NULL;
    }

    N = sdet->N;
    new_orbitals = (unsigned int *) malloc(sdet->N * sizeof(int));
    new_coef = pString->coef * sdet->coef;

    for (unsigned int i = 0; i < N; i++) {
        switch(pString->paulis[i]) {
            case 0: // I
                new_orbitals[i] = sdet->orbitals[i];
                break;
            case 1: // X
                new_orbitals[i] = sdet->orbitals[i] ^ 1;
                break;
            case 2: // Y
                new_orbitals[i] = sdet->orbitals[i] ^ 1;
                new_coef = new_coef * ((double complex) I) * (1 - 2 * (int) sdet->orbitals[i]);
                break;
            case 3: // Z
                new_orbitals[i] = sdet->orbitals[i];
                new_coef = new_coef * (1 - 2 * (int) sdet->orbitals[i]);
                break;
        }
    }
    new_sdet = slater_determinant_init_c(N, new_coef, new_orbitals);
    free(new_orbitals);
    return new_sdet;
}

void free_wavefunction_c(WavefunctionC *wfn) {
    if (wfn == NULL) return;

    if (wfn->slater_determinants) {
        for (unsigned int i = 0; i < wfn->s; i++) {
            if (wfn->slater_determinants[i]) {
                free_slater_determinant_c(wfn->slater_determinants[i]);
                wfn->slater_determinants[i] = NULL;
            }
        }
        free(wfn->slater_determinants);
        wfn->slater_determinants = NULL;
    }

    free(wfn);
}

WavefunctionC *wavefunction_init_c(unsigned int s_max) {
    WavefunctionC *wfn = (WavefunctionC *) malloc(sizeof(WavefunctionC));
    wfn->s_max = s_max;
    wfn->s = 0;
    wfn->slater_determinants = (SlaterDeterminantC **) malloc(s_max * sizeof(SlaterDeterminantC *));
    return wfn;
}

WavefunctionC *wavefunction_realloc_c(WavefunctionC *wfn, unsigned int new_s_max) {

    WavefunctionC *new_wfn;

    if (wfn->s_max >= new_s_max) {
        fprintf(stderr, "Error: Realloc must allocate more space\n");
        return NULL;
    }

    new_wfn = wavefunction_init_c(new_s_max);
    if (!new_wfn) return NULL; 

    for (unsigned int i = 0; i < wfn->s; i++) {
        new_wfn->slater_determinants[i] = wfn->slater_determinants[i];
    }
    new_wfn->s = wfn->s;

    free(wfn->slater_determinants);
    wfn->slater_determinants = NULL;


    return new_wfn;
}

WavefunctionC *wavefunction_append_slater_determinant_c(WavefunctionC *wfn, SlaterDeterminantC *sdet) {

    WavefunctionC *new_wfn;
    unsigned int appended_at;

    if (wfn->s == wfn->s_max) {
        new_wfn = wavefunction_realloc_c(wfn, 2 * wfn->s_max);
        free(wfn);
        wfn = new_wfn;
    }

    appended_at = wfn->s;
    for (unsigned int i = 0; i < wfn->s; i++) {
        if (sdet->encoding < wfn->slater_determinants[i]->encoding) {
            for (unsigned int j = 0; j < wfn->s - i; j++) {
                wfn->slater_determinants[wfn->s - j] = wfn->slater_determinants[wfn->s - j - 1];
            }
            appended_at = i;
            wfn->slater_determinants[appended_at] = sdet;
            wfn->s += 1;
            break;
        } else if (sdet->encoding == wfn->slater_determinants[i]->encoding) {
            appended_at = i;
            wfn->slater_determinants[appended_at]->coef += sdet->coef;
            free_slater_determinant_c(sdet);
            break;
        } 
    }
    if (appended_at == wfn->s) {
        wfn->slater_determinants[appended_at] = sdet;
        wfn->s += 1;
    } 
    return wfn;
}

double wavefunction_norm_c(WavefunctionC *wfn) {
    double norm = 0;
    for (unsigned int i = 0; i < wfn->s; i++) {
        double complex coef = wfn->slater_determinants[i]->coef;
        norm = norm + creal(coef) * creal(coef) + cimag(coef) * cimag(coef);
    }
    return sqrt(norm);
}

WavefunctionC *wavefunction_scalar_multiplication_c(WavefunctionC *wfn, double complex scalar) {
    WavefunctionC *new_wfn = wavefunction_init_c(wfn->s_max);

    for (unsigned int i = 0; i < wfn->s; i++) {
        new_wfn->slater_determinants[i] = slater_determinant_scalar_multiplication_c(wfn->slater_determinants[i], scalar);
    }
    new_wfn->s = wfn->s;

    return new_wfn;
}

WavefunctionC *wavefunction_adjoint_c(WavefunctionC *wfn) {
    WavefunctionC *new_wfn = wavefunction_init_c(wfn->s_max);

    for (unsigned int i = 0; i < wfn->s; i++) {
        new_wfn->slater_determinants[i] = slater_determinant_adjoint_c(wfn->slater_determinants[i]);
    }
    new_wfn->s = wfn->s;

    return new_wfn;
}

double complex wavefunction_multiplication_c(WavefunctionC *bra, WavefunctionC *ket) {
    double complex product = 0;
    for (unsigned int i = 0; i < bra->s; i++) {
        for (unsigned int j = 0; j < ket->s; j++) {
            product = product + slater_dermininant_multiplication_c(bra->slater_determinants[i], ket->slater_determinants[j]);
        }
    }
    return product;
}

char *wavefunction_to_string_c(WavefunctionC *wfn, char bra_or_ket) {

    char *sdet_str;
    char *buffer;
    char *first_sdet_str;
    size_t buffer_size;

    if (!wfn) {
        fprintf(stderr, "Error: Received NULL wavefunction.\n");
        return NULL;
    }

    if (wfn->s == 0) {
        char *empty_str = (char *) malloc(1);
        if (!empty_str) return NULL;
        empty_str[0] = '\0';
        return empty_str;  
    }

    buffer_size = 1; 

    for (unsigned int i = 0; i < wfn->s; i++) {
        sdet_str = slater_determinant_to_string_c(wfn->slater_determinants[i], bra_or_ket);
        if (!sdet_str) {
            fprintf(stderr, "Error: Failed to allocate memory for Slater determinant string.\n");
            return NULL;
        }
        buffer_size += strlen(sdet_str) + 3; 
        free(sdet_str);
    }

    buffer = (char *) malloc(buffer_size);
    buffer[0] = '\0'; 

    first_sdet_str = slater_determinant_to_string_c(wfn->slater_determinants[0], bra_or_ket);
    if (!first_sdet_str) return NULL;
    strcpy(buffer, first_sdet_str);
    free(first_sdet_str);

    for (unsigned int i = 1; i < wfn->s; i++) {
        strcat(buffer, " + ");
        sdet_str = slater_determinant_to_string_c(wfn->slater_determinants[i], bra_or_ket);
        if (!sdet_str) return NULL;
        strcat(buffer, sdet_str);
        free(sdet_str); 
    }

    return buffer;
}

WavefunctionC *wavefunction_pauli_string_multiplication_c(PauliStringC *pString, WavefunctionC *wfn) {
    WavefunctionC *new_wfn = wavefunction_init_c(wfn->s_max);

    for (unsigned int i = 0; i < wfn->s; i++) {
        new_wfn = wavefunction_append_slater_determinant_c(new_wfn, slater_determinant_pauli_string_multiplication_c(pString, wfn->slater_determinants[i]));
    }

    return new_wfn;
}

WavefunctionC *wavefunction_pauli_sum_multiplication_c(PauliSumC *pSum, WavefunctionC *wfn) {
    WavefunctionC *new_wfn = wavefunction_init_c(wfn->s_max);

    for (unsigned int i = 0; i < pSum->p; i++) {
        for (unsigned int j = 0; j < wfn->s; j++) {
            new_wfn = wavefunction_append_slater_determinant_c(new_wfn, slater_determinant_pauli_string_multiplication_c(pSum->pauli_strings[i], wfn->slater_determinants[j]));
        }
    }

    return new_wfn;
}

static inline SlaterDeterminantC *evolution_helper_cosh(PauliStringC *pString, SlaterDeterminantC *sdet, double epsilon) {
    unsigned int N = sdet->N;
    double complex new_coef = (double complex) (ccosh(pString->coef * epsilon) * sdet->coef);
    return slater_determinant_init_c(N, new_coef, sdet->orbitals);
}

static inline SlaterDeterminantC *evolution_helper_sinh(PauliStringC *pString, SlaterDeterminantC *sdet, double epsilon) {

    unsigned int N; 
    unsigned int *new_orbitals;
    double complex new_coef;
    SlaterDeterminantC *new_sdet;

    if (pString->N != sdet->N) {
        fprintf(stderr, "Error: Pauli strings have different number of qubits.\n");
        return NULL;
    }

    N = sdet->N;
    new_orbitals = (unsigned int *) malloc(sdet->N * sizeof(int));
    new_coef = (csinh(pString->coef * epsilon) * sdet->coef);

    for (unsigned int i = 0; i < N; i++) {
        switch(pString->paulis[i]) {
            case 0: // I
                new_orbitals[i] = sdet->orbitals[i];
                break;
            case 1: // X
                new_orbitals[i] = sdet->orbitals[i] ^ 1;
                break;
            case 2: // Y
                new_orbitals[i] = sdet->orbitals[i] ^ 1;
                new_coef = new_coef * (double complex) I * (1 - 2 * (int) sdet->orbitals[i]);
                break;
            case 3: // Z
                new_orbitals[i] = sdet->orbitals[i];
                new_coef = new_coef * (1 - 2 * (int) sdet->orbitals[i]);
                break;
        }
    }
    new_sdet = slater_determinant_init_c(N, new_coef, new_orbitals);
    free(new_orbitals);
    return new_sdet;
}

WavefunctionC *wavefunction_pauli_string_evolution_c(PauliStringC *pString, WavefunctionC *wfn, double epsilon) {
    WavefunctionC *new_wfn = wavefunction_init_c(wfn->s_max);

    for (unsigned int i = 0; i < wfn->s; i++) {
        new_wfn = wavefunction_append_slater_determinant_c(new_wfn, evolution_helper_cosh(pString, wfn->slater_determinants[i], epsilon));
        new_wfn = wavefunction_append_slater_determinant_c(new_wfn, evolution_helper_sinh(pString, wfn->slater_determinants[i], epsilon));
    }

    return new_wfn;
}

WavefunctionC *wavefunction_pauli_sum_evolution_c(PauliSumC *pSum, WavefunctionC *wfn, double epsilon) {
    WavefunctionC *new_wfn;

    for (unsigned int i = 0; i < pSum->p; i++) {
        new_wfn = wavefunction_pauli_string_evolution_c(pSum->pauli_strings[i], wfn, epsilon);
        if (i > 0) {
            free_wavefunction_c(wfn);
        }
        wfn = new_wfn;
    }

    return wfn;
}
