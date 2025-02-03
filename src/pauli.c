#include "pauli.h"

PauliStringC *pauli_string_init_as_chars_c(unsigned int N, double complex coef, char paulis[]) {
    PauliStringC *pString = (PauliStringC *) malloc(sizeof(PauliStringC));
    pString->N = N;
    pString->coef = coef;

    pString->paulis = (unsigned int *) malloc(N * sizeof(unsigned int));
    for (unsigned int i = 0; i < N; i++) {
        switch (paulis[i]) {
            case 'I': 
                pString->paulis[i] = 0; 
                break;
            case 'X': 
                pString->paulis[i] = 1; 
                break;
            case 'Y': 
                pString->paulis[i] = 2; 
                break;
            case 'Z': 
                pString->paulis[i] = 3; 
                break;
            default: 
                fprintf(stderr, "Error: Invalid Pauli operator '%c'.\n", paulis[i]);
                free(pString->paulis);
                free(pString);
                return NULL;
        }
    }
    return pString;
}

PauliStringC *pauli_string_init_as_ints_c(unsigned int N, double complex coef, unsigned int paulis[]) {
    PauliStringC *pString = (PauliStringC *) malloc(sizeof(PauliStringC));
    pString->N = N;
    pString->coef = coef;

    pString->paulis = (unsigned int *) malloc(N * sizeof(unsigned int));
    for (unsigned int i = 0; i < N; i++) {
        pString->paulis[i] = paulis[i];
    }
    return pString;
}

void free_pauli_string_c(PauliStringC *pString) {
    free(pString->paulis);
    free(pString);
}

char *pauli_string_to_string_c(PauliStringC *pString) {

    size_t buffer_size = 40 // for coef
                        + pString->N // for paulis
                        + 1; // for terminator

    // Allocate memory for the buffer
    char *buffer = (char *) malloc(buffer_size);

    char pauli_as_char[] = {'I', 'X', 'Y', 'Z'};

    // Write the coefficient part to the buffer
    snprintf(buffer, buffer_size, "(%.4lf + %.4lfi)", creal(pString->coef), cimag(pString->coef));

    // Append the Pauli operators
    for (unsigned int i = 0; i < pString->N; i++) {
        char pauli_char[2]; // To hold a single Pauli operator (1 digit + null terminator)
        pauli_char[0] = pauli_as_char[pString->paulis[i]];
        pauli_char[1] = '\0';
        strcat(buffer, pauli_char);
    }

    return buffer;
}

PauliStringC *pauli_string_scalar_multiplication_c(PauliStringC *pString, double complex scalar) {
    PauliStringC *new_pString = pauli_string_init_as_ints_c(pString->N, pString->coef * scalar, pString->paulis);
    return new_pString;
}

PauliStringC *pauli_string_adjoint_c(PauliStringC *pString) {
    double complex coef = pString->coef;
    PauliStringC *new_pString = pauli_string_init_as_ints_c(pString->N, (double complex) (creal(coef) + ((double complex) -I) * cimag(coef)), pString->paulis);
    return new_pString;
}

double pauli_string_comparison_c(PauliStringC *left, PauliStringC *right) {
    double comparison = 1;

    if (left->N != right->N) {
        fprintf(stderr, "Error: Pauli strings have different number of qubits.\n");
        return 0;
    }

    for (unsigned int i = 0; i < left->N; i++) {
        if (left->paulis[i] != right->paulis[i]) {
            comparison = 0;
            break;
        } 
    }
    return comparison;
}

PauliStringC *pauli_string_multiplication_c(PauliStringC *left, PauliStringC *right) {
    double complex coef;
    unsigned int N;
    unsigned int caylay_pauli[4][4] = {
        {0, 1, 2, 3},
        {1, 0, 3, 2},
        {2, 3, 0, 1},
        {3, 2, 1, 0}
    };

    double complex caylay_coef[4][4] = {
        {1.0, 1.0, 1.0, 1.0},
        {1.0, 1, (double complex) I, (double complex) -I},
        {1.0, (double complex) -I , 1.0, (double complex) I},
        {1.0, (double complex) I, (double complex) -I, 1.0}
    };

    unsigned int *paulis;

    PauliStringC *product;

    if (left->N != right->N) {
        fprintf(stderr, "Error: Pauli strings have different number of qubits.\n");
        return NULL;
    }

    coef = left->coef * right->coef;

    N = left->N;

    paulis = (unsigned int *) malloc(N * sizeof(unsigned int));

    for (unsigned int i = 0; i < N; i++) {
        paulis[i] = caylay_pauli[left->paulis[i]][right->paulis[i]];
        coef = coef * caylay_coef[left->paulis[i]][right->paulis[i]];
    }
    
    product = pauli_string_init_as_ints_c(N, coef, paulis);
    free(paulis);
    return product;
}

PauliSumC *pauli_sum_init_c(unsigned int p_max) {
    PauliSumC *pSum = (PauliSumC *) malloc(sizeof(PauliSumC));
    pSum->p_max = p_max;
    pSum->p = 0;
    pSum->pauli_strings = (PauliStringC **) malloc(p_max * sizeof(PauliStringC *));
    return pSum;
}

void free_pauli_sum_c(PauliSumC *pSum) {
    for (unsigned int i = 0; i < pSum->p; i++) {
        free_pauli_string_c(pSum->pauli_strings[i]);
    }
    free(pSum->pauli_strings);
    free(pSum);
}

PauliSumC *pauli_sum_realloc_c(PauliSumC *pSum, unsigned int new_p_max) {

    PauliSumC *new_pSum;

    if (pSum->p_max >= new_p_max) {
        fprintf(stderr, "Error: Realloc must allocate more space\n");
        return NULL;
    }

    new_pSum = pauli_sum_init_c(new_p_max);
    for (unsigned int i = 0; i < pSum->p; i++) {
        new_pSum->pauli_strings[i] = pSum->pauli_strings[i];
    }
    new_pSum->p = pSum->p;

    free(pSum->pauli_strings);
    free(pSum);
    return new_pSum;
}

char *pauli_sum_to_string_c(PauliSumC *pSum) {
    size_t buffer_size = 1; // Space for null terminator
    char **pauli_strings; 
    char *buffer;
    
    pauli_strings = (char **) malloc(pSum->p * sizeof(char *));

    // First, collect all individual Pauli string representations
    for (unsigned int i = 0; i < pSum->p; i++) {
        pauli_strings[i] = pauli_string_to_string_c(pSum->pauli_strings[i]);
        buffer_size += strlen(pauli_strings[i]) + 3; // +3 for " + " separator
    }

    // Allocate buffer
    buffer = (char *) malloc(buffer_size);
    if (!buffer) return NULL; // Check for allocation failure

    // Initialize the buffer with the first Pauli string
    snprintf(buffer, buffer_size, "%s", pauli_strings[0]);

    // Append remaining Pauli strings
    for (unsigned int i = 1; i < pSum->p; i++) {
        strncat(buffer, " + ", buffer_size - strlen(buffer) - 1);
        strncat(buffer, pauli_strings[i], buffer_size - strlen(buffer) - 1);
    }

    // Free intermediate Pauli strings
    for (unsigned int i = 0; i < pSum->p; i++) {
        free(pauli_strings[i]);
    }
    free(pauli_strings);

    return buffer;
}

PauliSumC *pauli_sum_append_pauli_string_c(PauliSumC *pSum, PauliStringC *pString) {
    if (pSum->p == pSum->p_max) {
        pSum = pauli_sum_realloc_c(pSum, pSum->p_max + 1);
    }

    pSum->pauli_strings[pSum->p] = pString;
    pSum->p++;
    return pSum;
}

PauliSumC *pauli_sum_scalar_multiplication_c(PauliSumC *pSum, double complex scalar) {
    PauliSumC *new_pSum = pauli_sum_init_c(pSum->p_max);
    for (unsigned int i = 0; i < pSum->p; i++) {
        new_pSum = pauli_sum_append_pauli_string_c(new_pSum, pauli_string_scalar_multiplication_c(pSum->pauli_strings[i], scalar));
    }
    return new_pSum;
}

PauliSumC *pauli_sum_adjoint_c(PauliSumC *pSum) {
    PauliSumC *new_pSum = pauli_sum_init_c(pSum->p_max);
    for (unsigned int i = 0; i < pSum->p; i++) {
        new_pSum = pauli_sum_append_pauli_string_c(new_pSum, pauli_string_adjoint_c(pSum->pauli_strings[i]));
    }
    return new_pSum;
}

PauliSumC *pauli_sum_multiplication_c(PauliSumC *left, PauliSumC *right) {
    PauliSumC *new_pSum = pauli_sum_init_c(left->p_max);
    for (unsigned int i = 0; i < left->p; i++) {
        for (unsigned int j = 0; j < right->p; j++) {
            new_pSum = pauli_sum_append_pauli_string_c(new_pSum, pauli_string_multiplication_c(left->pauli_strings[i], right->pauli_strings[j]));
        }
    }
    return new_pSum;
}
