from sparse_sim import *


def test_pauli_string_initialization_scaling_freeing():
    paulis0 = ['I', 'X', 'I', 'X']
    pString0 = PauliString(4, 1 + 0j, paulis0)

    paulis1 = ['I', 'X', 'I', 'X']
    pString1 = PauliString(4, 1+1j, paulis1)

    print(pString0)

    pString2 = pauli_string_scalar_multiplication(pString1, 2 + 0j)
    print(f"2 * {pString1} = {pString2}")

    pString3 = pString2.adjoint()
    print(f"Adjoint of {pString2} = {pString3}")


def test_multiplication():
    paulis0 = ["I", "Y", "I", "X"]
    pString0 = PauliString(4, 1 + 0j, paulis0)

    paulis1 = ["I", "X", "I", "I"]
    pString1 = PauliString(4, 1j, paulis1)

    pString2 = pauli_string_multiplication(pString0, pString1)

    print(f"{pString0} * {pString1} = {pString2}")


def test_pauli_sum_initialization_scaling_freeing():
    paulis0 = ['I', 'Y', 'I', 'X']
    pString0 = PauliString(4, 1 + 0j, paulis0)

    paulis1 = ['I', 'X', 'I', 'I']
    pString1 = PauliString(4, 1j, paulis1)

    pSum0 = PauliSum()
    pSum0.append_pauli_string(pString0)
    pSum0.append_pauli_string(pString1)

    print(f"{pSum0} contains {pSum0.p()} pStrings")
    pSum1 = pauli_sum_scalar_multiplication(pSum0, 2 + 0j)
    print(f"2 * {pSum0} = {pSum1}")


def test_pauli_sum_multiplication():
    paulis0 = ['I', 'Y', 'I', 'X']
    pString0 = PauliString(4, 1 + 0j, paulis0)

    paulis1 = ['I', 'X', 'I', 'I']
    pString1 = PauliString(4, 1j, paulis1)

    pSum0 = PauliSum()
    pSum0.append_pauli_string(pString0)
    pSum0.append_pauli_string(pString1)

    paulis2 = ['I', 'Y', 'I', 'X']
    pString2 = PauliString(4, 1 + 0j, paulis2)

    paulis3 = ['I', 'X', 'I', 'I']
    pString3 = PauliString(4, 1j, paulis3)

    pSum1 = PauliSum()
    pSum1.append_pauli_string(pString2)
    pSum1.append_pauli_string(pString3)

    pSum2 = pauli_sum_multiplication(pSum0, pSum1)

    print(f"({pSum0}) * ({pSum1}) = {pSum2}")


def main():
    print("\nTesting PauliString initialization, scaling, and freeing")
    test_pauli_string_initialization_scaling_freeing()

    print("\nTesting PauliString Multiplication")
    test_multiplication()

    print("\nTesting PauliSum initialization, scaling, and freeing")
    test_pauli_sum_initialization_scaling_freeing()

    print("\nTesting PauliSum Multiplication")
    test_pauli_sum_multiplication()


if __name__ == "__main__":
    main()
