from math import log2
import random
import time
import numpy as np

from sparse_sim.cython.pauli import *
from sparse_sim.cython.wavefunction import *
from sparse_sim.cython.density_matrix import *
# from sparse_sim.fermion.rdm import *


def test_initialization_scaling_freeing():
    oprod0 = OuterProduct(4, 1 + 0j, [0, 1, 0, 1], [0, 1, 0, 1])

    oprod1 = OuterProduct(4, 1 + 0j, [1, 0, 1, 0], [1, 0, 1, 0])

    dm = DensityMatrix(4)

    dm = dm + oprod0
    dm = dm + oprod1

    print(
        f"DensityMatrix initialized with {dm.o} outer products: {oprod0}, {oprod1}\n")

    print(dm)

    trace = dm.trace()
    print(f"Trace: {trace}")

    normalized_dm = (1 / trace) * dm
    print(f"Normalized: {normalized_dm}")


def test_density_matrix_from_wavefunction():
    sdet0 = SlaterDeterminant(4, 1 / np.sqrt(2), [0, 1, 0, 1])

    sdet1 = SlaterDeterminant(4, 1j / np.sqrt(2), [1, 0, 1, 0])

    wfn = Wavefunction(4)

    wfn.append_slater_determinant(sdet0)
    wfn.append_slater_determinant(sdet1)

    bra = wfn.adjoint()

    print(f"({wfn}) ({bra}) = ")

    dm = density_matrix_from_wavefunction(wfn)
    print(f"{dm}\n")


def test_density_matrix_multiplication():

    sdet0 = SlaterDeterminant(4, 1 / np.sqrt(2), [0, 0, 0, 0])

    sdet1 = SlaterDeterminant(4, 1j / np.sqrt(2), [1, 1, 1, 1])

    wfn1 = sdet0 + sdet1

    dm1 = density_matrix_from_wavefunction(wfn1)

    dm2 = DensityMatrix(4)

    for k in range(2):
        for l in range(2):
            ket_orbitals = [0, 0, k, l]
            bra_orbitals = [0, 0, l, k]
            oprod = OuterProduct(4, 1 + 0j, ket_orbitals, bra_orbitals)
            dm2.append_outer_product(oprod)

    print(f"dm1 with trace {dm1.trace()}: {dm1}")
    print(f"dm2 with trace {dm2.trace()}: {dm2}\n")

    print(f"dm1 * dm1 = {dm1 * dm1}")
    print(f"with trace {(dm1 * dm1).trace()}\n")

    print(f"dm2 * dm2 = {dm2 * dm2}")
    print(f"with trace {(dm2 * dm2).trace()}\n")

    print(f"dm1 * dm2 = {dm1 * dm2}")
    print(f"with trace {(dm1 * dm2).trace()}\n")

    print(f"dm2 * dm1 = {dm2 * dm1}")
    print(f"with trace {(dm2 * dm1).trace()}\n")


def test_appending_outer_products():

    print("Order of Adding:")

    dm1 = DensityMatrix(4)

    for i in range(2):
        for j in range(2):
            for k in range(2):
                for l in range(2):
                    ket_orbitals = [i, j, k, l]
                    bra_orbitals = [i, j, k, l]
                    oprod = OuterProduct(4, 1 + 0j, ket_orbitals, bra_orbitals)
                    dm1.append_outer_product(oprod)
                    # print(f"Added OuterProduct: {oprod}")

    print(
        f"\nDensityMatrix after adding outer products in order:\n{dm1}\n with trace {dm1.trace()}\n")

    print("Reverse Order of Adding:")

    dm2 = DensityMatrix(4)

    for i in reversed(range(2)):
        for j in reversed(range(2)):
            for k in reversed(range(2)):
                for l in reversed(range(2)):
                    ket_orbitals = [l, k, j, i]
                    bra_orbitals = [l, k, j, i]
                    oprod = OuterProduct(4, 1 + 0j, ket_orbitals, bra_orbitals)
                    dm2.append_outer_product(oprod)
                    # print(f"Added OuterProduct: {oprod}")

    print(
        f"\nDensityMatrix after adding outer products in reverse order:\n{dm2}\n with trace {dm2.trace()}\n")


def test_density_matrix_pauli_sum_multiplication():

    oprod0 = OuterProduct(4, 1 + 0j, [0, 1, 0, 1], [0, 1, 0, 1])
    oprod1 = OuterProduct(4, 1 + 0j, [1, 0, 1, 0], [1, 0, 1, 0])

    dm = DensityMatrix(4)
    dm.append_outer_product(oprod0)
    dm.append_outer_product(oprod1)

    pString0 = PauliString(4, 1, ["I", "Y", "I", "X"])
    pString1 = PauliString(4, 0.5, ["X", "I", "X", "I"])

    pSum = PauliSum(4)
    pSum.append_pauli_string(pString0)
    pSum.append_pauli_string(pString1)

    new_dm_left = pSum * dm
    new_dm_right = dm * pSum

    print(f"{pSum} * {dm}\n = {new_dm_left}\n\n")
    print(f"{dm} * {pSum}\n = {new_dm_right}\n")


def test_density_matrix_pauli_string_evolution():
    oprod0 = OuterProduct(4, 1 + 0j, [0, 1, 0, 1], [0, 1, 0, 1])

    dm = DensityMatrix(4)
    dm.append_outer_product(oprod0)

    pString = PauliString(4, 1j, ["X", "X", "X", "X"])

    old_dm = dm

    for _ in range(100000):
        dm = density_matrix_pauli_string_evolution(pString, dm, 0.01)

    print(f"exp(1000 * {pString}) [{old_dm}]\n = {dm}")
    print(f"Trace: {dm.trace()}\n")


def test_density_matrix_pauli_sum_evolution():
    oprod = OuterProduct(4, 1 + 0j, [0, 1, 0, 1], [0, 1, 0, 1])

    dm = DensityMatrix(4)
    dm.append_outer_product(oprod)

    pString0 = PauliString(4, 0.5j, ["X", "X", "X", "X"])
    pString1 = PauliString(4, 0.5j, ["X", "X", "X", "X"])

    pSum = PauliSum(4)
    pSum.append_pauli_string(pString0)
    pSum.append_pauli_string(pString1)

    old_dm = dm

    for _ in range(100000):
        dm = density_matrix_pauli_sum_evolution(pSum, dm, 0.01)

    trace = dm.trace()

    print(f"exp(1000 * {pSum}) [{old_dm}]\n = {dm}")
    print(f"Trace: {dm.trace()}\n")


def test_density_matrix_CPTP_evolution():
    oprod = OuterProduct(2, 1 + 0j, [0, 1], [0, 1])

    dm = DensityMatrix(2)
    dm.append_outer_product(oprod)

    H = PauliSum(2)
    H.append_pauli_string(PauliString(2, 1, ["Z", "Z"]))
    H.append_pauli_string(PauliString(2, 0.5, ["X", "X"]))

    Ls = []
    L1 = PauliSum(2)
    L1.append_pauli_string(PauliString(2, 0.5, ["I", "X"]))
    Ls.append(L1)

    L2 = PauliSum(2)
    L2.append_pauli_string(PauliString(2, 0.5, ["X", "I"]))
    Ls.append(L2)

    old_dm = dm

    for _ in range(100000):
        dm = density_matrix_CPTP_evolution(H, Ls, dm, 0.01)

    print(f"CPTP Evolution with \nH={H} \nLs={Ls} \non [{old_dm}]\n= {dm}")
    print(f"Trace: {dm.trace()}\n")


def test_density_matrix_speed():
    N = 10
    num_oprods = 2
    num_operations = 20

    dm = DensityMatrix(N)

    for i in range(num_oprods):
        orbitals = []
        for j in range(N):
            orbitals.append(random.randint(0, 1))

        coef = np.random.rand()

        oprod = OuterProduct(N, coef, orbitals, orbitals)
        dm.append_outer_product(oprod)

    trace = dm.trace()
    dm = (1 / trace) * dm

    print("DensityMatrix initialized.")

    pauli_sums = []
    for i in range(num_operations):
        pSum = PauliSum(N)
        for j in range(3):
            paulis = []
            for k in range(N):
                pauli = random.choice(["I", "X", "Y", "Z"])
                paulis.append(pauli)
            coef = 1j * np.random.rand()
            pString = PauliString(N, coef, paulis)
            pSum.append_pauli_string(pString)
        pauli_sums.append(pSum)

    print("PauliSums generated.")

    start_time = time.time()
    for operation in range(num_operations):
        pSum = pauli_sums[operation]
        print(f"{operation}: {pSum}")
        dm = density_matrix_pauli_sum_evolution(pSum, dm, 0.01)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(
        f"Completed {num_operations}  Pauli sum evolutions in {elapsed_time:.4f} seconds.")
    print(f"Final density matrix trace: {dm.trace()}\n")
    print(
        f"Final number of outer products is {dm.o} showing a 2^{log2(dm.o):.2f} increase.\n")


def test_density_matrix_cleaning():
    oprod0 = OuterProduct(4, 1e-10 + 0j, [0, 1, 0, 1], [0, 1, 0, 1])
    oprod1 = OuterProduct(4, 1 + 0j, [1, 0, 1, 0], [1, 0, 1, 0])

    dm = oprod0 + oprod1

    print(f"Before cleaning: {dm} with trace {dm.trace()}\n")

    dm_clean = dm.remove_near_zero_terms(1e-9)

    print(f"After cleaning: {dm_clean} with trace {dm_clean.trace()}\n")


'''def test_density_matrix_tomography_and_probability_distribution():

    sdet0 = SlaterDeterminant(4, 1/np.sqrt(2), [0, 1, 0, 1])
    sdet1 = SlaterDeterminant(4, 1/np.sqrt(2), [1, 0, 1, 0])

    wfn = sdet0 + sdet1

    pString0 = PauliString(4, 0.5, ["I", "X", "I", "X"])
    pString1 = PauliString(4, 0.5, ["X", "I", "X", "I"])
    pString2 = PauliString(4, 0.25, ["Y", "Y", "I", "I"])
    pString3 = PauliString(4, 0.25, ["I", "I", "Y", "Y"])
    pString4 = PauliString(4, 1, ["Z", "Z", "Z", "Z"])
    pSum = PauliSum(4)
    pSum.append_pauli_string(pString0)
    pSum.append_pauli_string(pString1)
    pSum.append_pauli_string(pString2)
    pSum.append_pauli_string(pString3)
    pSum.append_pauli_string(pString4)

    for _ in range(100):
        wfn = wavefunction_pauli_sum_evolution(pSum, wfn, 0.01)

    print(f"|wfn> = {wfn}")

    dm = density_matrix_from_wavefunction(wfn)
    print(f"dm = |wfn><wfn|")

    D2 = RDM(2, 4)

    measurements = D2.aggregate_measurements_recursive()

    wfn_tomography = wavefunction_perform_tomography(wfn, measurements)
    dm_tomography = density_matrix_perform_tomography(dm, measurements)

    same_tomo = True
    for fProd in D2.prods:
        wfn_exp = pauli_sum_evaluate_expectation(fProd.pSum, wfn_tomography)
        dm_exp = pauli_sum_evaluate_expectation(fProd.pSum, dm_tomography)
        if not np.isclose(wfn_exp, dm_exp):
            same_tomo = False

    if same_tomo:
        print("Tomography results of 2rdm from Wavefunction and DensityMatrix match.\n")
    else:
        print(
            "Tomography results of 2rdm from Wavefunction and DensityMatrix DO NOT match.\n")

    wfn_prob_dist = wavefunction_to_probability_distribution(wfn)
    dm_prob_dist = density_matrix_to_probability_distribution(dm)

    same_prob_dist = True
    for state, wfn_prob in wfn_prob_dist.items():
        dm_prob = dm_prob_dist.get(state, 0.0)
        if not np.isclose(wfn_prob, dm_prob):
            same_prob_dist = False

    if same_prob_dist:
        print("Probability distributions from Wavefunction and DensityMatrix match.\n")
    else:
        print(
            "Probability distributions from Wavefunction and DensityMatrix DO NOT match.\n")'''


def main():
    print("\nTesting DensityMatrix Initialization, Scaling, and Freeing:")
    test_initialization_scaling_freeing()

    print("\nTesting DensityMatrix from Wavefunction:")
    test_density_matrix_from_wavefunction()

    print("\nTesting DensityMatrix Multiplication:")
    test_density_matrix_multiplication()

    print("\nTesting Appending OuterProducts to DensityMatrix:")
    test_appending_outer_products()

    print("\nTesting DensityMatrix and PauliSum Multiplication:")
    test_density_matrix_pauli_sum_multiplication()

    print("\nTesting DensityMatrix PauliString Evolution:")
    test_density_matrix_pauli_string_evolution()

    print("\nTesting DensityMatrix PauliSum Evolution:")
    test_density_matrix_pauli_sum_evolution()

    print("\nTesting DensityMatrix CPTP Evolution:")
    test_density_matrix_CPTP_evolution()

    print("\nTesting DensityMatrix Speed with Multiple PauliSum Evolutions:")
    # test_density_matrix_speed()

    print("\nTesting DensityMatrix Cleaning:")
    test_density_matrix_cleaning()

    # print("\nTesting DensityMatrix Tomography and Probability Distribution:")
    # test_density_matrix_tomography_and_probability_distribution()


if __name__ == "__main__":
    main()
