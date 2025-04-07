from projector import *
from hamiltonian import *
import numpy as np


def test_projector_H2():
    np.set_printoptions(linewidth=np.inf, precision=4,
                        formatter={'float_kind': '{:e}'.format})

    sdet1010 = SlaterDeterminant(4, 1, [1, 0, 1, 0])
    sdet0101 = SlaterDeterminant(4, 1, [0, 1, 0, 1])
    sdet0 = SlaterDeterminant(1, 1, [0])
    sdet1 = SlaterDeterminant(1, 1, [1])

    four_to_one = Projector(4, 1)
    four_to_one = four_to_one + (sdet1010, sdet0) + (sdet0101, sdet1)

    for pair in four_to_one.mapping:
        (original_sdet_right, target_sdet_right) = pair
        print(f"{original_sdet_right} -> {target_sdet_right}")

    H2 = gto.M(
        atom="H 0 0 0; H 0 0 0.7348654",
        basis="sto-3g",
        charge=0,
        spin=0,
        unit="Angstrom",
    )

    H = init_Hamiltonian_from_pyscf(H2)

    print("Performing mapping")

    for prod in H.prods:
        print(prod)
        new_prods = prod.map(four_to_one)
        if len(new_prods) > 0:
            for new_prod in new_prods:
                print(new_prod)
        print("\n")

    H_1q = H.map(four_to_one)
    print("H_1q:")
    for prod in H_1q.prods:
        print(prod)

    if False:
        print("As matrices")
        print(H.pSum.matrix.real)
        print(H_1q.pSum.matrix.real)

    if False:
        print("As Pauli sums")
        print(H.pSum)
        print(H_1q.pSum)


def test_projector_H2_time_evolution():
    sdet1010 = SlaterDeterminant(4, 1, [1, 0, 1, 0])
    sdet0101 = SlaterDeterminant(4, 1, [0, 1, 0, 1])
    sdet0 = SlaterDeterminant(1, 1, [0])
    sdet1 = SlaterDeterminant(1, 1, [1])

    four_to_one = Projector(4, 1)
    four_to_one = four_to_one + (sdet1010, sdet0) + (sdet0101, sdet1)

    H2 = gto.M(
        atom="H 0 0 0; H 0 0 0.7348654",
        basis="sto-3g",
        charge=0,
        spin=0,
        unit="Angstrom",
    )

    H = init_Hamiltonian_from_pyscf(H2)

    H_1q = H.map(four_to_one)

    # Time evolution
    time_step = 0.0001 * 1j
    num_steps = 10000

    # Initial states
    psi_4q = Wavefunction(4) + sdet1010
    psi_1q = Wavefunction(1) + sdet0

    # Time evolution loop
    for step in range(num_steps):
        # Evolve 4-qubit state
        if False:
            for prod in H.prods:
                psi_4q = wavefunction_pauli_sum_evolution(
                    prod.pSum, psi_4q, time_step)
                print(f"{prod}: {psi_4q}")

            for prod in H_1q.prods:
                psi_1q = wavefunction_pauli_sum_evolution(
                    prod.pSum, psi_1q, time_step)
                print(f"{prod}: {psi_1q}")

        if True:
            # Project to 1-qubit space
            psi_4q = wavefunction_pauli_sum_evolution(
                H.pSum, psi_4q, time_step)
            psi_1q = wavefunction_pauli_sum_evolution(
                H_1q.pSum, psi_1q, time_step)
            if step % 1000 == 0:
                print(f"{psi_4q} ~ {psi_1q}")

    print(f"{psi_4q.norm()} ~ {psi_1q.norm()}")


def test_projector_expectation():
    sdet1010 = SlaterDeterminant(4, 1, [1, 0, 1, 0])
    sdet0101 = SlaterDeterminant(4, 1, [0, 1, 0, 1])
    sdet0 = SlaterDeterminant(1, 1, [0])
    sdet1 = SlaterDeterminant(1, 1, [1])

    four_to_one = Projector(4, 1)
    four_to_one = four_to_one + (sdet1010, sdet0) + (sdet0101, sdet1)

    H2 = gto.M(
        atom="H 0 0 0; H 0 0 0.7348654",
        basis="sto-3g",
        charge=0,
        spin=0,
        unit="Angstrom",
    )

    H = init_Hamiltonian_from_pyscf(H2)

    H_1q = H.map(four_to_one)

    # Initial states
    psi_4q = Wavefunction(4) + (0.9983) * sdet1010 + (-0.1115) * sdet0101
    psi_1q = Wavefunction(1) + (0.9983) * sdet0 + (-0.1115) * sdet1
    psi_4q = (1 / psi_4q.norm()) * psi_4q
    psi_1q = (1 / psi_1q.norm()) * psi_1q
    print(f"psi_4q norm: {psi_4q.norm()}")
    print(f"psi_1q norm: {psi_1q.norm()}")

    tomography_4q = wavefunction_perform_tomography(
        psi_4q, H.aggregate_measurements())
    tomography_1q = wavefunction_perform_tomography(
        psi_1q, H_1q.aggregate_measurements())

    exp_4q = H.energy(tomography_4q)
    exp_1q = H_1q.energy(tomography_1q)

    print(f"Expectation value for the original Hamiltonian: {exp_4q}")
    print(f"Expectation value for the projected Hamiltonian: {exp_1q}")


def main():
    print("Testing Projector class")
    test_projector_H2()

    print("\nTesting Projector class with time evolution")
    test_projector_H2_time_evolution()

    print("\nTesting Projector class with expectation values")
    test_projector_expectation()


if __name__ == "__main__":
    main()
