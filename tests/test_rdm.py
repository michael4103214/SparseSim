from rdm import *


def test_rdm_init_and_freeing():

    N = 2
    p = 2

    trdm = RDM(p, N)
    for prod in trdm.prods:
        print(prod)
    print(f"{len(trdm.prods)} elements in {trdm}")


def test_rdm_measurement():

    ordm = RDM(1, 2)
    for prod in ordm.prods:
        print(prod)
    print(f"{len(ordm.prods)} elements in {ordm}")

    orbitals0 = [1, 0]
    orbitals1 = [0, 1]
    sdet0 = SlaterDeterminant(2, 1 + 0j, orbitals0)
    sdet1 = SlaterDeterminant(2, 0 + 1j, orbitals1)

    wfn = Wavefunction(2)

    wfn.append_slater_determinant(sdet0)
    wfn.append_slater_determinant(sdet1)

    measurements = ordm.aggregate_measurements_recursive()
    print(f"Wavefunction: {wfn}")
    print(f"Measurements: {measurements}")

    tomography = wavefunction_perform_tomography(wfn, measurements)

    print(f"Tomography data:\n {tomography}")
    print(
        f"Expectation value: {ordm.evaluate_expectation(tomography)} = {ordm.prods[0].evaluate_expectation(tomography)} + {ordm.prods[1].evaluate_expectation(tomography)} + {ordm.prods[2].evaluate_expectation(tomography)} + {ordm.prods[3].evaluate_expectation(tomography)}")


def main():
    print("Testing rdm init and freeing")
    test_rdm_init_and_freeing()
    print("\nTesting RDM measurement")
    test_rdm_measurement()


if __name__ == "__main__":
    main()
