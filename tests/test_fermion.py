from fermion import *


def test_fermionic_operator_initialization_and_freeing():

    fOp1 = FermionicOperator("+", 1, 2)

    fOp2 = FermionicOperator("-", 0, 2)

    print(f"{fOp1} = {fOp1.pSum}")
    print(f"{fOp2} = {fOp2.pSum}")


def test_fermionic_operator_adjoint():
    fOp = FermionicOperator("+", 1, 2)
    adjoint_fOp = fOp.adjoint()

    print(f"({fOp})^\dagger = {adjoint_fOp}")
    print(f"({fOp.pSum})^\dagger = {adjoint_fOp.pSum}")


def test_fermionic_product_initialization_and_freeing():
    fOP1 = FermionicOperator("+", 1, 2)
    fOP2 = FermionicOperator("-", 0, 2)
    fOP3 = FermionicOperator("+", 0, 2)
    fOP4 = FermionicOperator("-", 1, 2)

    fProd1 = FermionicProduct(1, [fOP1, fOP2], 2)
    fProd2 = FermionicProduct(1, [fOP3, fOP4], 2)

    print(f"{fProd1} = {fProd1.pSum}")
    print(f"{fProd2} = {fProd2.pSum}")

    print(f"{fProd2.pSum} = {fProd1.adjoint().pSum}")


def test_operator_expectation():
    fOp1 = FermionicOperator("+", 0, 2)
    fOp2 = FermionicOperator("-", 0, 2)
    fOp3 = FermionicOperator("+", 1, 2)
    fOp4 = FermionicOperator("-", 1, 2)

    fProd1 = FermionicProduct(1, [fOp1, fOp2], 2)
    fProd2 = FermionicProduct(1, [fOp3, fOp4], 2)
    fProd3 = FermionicProduct(1, [fOp3, fOp2], 2)
    fProd4 = FermionicProduct(1, [fOp1, fOp4], 2)

    op1 = Operator([fProd1, fProd2, fProd3, fProd4], 2)
    op2 = Operator([fProd1, fProd2], 2)

    orbitals0 = [1, 0]
    orbitals1 = [0, 1]
    sdet0 = SlaterDeterminant(2, 1 + 0j, orbitals0)
    sdet1 = SlaterDeterminant(2, 0 + 1j, orbitals1)

    wfn = Wavefunction()

    wfn.append_slater_determinant(sdet0)
    wfn.append_slater_determinant(sdet1)

    measurements = op1.aggregate_measurements() | op2.aggregate_measurements()
    print(f"Wavefunction: {wfn}")
    print(f"fProd1: {fProd1.pSum}")
    print(f"fProd2: {fProd2.pSum}")
    print(f"fProd3: {fProd3.pSum}")
    print(f"fProd4: {fProd4.pSum}")
    print(f"Measurements: {measurements}")

    tomography = measurements_calculate_tomography(measurements, wfn)

    print(f"Tomography data:\n {tomography}")
    print(
        f"Expectation value: {op1.evaluate_expectation(tomography)} = {fProd1.evaluate_expectation(tomography)} + {fProd2.evaluate_expectation(tomography)} + {fProd3.evaluate_expectation(tomography)} + {fProd4.evaluate_expectation(tomography)}")
    print(
        f"Expectation value: {op2.evaluate_expectation(tomography)} = {fProd1.evaluate_expectation(tomography)} + {fProd2.evaluate_expectation(tomography)}")


def main():
    print("Testing Fermionic Operator Initialization and Freeing")
    test_fermionic_operator_initialization_and_freeing()
    print("\nTesting Fermionic Operator Adjoint")
    test_fermionic_operator_adjoint()
    print("\nTesting Fermionic Product Initialization and Freeing")
    test_fermionic_product_initialization_and_freeing()
    print("\nTesting Operator Expectation")
    test_operator_expectation()


if __name__ == "__main__":
    main()
