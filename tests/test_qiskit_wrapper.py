from qiskit_wrapper import *
import qiskit_aer as Aer


def test_qiskit_expectation():
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

    orbitals = [1, 0]
    sdet = SlaterDeterminant(2, 1 + 0j, orbitals)

    paulis0 = ["X", "X"]
    pString0 = PauliString(2, 1j * np.pi / 4, paulis0)
    paulis1 = ["I", "I"]
    pString1 = PauliString(2, 1j * 1, paulis1)
    pSum = pString0 + pString1

    circuit = q.QuantumCircuit(2, 2)
    circuit = circuit.compose(qiskit_create_initialization_circut(sdet))
    circuit = circuit.compose(qiskit_create_pauli_sum_evolution_circuit(pSum))
    backend = Aer.AerSimulator()

    measurements_r = op1.aggregate_measurements_recursive()
    print(f"fProd1: {fProd1.pSum}")
    print(f"fProd2: {fProd2.pSum}")
    print(f"fProd3: {fProd3.pSum}")
    print(f"fProd4: {fProd4.pSum}")

    tomography = qiskit_perform_tomoraphy(circuit, measurements_r, backend)
    print(
        f"Tomography data:\n {[f'{key}: {value}' for key, value in tomography.items()]}")

    circuit.save_statevector()
    result = backend.run(circuit).result()
    statevector = result.get_statevector()
    print(f"Statevector: {statevector.data}")

    print(f"{fProd3.evaluate_expectation(tomography)}")

    print(f"Op1 = {op1.evaluate_expectation(tomography)} = {fProd1.evaluate_expectation(tomography)} + {fProd2.evaluate_expectation(tomography)} + {fProd3.evaluate_expectation(tomography)} + {fProd4.evaluate_expectation(tomography)}")
    print(f"Op2 = {op2.evaluate_expectation(tomography)} = {fProd1.evaluate_expectation(tomography)} + {fProd2.evaluate_expectation(tomography)}")


def test_qiskit_probability_distribution():
    paulis0 = ["X", "X"]
    pString0 = PauliString(2, 1j * np.pi / 4, paulis0)
    paulis1 = ["I", "I"]
    pString1 = PauliString(2, 1j * 1, paulis1)
    pSum = pString0 + pString1

    sdet = SlaterDeterminant(2, 1 + 0j, [1, 0])

    circuit = q.QuantumCircuit(2, 0)
    circuit = circuit.compose(qiskit_create_initialization_circut(sdet))
    circuit = circuit.compose(qiskit_create_pauli_sum_evolution_circuit(pSum))
    backend = Aer.AerSimulator()

    circuit.save_statevector()
    result = backend.run(circuit).result()
    statevector = result.get_statevector()
    print(f"Statevector: {statevector.data}")

    prob_dist = qiskit_probability_distribution(circuit, backend, 2**13)
    print(f"Probability Distribution: {prob_dist}")
    print(f"P_|10> = {slater_determinant_probability(sdet, prob_dist)}")


def main():
    print("Testing Qiskit Expectation")
    test_qiskit_expectation()
    print("\nTest Qiskit Probability Distribution")
    test_qiskit_probability_distribution()


if __name__ == "__main__":
    main()
