from qiskit_wrapper import *


from mthree import M3Mitigation
import numpy as np
import qiskit as q
import qiskit_aer as Aer
from qiskit_aer.noise import NoiseModel
from qiskit_ibm_runtime import QiskitRuntimeService


def test_qiskit_expectation():
    fOp1 = FermionicOperator("+", 0, 2)
    fOp2 = FermionicOperator("-", 0, 2)
    fOp3 = FermionicOperator("+", 1, 2)
    fOp4 = FermionicOperator("-", 1, 2)

    fProd1 = Product(1, [fOp1, fOp2], 2)
    fProd2 = Product(1, [fOp3, fOp4], 2)
    fProd3 = Product(1, [fOp3, fOp2], 2)
    fProd4 = Product(1, [fOp1, fOp4], 2)

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
    circuit = circuit.compose(
        qiskit_create_initialization_from_slater_determinant_circuit(sdet))
    circuit = circuit.compose(qiskit_create_pauli_sum_evolution_circuit(pSum))
    backend = Aer.AerSimulator()

    measurements_r = op1.aggregate_measurements_recursive()
    print(f"fProd1: {fProd1.pSum}")
    print(f"fProd2: {fProd2.pSum}")
    print(f"fProd3: {fProd3.pSum}")
    print(f"fProd4: {fProd4.pSum}")

    tomography = qiskit_perform_tomography(circuit, measurements_r, backend)
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
    pString0 = PauliString(2, 1j * np.pi / 8, paulis0)
    paulis1 = ["I", "I"]
    pString1 = PauliString(2, 1j * 1, paulis1)
    pSum = pString0 + pString1

    sdet = SlaterDeterminant(2, 1 + 0j, [1, 0])

    circuit = q.QuantumCircuit(2, 0)
    circuit = circuit.compose(
        qiskit_create_initialization_from_slater_determinant_circuit(sdet))
    circuit = circuit.compose(qiskit_create_pauli_sum_evolution_circuit(pSum))
    backend = Aer.AerSimulator()

    statevector = qiskit_statevector(circuit)
    print(f"Statevector: {statevector}")
    print(
        f"P_|10> = {slater_determinant_probability_from_statevector(sdet, statevector)}")

    prob_dist = qiskit_probability_distribution(circuit, backend, 2**13)
    print(f"Probability Distribution: {prob_dist}")
    print(f"P_|10> = {slater_determinant_probability(sdet, prob_dist)}")


def test_qiskit_statevector_expectation():
    fOp1 = FermionicOperator("+", 0, 2)
    fOp2 = FermionicOperator("-", 0, 2)
    fOp3 = FermionicOperator("+", 1, 2)
    fOp4 = FermionicOperator("-", 1, 2)

    fProd1 = Product(1, [fOp1, fOp2], 2)
    fProd2 = Product(1, [fOp3, fOp4], 2)
    fProd3 = Product(1, [fOp3, fOp2], 2)
    fProd4 = Product(1, [fOp1, fOp4], 2)

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
    circuit = circuit.compose(
        qiskit_create_initialization_from_slater_determinant_circuit(sdet))
    circuit = circuit.compose(qiskit_create_pauli_sum_evolution_circuit(pSum))

    measurements_r = op1.aggregate_measurements_recursive()
    print(f"fProd1: {fProd1.pSum}")
    print(f"fProd2: {fProd2.pSum}")
    print(f"fProd3: {fProd3.pSum}")
    print(f"fProd4: {fProd4.pSum}")

    tomography = qiskit_perform_tomography_statevector(circuit, measurements_r)
    print(
        f"Tomography data:\n {[f'{key}: {value}' for key, value in tomography.items()]}")

    print(f"Op1 = {op1.evaluate_expectation(tomography)} = {fProd1.evaluate_expectation(tomography)} + {fProd2.evaluate_expectation(tomography)} + {fProd3.evaluate_expectation(tomography)} + {fProd4.evaluate_expectation(tomography)}")
    print(f"Op2 = {op2.evaluate_expectation(tomography)} = {fProd1.evaluate_expectation(tomography)} + {fProd2.evaluate_expectation(tomography)}")


def test_qiskit_noisy_probability_distribution():
    paulis0 = ["X"]
    pString0 = PauliString(1, 1j * np.pi / 8, paulis0)
    paulis1 = ["X"]
    pString1 = PauliString(1, -1j * np.pi / 8, paulis1)
    pSum = pString0 + pString1

    sdet = SlaterDeterminant(1, 1 + 0j, [0])

    circuit = q.QuantumCircuit(1, 0)
    circuit = circuit.compose(
        qiskit_create_initialization_from_slater_determinant_circuit(sdet))
    for i in range(3):
        circuit = circuit.compose(
            qiskit_create_pauli_sum_evolution_circuit(pSum))
        circuit.barrier()

    service = QiskitRuntimeService(channel="ibm_quantum")
    full_backend = service.backend("ibm_marrakesh")
    full_noise_model = NoiseModel.from_backend(full_backend)
    trimmed_noise_model = trim_noise_model(full_noise_model, {0: 0})
    gates = ['id', 'sx', 'x', 'cz', 'rz']

    backend = Aer.AerSimulator(
        method='density_matrix', noise_model=trimmed_noise_model, basis_gates=gates, n_qubits=1)
    mit = M3Mitigation(backend)
    cals_from_noise_model(mit, trimmed_noise_model)

    statevector = qiskit_statevector(circuit)
    print(f"Statevector: {statevector}")
    print(
        f"P_|10> = {slater_determinant_probability_from_statevector(sdet, statevector)}")
    prob_dist = qiskit_probability_distribution(circuit, backend, 2**15)
    print(f"Probability Distribution without ZNE: {prob_dist}")
    print(f"P_|10> = {slater_determinant_probability(sdet, prob_dist)}")
    prob_dist_zne = qiskit_probability_distribution_with_zne(
        circuit, backend, 2**15, [1, 3, 5], mit)
    print(f"Probability Distribution with ZNE: {prob_dist_zne}")
    print(f"P_|10> = {slater_determinant_probability(sdet, prob_dist_zne)}")


def main():
    print("Testing Qiskit Expectation")
    test_qiskit_expectation()
    print("\nTest Qiskit Probability Distribution")
    test_qiskit_probability_distribution()
    print("\nTesting Qiskit Statevector Expectation")
    test_qiskit_statevector_expectation()
    print("\nTesting Qiskit Noisy Probability Distribution")
    test_qiskit_noisy_probability_distribution()


if __name__ == "__main__":
    main()
