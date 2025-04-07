
from fermion import *

import numpy as np
import qiskit as q
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator
import qiskit_aer as Aer
from typing import Set


class InitOperators:
    ops: list  # List of Operators
    epsilons: list  # List of epsilon values
    N: int  # Total number of sites / qubits
    M: int  # Number of Operators

    def __init__(self, N):
        self.N = N
        self.M = 0
        self.ops = []
        self.epsilons = []

    def add_operator(self, operator, epsilon):
        self.ops.append(operator)
        self.epsilons.append(epsilon)
        self.M += 1

    def adjoint(self):
        adjoint_init_ops = InitOperators(self.N)
        for m in range(self.M):
            adjoint_init_ops.add_operator(
                self.ops[self.M-m-1].adjoint(), self.epsilons[self.M-m-1].conj())
        return adjoint_init_ops

    def create_initialization_circuit(self):
        N = self.N
        circ = q.QuantumCircuit(N, 0)
        for m in range(self.M):
            circ = circ.compose(
                qiskit_create_pauli_sum_evolution_circuit(self.ops[m].pSum, self.epsilons[m]))
        return circ


def qiskit_create_initialization_from_slater_determinant_circuit(sDet: SlaterDeterminant):
    N = sDet.N
    circ = q.QuantumCircuit(N, 0)
    orbitals = sDet.orbitals
    for i in range(N):
        if orbitals[i] == 1:
            circ.x(i)
    return circ


def qiskit_create_pauli_string_evolution_circuit(pString: PauliString, epsilon: np.complex128 = 1.0):
    N = pString.N
    circ = q.QuantumCircuit(N, 0)

    coef = pString.coef
    theta = -1 * (coef * epsilon).imag
    string = pString.string

    q_pString = SparsePauliOp(string[::-1])
    rotation = PauliEvolutionGate(q_pString, theta)
    circ.append(rotation, range(N))
    return circ.decompose()


def qiskit_create_pauli_sum_evolution_circuit(pSum: PauliSum, epsilon: np.complex128 = 1.0):

    pStrings = pSum.get_pauli_strings()

    circ = q.QuantumCircuit(pSum.N, 0)
    for pString in pStrings:
        circ = circ.compose(
            qiskit_create_pauli_string_evolution_circuit(pString, epsilon))
    return circ


def qiskit_pauli_string_measurement(circuit: q.QuantumCircuit, pString_as_string: str, backend, shots=2**13):
    estimator = Estimator()

    observable = SparsePauliOp(pString_as_string[::-1])

    result = estimator.run(circuit, observable,
                           backend=backend, shots=shots).result()
    return result.values[0]


def qiskit_perform_tomoraphy(circuit: q.QuantumCircuit, measurements: Set[str], backend, shots=2**13):
    tomography = {}
    for measurement in measurements:
        tomography[measurement] = qiskit_pauli_string_measurement(
            circuit, measurement, backend, shots)

    return tomography


def qiskit_probability_distribution(circuit: q.QuantumCircuit, backend, shots=2**13):
    circuit = circuit.copy()
    circuit.measure_all()
    num_qubits = circuit.num_qubits
    job = backend.run(circuit, shots=shots)
    result = job.result().data(0)
    counts = result.get("counts")

    total_counts = sum(counts.values())
    probabilities = {bin(int(state, 16))[2:].zfill(num_qubits)[::-1]: count /
                     total_counts for state, count in counts.items()}

    return probabilities


def slater_determinant_probability(sDet: SlaterDeterminant, prob_dist):
    orbitals = sDet.orbitals
    bit_string = string = ''.join([str(i) for i in orbitals])
    if bit_string not in prob_dist:
        return 0
    return prob_dist[bit_string]


def qiskit_statevector(circuit: q.QuantumCircuit):
    circuit = circuit.copy()
    backend = Aer.AerSimulator()
    circuit.save_statevector()
    result = backend.run(circuit).result()
    statevector = result.get_statevector()
    return statevector.data


def slater_determinant_probability_from_statevector(sDet: SlaterDeterminant, statevector):
    coef = statevector[sDet.encoding]
    return np.abs(coef)**2


def qiskit_pauli_string_measurement_statevector(circuit: q.QuantumCircuit, pString_as_string: str):
    circuit = circuit.copy()
    backend = Aer.AerSimulator()
    circuit.save_statevector()
    result = backend.run(circuit).result()
    statevector = result.get_statevector()

    observable = SparsePauliOp(pString_as_string[::-1])
    return statevector.expectation_value(observable)


def qiskit_perform_tomography_statevector(circuit: q.QuantumCircuit, measurements: Set[str]):
    tomography = {}
    for measurement in measurements:
        tomography[measurement] = qiskit_pauli_string_measurement_statevector(
            circuit, measurement)

    return tomography
