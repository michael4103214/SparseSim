
from fermion import *

import qiskit as q
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator
from typing import Set


def qiskit_create_initialization_circut(sDet: SlaterDeterminant):
    N = sDet.N
    circ = q.QuantumCircuit(N, 0)
    orbitals = sDet.orbitals
    for i in range(N):
        if orbitals[i] == 1:
            circ.x(i)
    return circ


def qiskit_create_pauli_string_evolution_circuit(pString: PauliString):
    N = pString.N
    circ = q.QuantumCircuit(N, 0)

    coef = pString.coef
    theta = -1 * coef.imag
    string = pString.string

    q_pString = SparsePauliOp(string[::-1])
    rotation = PauliEvolutionGate(q_pString, theta)
    circ.append(rotation, range(N))
    return circ.decompose()


def qiskit_create_pauli_sum_evolution_circuit(pSum: PauliSum):

    pStrings = pSum.get_pauli_strings()

    circ = q.QuantumCircuit(pSum.N, 0)
    for pString in pStrings:
        circ = circ.compose(
            qiskit_create_pauli_string_evolution_circuit(pString))
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
