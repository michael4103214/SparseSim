from fermion import *

import copy
from mthree import M3Mitigation
import numpy as np
import qiskit as q
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import SparsePauliOp
from qiskit_ibm_runtime import Estimator
import qiskit_aer as Aer
from scipy.optimize import curve_fit
from typing import Set
import warnings
from scipy.optimize import OptimizeWarning

warnings.filterwarnings("ignore", category=OptimizeWarning)


class InitOperators:
    ops: list  # List of Operators
    epsilons: list  # List of epsilon values
    N: int  # Total number of sites / qubits
    M: int  # Number of Operators
    direction: str  # Direction of the circuit initialization

    def __init__(self, N):
        self.N = N
        self.M = 0
        self.ops = []
        self.epsilons = []
        self.pSum_direction = 'forward'

    def add_operator(self, operator, epsilon):
        self.ops.append(operator)
        self.epsilons.append(epsilon)
        self.M += 1

    def adjoint(self):
        adjoint_init_ops = InitOperators(self.N)
        for m in range(self.M):
            adjoint_init_ops.add_operator(
                self.ops[self.M-m-1].adjoint(), self.epsilons[self.M-m-1].conj())
        if self.pSum_direction == 'forward':
            adjoint_init_ops.pSum_direction = 'backwards'
        elif self.pSum_direction == 'backward':
            adjoint_init_ops.pSum_direction = 'forwards'
        else:
            raise ValueError("Invalid initialization direction")
        return adjoint_init_ops

    def create_initialization_circuit(self):
        N = self.N
        circ = q.QuantumCircuit(N, 0)
        if self.pSum_direction == 'forwards':
            for m in range(self.M):
                circ = circ.compose(
                    qiskit_create_pauli_sum_evolution_circuit(self.ops[m].pSum, self.epsilons[m]))
        elif self.pSum_direction == 'backwards':
            for m in range(self.M):
                circ = circ.compose(
                    qiskit_create_backwards_pauli_sum_evolution_circuit(self.ops[m].pSum, self.epsilons[m]))

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


def qiskit_create_backwards_pauli_sum_evolution_circuit(pSum: PauliSum, epsilon: np.complex128 = 1.0):

    pStrings = pSum.get_pauli_strings()[::-1]

    circ = q.QuantumCircuit(pSum.N, 0)
    for pString in pStrings:
        circ = circ.compose(
            qiskit_create_pauli_string_evolution_circuit(pString, epsilon))
    return circ


def qiskit_circuit_fold(circuit: q.QuantumCircuit, fold_factor: int):
    assert fold_factor % 2 == 1, "Fold factor must be an odd integer."

    num_folds = (fold_factor - 1) // 2

    folded_circuit = circuit.copy()
    for _ in range(num_folds):
        folded_circuit = folded_circuit.compose(circuit.inverse())
        folded_circuit = folded_circuit.compose(circuit)

    return folded_circuit


def qiskit_pauli_string_measurement(circuit: q.QuantumCircuit, pString_as_string: str, backend, shots=2**13):
    estimator = Estimator(mode=backend, options={"default_shots": shots})

    observable = SparsePauliOp(pString_as_string[::-1])

    result = estimator.run([(circuit, observable)]).result()
    return result[0].data.evs


def qiskit_pauli_string_measurement_with_zne(circuit: q.QuantumCircuit, pString_as_string: str, backend, shots=2**13, fold_factors: list = [1, 3, 5]):
    results = []
    for fold_factor in fold_factors:
        folded_circuit = qiskit_circuit_fold(circuit, fold_factor)
        result = qiskit_pauli_string_measurement(
            folded_circuit, pString_as_string, backend, shots)

        results.append(result)

    def linear(x, a, b):
        return a * x + b

    params, _ = curve_fit(linear, fold_factor, results)
    zne_estimate = linear(0, *params)

    return zne_estimate


def qiskit_perform_tomography(circuit: q.QuantumCircuit, measurements: Set[str], backend, shots: int = 2**13):
    tomography = {}
    for measurement in measurements:
        tomography[measurement] = qiskit_pauli_string_measurement(
            circuit, measurement, backend, shots)

    return tomography


def qiskit_perform_tomography_with_zne(circuit: q.QuantumCircuit, measurements: Set[str], backend, shots: int = 2**13, fold_factors: list = [1, 3, 5]):
    tomography = {}
    for measurement in measurements:
        tomography[measurement] = qiskit_pauli_string_measurement_with_zne(
            circuit, measurement, backend, shots, fold_factors)

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


def qiskit_probability_distribution_with_zne(circuit: q.QuantumCircuit, backend, shots=2**13, fold_factors: list = [1, 3, 5], mit: M3Mitigation = None):
    circuit = circuit.copy()
    circuit = q.transpile(circuit, backend, optimization_level=3)
    results = []
    for fold_factor in fold_factors:
        folded_circuit = qiskit_circuit_fold(circuit, fold_factor)
        folded_circuit.measure_all()
        job = backend.run(folded_circuit, shots=shots)
        result = job.result().data(0)
        counts = result.get("counts")

        folded_counts = {bin(int(state, 16))[2:].zfill(folded_circuit.num_qubits)[
            ::-1]: count for state, count in counts.items()}

        if mit is not None:
            folded_counts = mit.apply_correction(
                folded_counts, range(folded_circuit.num_qubits))

        results.append(folded_counts)

    def linear(x, a, b):
        return a * x + b

    fitted_probs = {}
    for key in results[0].keys():
        probs_to_fit = []
        for result in results:
            probs_to_fit.append(result.get(key, 0))
        params, _ = curve_fit(linear, fold_factors, probs_to_fit)
        fitted_probs[key] = linear(0, *params)

    for key, value in fitted_probs.items():
        if value < 0:
            fitted_probs[key] = 0.0
        elif value > 1:
            fitted_probs[key] = 1.0
        else:
            fitted_probs[key] = value
    total_prob = sum(fitted_probs.values())
    probabilities = {key: count / total_prob for key,
                     count in fitted_probs.items()}

    return probabilities


def slater_determinant_probability(sDet: SlaterDeterminant, prob_dist):
    orbitals = sDet.orbitals
    bit_string = ''.join([str(i) for i in orbitals])
    if bit_string not in prob_dist:
        return 0
    return prob_dist[bit_string]


def qiskit_statevector(circuit: q.QuantumCircuit):
    circuit = circuit.copy()
    backend = Aer.AerSimulator(method='statevector')
    circuit.save_statevector()
    result = backend.run(circuit).result()
    statevector = result.get_statevector()
    return statevector.data


def slater_determinant_probability_from_statevector(sDet: SlaterDeterminant, statevector):
    coef = statevector[sDet.encoding]
    return np.abs(coef)**2


def qiskit_pauli_string_measurement_statevector(circuit: q.QuantumCircuit, pString_as_string: str):
    circuit = circuit.copy()
    backend = Aer.AerSimulator(method='statevector')
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


def trim_noise_model(noise_model: Aer.noise.NoiseModel, qubit_mapping: dict):
    noise_model_as_dict = noise_model.to_dict()
    new_noise_model_as_dict = {}
    new_noise_model_as_dict["errors"] = []

    qubits_to_use = qubit_mapping.keys()

    for error in noise_model_as_dict["errors"]:
        qubits_in_error = error["gate_qubits"][0]
        add_to_new_model = True
        for qubit_in_error in qubits_in_error:
            if qubit_in_error not in qubits_to_use:
                add_to_new_model = False
                break

        if add_to_new_model:
            new_error = copy.deepcopy(error)
            new_gate_qubits = tuple(qubit_mapping.get(q, q)
                                    for q in qubits_in_error)
            new_error["gate_qubits"][0] = new_gate_qubits
            new_noise_model_as_dict["errors"].append(new_error)

    new_noise_model = Aer.noise.NoiseModel.from_dict(new_noise_model_as_dict)
    return new_noise_model


def cals_from_noise_model(mitigation: M3Mitigation, noise_model: Aer.noise.NoiseModel):
    noise_model_as_dict = noise_model.to_dict()
    ro_probs = []
    for error in noise_model_as_dict["errors"]:
        if error["type"] == "roerror":
            probabilities = np.array(error["probabilities"])
            ro_probs.append(probabilities)
    mitigation.cals_from_matrices(ro_probs)
