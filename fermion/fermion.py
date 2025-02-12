import numpy as np

from sparse_sim import *


class FermionicOperator:
    op: str  # '+' for creation, '-' for annihilation
    idx: int  # Index of the site
    N: int  # Total number of sites / qubits
    pSum: PauliSum  # PauliSum object representing the operator using Jordan-Wigner Transformation

    def __init__(self, operator, index, N):

        assert operator in [
            '+', '-'], "op must be '+' (creation) or '-' (annihilation)"
        assert 0 <= index < N, "site must be a valid qubit index"

        self.op = operator
        self.idx = index
        self.N = N
        self.pSum = self.to_pSum()

    def to_pSum(self):

        paulis = ["I"] * self.N
        pSum = PauliSum()

        for i in range(self.idx):
            paulis[i] = "Z"

        paulis[self.idx] = "X"
        pString_x = PauliString(self.N, 0.5 + 0j, paulis)

        if self.op == "+":
            paulis[self.idx] = "Y"
            pString_y = PauliString(self.N, -0.5j, paulis)

        else:
            paulis[self.idx] = "Y"
            pString_y = PauliString(self.N, 0.5j, paulis)

        pSum.append_pauli_string(pString_x)
        pSum.append_pauli_string(pString_y)

        return pSum

    def adjoint(self):
        adjoint_op = "+"
        if self.op == "+":
            adjoint_op = "-"
        return FermionicOperator(adjoint_op, self.idx, self.N)

    def __str__(self):
        return f"{self.op}_{self.idx}"


class FermionicProduct:
    coef: complex  # Complex coefficient of the product
    ops: list  # List of FermionicOperators applied right to left
    N: int  # Total number of sites / qubits
    pSum: PauliSum  # PauliSum object representing the product using Jordan-Wigner Transformation

    def __init__(self, coef, ops, N):

        self.coef = coef
        self.ops = ops
        self.N = N
        self.pSum = self.to_pSum()

    def to_pSum(self):

        pSum = self.ops[0].pSum
        pSum = pauli_sum_scalar_multiplication(pSum, self.coef)

        for i in range(1, len(self.ops)):
            next_pSum = self.ops[i].pSum
            pSum = pauli_sum_multiplication(pSum, next_pSum)

        return pSum

    def adjoint(self):
        adjoint_coef = np.conj(self.coef)
        adjoint_ops = []
        for op in self.ops[::-1]:
            adjoint_ops.append(op.adjoint())
        return FermionicProduct(adjoint_coef, adjoint_ops, self.N)

    def evaluate_expectation(self, tomography):
        return pauli_sum_evaluate_expectation(self.pSum, tomography)

    def __str__(self):
        output = f"{self.coef} * ("
        for i, op in enumerate(self.ops):
            if i > 0:
                output = output + " "
            output = output + f"{op}"
        output = output + ")"
        return output


class Operator:
    fProds: list  # List of FermionicProducts
    N: int  # Total number of sites / qubits
    symbol: str  # Symbol used for printing the operator

    def __init__(self, fProds, N, symbol="Symbol Not Set"):

        self.fProds = fProds
        self.N = N
        self.symbol = symbol

    def add_fermoinic_product(self, fProd):
        self.fProds.append(fProd)

    def aggregate_measurements(self):
        measurements = set()
        for fProd in self.fProds:
            pSum = fProd.pSum
            measurements.update(pauli_sum_collect_measurements(pSum))

        return measurements

    def evaluate_expectation(self, tomography):

        exp = 0 + 0j
        for fProd in self.fProds:
            exp += fProd.evaluate_expectation(tomography)

        return exp

    def adjoint(self):
        adjoint_fProds = []
        for fProd in self.fProds:
            adjoint_fProds.append(fProd.adjoint())
        return Operator(adjoint_fProds, self.N)

    def __str__(self):
        return self.symbol
