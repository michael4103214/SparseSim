from fermion import *
import itertools
import numpy as np


class RDM(Operator):
    N: int  # Total number of sites / qubits
    p: int  # Order of rdm
    prods: list  # List of FermionicProducts'
    symbol: str  # Symbol used for printing the operator

    def __init__(self, p, N, prods=[]):
        self.p = p

        if len(prods) == 0:
            prods = generate_prods(p, N)

        super().__init__(prods, N, f"{p}D")

    def trace(self, tomography):
        if self.p != 1:
            print(f"Trace is not defined for p={self.p}")
            return 0

        tr = 0 + 0j
        for prod in self.prods:
            if prod.ops[0].idx == prod.ops[1].idx:
                tr = tr + prod.evaluate_expectation(tomography)

        return tr

    def save(self):
        output = [np.array([self.N, self.p])]
        for prod in self.prods:
            output.append(prod.save())
        output = np.array(output, dtype=object)

        return output


def load_rdm(data):
    row0 = data[0]
    N = int(row0[0])
    p = int(row0[1])
    prods = []

    for i in range(1, len(data)):
        prod_data = data[i]
        prod = load_product(prod_data)
        prods.append(prod)
    return RDM(p, N, prods)


def generate_prods(p, N):
    prods = []

    indices = range(N)

    for idx_tuple in itertools.product(itertools.permutations(indices, p), itertools.permutations(indices, p)):
        fOps = []
        for idx, site in enumerate(idx_tuple[0] + idx_tuple[1]):
            sign = '+' if idx < p else '-'
            fOps.append(FermionicOperator(sign, site, N))
        prod = Product(1, fOps, N)
        prods.append(prod)

    return prods
