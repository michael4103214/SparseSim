from fermion import *
import itertools


class RDM(Operator):
    p: int  # Order of rdm
    prods: list  # List of FermionicProducts
    N: int  # Total number of sites / qubits
    symbol: str  # Symbol used for printing the operator

    def __init__(self, p, N):
        self.p = p

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
