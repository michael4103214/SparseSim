from fermion import *
import itertools


class RDM(Operator):
    p: int  # Order of rdm
    fProds: list  # List of FermionicProducts
    N: int  # Total number of sites / qubits
    symbol: str  # Symbol used for printing the operator

    def __init__(self, p, N):
        self.p = p

        fProds = generate_fProds(p, N)
        super().__init__(fProds, N, f"{p}D")

    def trace(self, tomography):
        if self.p != 1:
            print(f"Trace is not defined for p={self.p}")
            return 0

        tr = 0 + 0j
        for fProd in self.fProds:
            if fProd.ops[0].idx == fProd.ops[1].idx:
                tr = tr + fProd.evaluate_expectation(tomography)

        return tr


def generate_fProds(p, N):
    fProds = []

    indices = range(N)

    for idx_tuple in itertools.product(itertools.permutations(indices, p), itertools.permutations(indices, p)):
        fOps = []
        for idx, site in enumerate(idx_tuple[0] + idx_tuple[1]):
            sign = '+' if idx < p else '-'
            fOps.append(FermionicOperator(sign, site, N))
        fProd = FermionicProduct(1, fOps, N)
        fProds.append(fProd)

    return fProds
