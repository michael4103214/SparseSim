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


def generate_fProds(p, N):
    fProds = []

    indices = range(N)

    for idx_tuple in itertools.product(indices, repeat=2 * p):
        fOps = []
        for idx, site in enumerate(idx_tuple):
            sign = '+' if idx < p else '-'
            fOps.append(FermionicOperator(sign, site, N))
        fProd = FermionicProduct(1, fOps, N)
        fProds.append(fProd)

    return fProds
