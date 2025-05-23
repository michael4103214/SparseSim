from fermion import *
from pyscf import gto, scf, ao2mo
from pyscf.gto import Mole


class Hamiltonian(Operator):
    prods: list  # List of Products
    N: int  # Total number of sites / qubits
    symbol: str  # Symbol used for printing the operator
    nuc: float  # Nuclear-nuclear repulsion energy

    def __init__(self, prods, nuc, N, symbol="H"):
        super().__init__(prods, N, symbol)
        self.nuc = nuc

    def energy(self, tomography):
        return self.nuc + self.evaluate_expectation(tomography).real

    def map(self, projector):
        new_prods = []

        for prod in self.prods:
            new_prods.extend(prod.map(projector))

        return Hamiltonian(new_prods, self.nuc, projector.target_N)


def init_Hamiltonian_from_pyscf(mol, mf=None):
    if mf is None:
        mf = scf.ROHF(mol)
        mf.verbose = 0
        mf.run()

    C = mf.mo_coeff

    h_core = mf.get_hcore()
    # Convert to MO basis
    h_mo = C.T @ h_core @ C

    Na = h_mo.shape[0]
    Nb = Na
    N = Na + Nb
    # Compute two-electron integrals in AO basis
    eri_ao = mol.intor("int2e")
    # Convert to MO basis, uncompress the tensor to full 4-index form
    eri_mo = ao2mo.incore.full(
        eri_ao, C, compact=False).reshape(Na, Na, Na, Na).transpose(0, 2, 1, 3)

    prods = []
    for i in range(Na):
        for j in range(Na):
            if abs(h_mo[i, j]) > 1e-10:
                fOp1 = FermionicOperator('+', i, N)
                fOp2 = FermionicOperator('-', j, N)
                prod = Product(h_mo[i, j], [fOp1, fOp2], N)
                prods.append(prod)

    for i in range(Nb):
        for j in range(Nb):
            if abs(h_mo[i, j]) > 1e-10:
                fOp1 = FermionicOperator('+', Na + i, N)
                fOp2 = FermionicOperator('-', Na + j, N)
                prod = Product(h_mo[i, j], [fOp1, fOp2], N)
                prods.append(prod)

    for i in range(Na):
        for j in range(Na):
            for k in range(Na):
                for l in range(Na):
                    if (i != j) and (l != k):
                        val = eri_mo[i, j, k, l]
                        if abs(val) > 1e-10:
                            fOp1 = FermionicOperator('+', i, N)
                            fOp2 = FermionicOperator('+', j, N)
                            fOp3 = FermionicOperator('-', l, N)
                            fOp4 = FermionicOperator('-', k, N)
                            prod = Product(
                                0.5 * val, [fOp1, fOp2, fOp3, fOp4], N)
                            prods.append(prod)

    for i in range(Na):
        for j in range(Nb):
            for k in range(Na):
                for l in range(Nb):
                    val = eri_mo[i, j, k, l]
                    if abs(val) > 1e-10:
                        fOp1 = FermionicOperator('+', i, N)
                        fOp2 = FermionicOperator('+', Na + j, N)
                        fOp3 = FermionicOperator('-', Na + l, N)
                        fOp4 = FermionicOperator('-', k, N)
                        prod = Product(
                            0.5 * val, [fOp1, fOp2, fOp3, fOp4], N)
                        prods.append(prod)

    for i in range(Nb):
        for j in range(Na):
            for k in range(Nb):
                for l in range(Na):
                    val = eri_mo[i, j, k, l]
                    if abs(val) > 1e-10:
                        fOp1 = FermionicOperator('+', Na + i, N)
                        fOp2 = FermionicOperator('+', j, N)
                        fOp3 = FermionicOperator('-', l, N)
                        fOp4 = FermionicOperator('-', Na + k, N)
                        prod = Product(
                            0.5 * val, [fOp1, fOp2, fOp3, fOp4], N)
                        prods.append(prod)

    for i in range(Nb):
        for j in range(Nb):
            for k in range(Nb):
                for l in range(Nb):
                    if (i != j) and (l != k):
                        val = eri_mo[i, j, k, l]
                        if abs(val) > 1e-10:
                            fOp1 = FermionicOperator('+', Na + i, N)
                            fOp2 = FermionicOperator('+', Na + j, N)
                            fOp3 = FermionicOperator('-', Na + l, N)
                            fOp4 = FermionicOperator('-', Na + k, N)
                            prod = Product(
                                0.5 * val, [fOp1, fOp2, fOp3, fOp4], N)
                            prods.append(prod)

    return Hamiltonian(prods, mf.energy_nuc(), N)
