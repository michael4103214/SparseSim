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


def init_Hamiltonian_from_pyscf(mol):
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


'''def init_Hamiltonian_from_pyscf(mol):
    mf = scf.ROHF(mol)
    mf.verbose = 0
    mf.run()

    C_alpha, C_beta = mf.mo_coeff

    # Compute one-electron integrals in AO basis
    h_core = mf.get_hcore()
    # Convert to MO basis
    h_mo_alpha = C_alpha.T @ h_core @ C_alpha
    h_mo_beta = C_beta.T @ h_core @ C_beta

    Na = h_mo_alpha.shape[0]
    Nb = h_mo_beta.shape[0]
    N = Na + Nb

    # Compute two-electron integrals in AO basis
    eri_ao = mol.intor("int2e")
    # Convert to MO basis, uncompress the tensor to full 4-index form
    eri_mo_aa = ao2mo.incore.full(
        eri_ao, C_alpha, compact=False).reshape(Na, Na, Na, Na).transpose(0, 2, 1, 3)
    eri_mo_ab = ao2mo.incore.general(
        eri_ao, (C_alpha, C_alpha, C_beta, C_beta), compact=False).reshape(Na, Na, Nb, Nb).transpose(0, 2, 1, 3)
    eri_mo_ba = ao2mo.incore.general(
        eri_ao, (C_beta, C_beta, C_alpha, C_alpha), compact=False).reshape(Nb, Nb, Na, Na).transpose(0, 2, 1, 3)
    eri_mo_bb = ao2mo.incore.full(
        eri_ao, C_beta, compact=False).reshape(Na, Na, Na, Na).transpose(0, 2, 1, 3)

    prods = []

    for i in range(Na):
        for j in range(Na):
            if abs(h_mo_alpha[i, j]) > 1e-10:
                fOp1 = FermionicOperator('+', i, N)
                fOp2 = FermionicOperator('-', j, N)
                prod = Product(h_mo_alpha[i, j], [fOp1, fOp2], N)
                prods.append(prod)

    for i in range(Nb):
        for j in range(Nb):
            if abs(h_mo_beta[i, j]) > 1e-10:
                fOp1 = FermionicOperator('+', Na + i, N)
                fOp2 = FermionicOperator('-', Na + j, N)
                prod = Product(h_mo_beta[i, j], [fOp1, fOp2], N)
                prods.append(prod)

    for i in range(Na):
        for j in range(Na):
            for k in range(Na):
                for l in range(Na):
                    if (i != j) and (l != k):
                        val = eri_mo_aa[i, j, k, l]
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
                    val = eri_mo_ab[i, j, k, l]
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
                    val = eri_mo_ba[i, j, k, l]
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
                        val = eri_mo_bb[i, j, k, l]
                        if abs(val) > 1e-10:
                            fOp1 = FermionicOperator('+', Na + i, N)
                            fOp2 = FermionicOperator('+', Na + j, N)
                            fOp3 = FermionicOperator('-', Na + l, N)
                            fOp4 = FermionicOperator('-', Na + k, N)
                            prod = Product(
                                0.5 * val, [fOp1, fOp2, fOp3, fOp4], N)
                            prods.append(prod)

    return Hamiltonian(prods, mf.energy_nuc(), N)'''
