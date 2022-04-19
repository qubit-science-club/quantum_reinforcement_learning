import numpy as np
from scipy.linalg import logm
from torch import sum as t_sum


def entanglement_entropy(state):
    #state = np.array(state, ndmin=2)
    state = np.array(state.detach().numpy(), ndmin=2)

    ket = state.T
    bra = state.conj()
    rho_final = np.outer(ket,bra)
    num_wires = int(np.log2(state.size))
    S = []
    for d in range(1, num_wires):
        Ia = np.identity(2**d)
        Ib = np.identity(2**(num_wires-d))
        Tr_a = np.empty([2**d, 2**(num_wires-d), 2**(num_wires-d)], dtype=complex)
        for i in range(2**d):
            ai = np.array(Ia[i], ndmin=2).T
            Tr_a[i] = np.kron(ai.conj().T, Ib).dot(rho_final).dot(np.kron(ai,Ib))
        rho_b = Tr_a.sum(axis=0)
        rho_b_l2 = logm(rho_b)/np.log(2.0)
        S_rho_b = - rho_b.dot(rho_b_l2).trace()
        S.append(S_rho_b)
    return np.array(S).mean()

def classical_entropy(state):
    state_temp = state.detach()
    ket_2 = np.abs(state_temp)**2
    return - t_sum(ket_2 * np.log2(ket_2))
