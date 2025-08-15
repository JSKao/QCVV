# === Quantum State & Entanglement Unified API ===
"""
state_utils.py
----------------
Basic quantum state generation (GHZ, W, Dicke, Cluster, Graph), 
density matrix operations, entropy and spectrum calculation, 
and analytic entanglement witness (e.g., Mermin operator).

Main functions:
- ghz_state, w_state, dicke_state, cluster_state, graph_state
- density_matrix, partial_trace
- von_neumann_entropy, renyi_entropy, schmidt_spectrum
- mermin_operator, mermin_witness, check_mermin_witness
"""

import numpy as np
from itertools import combinations, product
import cvxpy as cp

# --- Numerics API  ---

# 1. State generation
def ghz_state(N):
    """GHZ state |00...0> + |11...1>"""
    state = np.zeros(2**N, dtype=complex)
    state[0] = 1/np.sqrt(2)
    state[-1] = 1/np.sqrt(2)
    return state

def w_state(N):
    """W state: equal superposition of single excitation"""
    state = np.zeros(2**N, dtype=complex)
    for i in range(N):
        state[2**i] = 1
    return state / np.sqrt(N)

def dicke_state(N, k):
    """Dicke state: symmetric k-excitation"""
    state = np.zeros(2**N, dtype=complex)
    for pos in combinations(range(N), k):
        idx = sum([2**p for p in pos])
        state[idx] = 1
    return state / np.sqrt(np.math.comb(N, k))

# Cluster state
def cluster_state(N):
    """1D cluster state on N qubits."""
    state = np.ones(2**N, dtype=complex) / np.sqrt(2**N)
    for i in range(N-1):
        for j in range(2**N):
            # If both qubit i and i+1 are 1, add a -1 phase (CZ gate)
            if ((j >> i) & 1) and ((j >> (i+1)) & 1):
                state[j] *= -1
    return state

# Graph state (define from adjacency matrix)
def graph_state(adj_matrix):
    """General graph state from adjacency matrix."""
    N = adj_matrix.shape[0]
    state = np.ones(2**N, dtype=complex) / np.sqrt(2**N)
    for i in range(N):
        for j in range(i+1, N):
            if adj_matrix[i, j]:
                for k in range(2**N):
                    if ((k >> i) & 1) and ((k >> j) & 1):
                        state[k] *= -1
    return state

# 2. Density matrix
def density_matrix(state):
    """Return density matrix for pure state or mixed state"""
    if state.ndim == 1:
        return np.outer(state, np.conj(state))
    return state

# 3. Partial trace
def partial_trace(rho, keep, dims):
    """
    Partial trace over subsystems not in 'keep'.
    rho: density matrix (2^N x 2^N)
    keep: list of subsystem indices to keep (e.g., [0,2])
    dims: list of dimensions for each subsystem (e.g., [2,2,2] for 3 qubits)
    """
    N = len(dims)
    traced = list(set(range(N)) - set(keep))
    reshaped = rho.reshape([2]*N*2)
    perm = keep + [i for i in range(N) if i not in keep]
    perm2 = [i+N for i in perm]
    perm_full = perm + perm2
    reshaped = np.transpose(reshaped, perm_full)
    d_keep = int(np.prod([dims[i] for i in keep]))
    d_trace = int(np.prod([dims[i] for i in traced]))
    return np.trace(reshaped.reshape(d_keep, d_trace, d_keep, d_trace), axis1=1, axis2=3)

# 4. Entropy
def von_neumann_entropy(rho):
    """Von Neumann entropy for density matrix"""
    eigvals = np.linalg.eigvalsh(rho)
    eigvals = eigvals[eigvals > 1e-12]
    return -np.sum(eigvals * np.log2(eigvals))

def renyi_entropy(rho, alpha=2):
    """Renyi entropy of order alpha"""
    eigvals = np.linalg.eigvalsh(rho)
    eigvals = eigvals[eigvals > 1e-12]
    return 1/(1-alpha) * np.log2(np.sum(eigvals**alpha))

# 5. Schmidt spectrum (for bipartition)
def schmidt_spectrum(state, N, k):
    """
    Return Schmidt coefficients for bipartition: first k qubits vs rest
    """
    psi = state.reshape([2]*N).reshape(2**k, 2**(N-k))
    u, s, vh = np.linalg.svd(psi, full_matrices=False)
    return s

def negativity(rho, dims):
    """
    計算二體密度矩陣的 negativity（partial transpose 負特徵值之和）。
    """
    dA, dB = dims
    # Partial transpose on B
    rho_pt = rho.reshape(dA, dB, dA, dB).transpose(0,3,2,1).reshape(dA*dB, dA*dB)
    eigs = np.linalg.eigvalsh(rho_pt)
    return np.sum(np.abs(eigs[eigs < 0]))

def robustness_of_entanglement(rho, dims, s_min=0, s_max=2, tol=1e-3, max_iter=20):
    """
    以二分法近似計算 entanglement robustness。
    """
    dA, dB = dims
    I = np.eye(dA*dB)
    def partial_transpose_cvx(X, dims, sys=1):
        X = cp.reshape(X, (dA, dB, dA, dB), order='F')
        if sys == 0:
            X = cp.transpose(X, (2,1,0,3))
        else:
            X = cp.transpose(X, (0,3,2,1))
        return cp.reshape(X, (dA*dB, dA*dB), order='F')
    def is_feasible(s):
        X = cp.Variable((dA*dB, dA*dB), hermitian=True)
        constraints = [X >> 0, cp.trace(X) == 1]
        X_pt = partial_transpose_cvx(X, dims, sys=1)
        constraints += [X_pt >> 0]
        constraints += [X == (rho + s*I)/(1+s)]
        prob = cp.Problem(cp.Minimize(0), constraints)
        try:
            prob.solve(solver=cp.SCS, verbose=False)
            return prob.status == cp.OPTIMAL or prob.status == cp.FEASIBLE
        except Exception:
            return False

    # 二分法搜尋最小 s
    left, right = s_min, s_max
    for _ in range(max_iter):
        mid = (left + right) / 2
        if is_feasible(mid):
            right = mid
        else:
            left = mid
        if right - left < tol:
            break
    return right

def mermin_lhv_bound(N):
    # LHV bound for Mermin inequality
    if N % 2 == 1:
        return 2 ** ((N-1)//2)
    else:
        return 2 ** (N//2)
    
def mermin_operator(N):
    X = np.array([[0,1],[1,0]], dtype=complex)
    Y = np.array([[0,-1j],[1j,0]], dtype=complex)
    if N == 3:
        # M_3 = X⊗X⊗X - X⊗Y⊗Y - Y⊗X⊗Y - Y⊗Y⊗X
        return (
            np.kron(np.kron(X, X), X)
            - np.kron(np.kron(X, Y), Y)
            - np.kron(np.kron(Y, X), Y)
            - np.kron(np.kron(Y, Y), X)
        )
    elif N == 4:
        # M_4 = sum over all bitstrings with odd number of Y, sign = +1
        terms = []
        for bits in product([0,1], repeat=N):
            if sum(bits) % 2 == 1:
                op = 1
                for b in bits:
                    op = np.kron(op, X if b==0 else Y)
                terms.append(op)
        return sum(terms)
    elif N == 5:
        # M_5 = sum over all bitstrings with odd number of Y, sign = (-1)^{(nY-1)//2}
        terms = []
        for bits in product([0,1], repeat=N):
            nY = sum(bits)
            if nY % 2 == 1:
                sign = (-1)**((nY-1)//2)
                op = 1
                for b in bits:
                    op = np.kron(op, X if b==0 else Y)
                terms.append(sign * op)
        return sum(terms)
    else:
        raise NotImplementedError("Only N=3,4,5 supported.")

def mermin_witness(N):
    """
    W = LHV_bound * I - M_N
    """
    d = 2**N
    I = np.eye(d)
    M_N = mermin_operator(N)
    bound = mermin_lhv_bound(N)
    W = bound * I - M_N
    return W

def check_mermin_witness(rho, N):
    """
    計算 N-qubit 態 rho 的 Mermin witness 期望值
    """
    W = mermin_witness(N)
    val = np.real(np.trace(W @ rho))
    return val



# Based on what you need, you can implement symbolic operations for specific cases.
# For example, for small N, you can manually write out the steps for symbolic operations.
# For larger N, symbolic operations become complex and may require specialized libraries.
# For small N, you can manually write out the steps for symbolic operations.    
