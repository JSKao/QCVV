"""
symbolic_utils.py
-----------------
Symbolic (sympy-based) quantum state and entanglement tools.
Includes:
- Symbolic state generation (GHZ, W, Dicke)
- Symbolic density matrix and partial trace
- Symbolic entropy and Schmidt spectrum calculation
"""
# --- Symbolic operation API ---
import sympy as sp
import cvxpy as cp

def w_state_symbolic(N):
    """
    Return symbolic W state (sympy Matrix) for N qubits.
    |W_N> = 1/sqrt(N) (|100...0> + |010...0> + ... + |000...1>)
    """
    state = [0]*2**N
    for i in range(N):
        idx = 2**i
        state[idx] = 1
    norm = sp.sqrt(N)
    return sp.Matrix(state) / norm

def ghz_state_symbolic(N):
    """GHZ state |00...0> + |11...1> (symbolic)"""
    basis = [sp.zeros(2**N, 1) for _ in range(2)]
    basis[0][0] = 1
    basis[1][-1] = 1
    state = (basis[0] + basis[1]) / sp.sqrt(2)
    return state

def density_matrix_symbolic(state):
    """Symbolic density matrix for pure state"""
    return state * state.H

def partial_trace_symbolic(rho, keep, dims):
    """
    Symbolic partial trace over subsystems not in 'keep'.
    Only supports qubits (dims=[2,...]).
    """
    N = len(dims)
    traced = list(set(range(N)) - set(keep))
    # Build indices for kept/traced qubits
    idx = [sp.Idx(f'i{j}', 1) for j in range(N)]
    # This is a placeholder; symbolic partial trace for general N is complex.
    # For small N, you can manually sum over traced indices.
    raise NotImplementedError("General symbolic partial trace not implemented. For small N, expand manually.")

def symbolic_entropy(rho, base=2):
    """自動計算 symbolic von Neumann entropy, 支援任意維度"""
    eigval_dict = rho.eigenvals()
    S = 0
    for val, mult in eigval_dict.items():
        # 只考慮正實數本徵值
        if val.is_real and val > 0:
            S += -mult * val * sp.log(val, base)
    return sp.simplify(S)

def symbolic_renyi_entropy(rho, alpha, base=2):
    """自動計算 symbolic Renyi entropy, 支援任意維度"""
    eigval_dict = rho.eigenvals()
    sumeig = 0
    for val, mult in eigval_dict.items():
        if val.is_real and val > 0:
            sumeig += mult * val**alpha
    S_renyi = 1/(1-alpha) * sp.log(sumeig, base)
    return sp.simplify(S_renyi)

def partial_trace_symbolic_2qubit(rho, keep=1):
    """
    Symbolic partial trace for 2-qubit density matrix.
    keep=0: trace out qubit 1, keep qubit 0
    keep=1: trace out qubit 0, keep qubit 1
    """
    # rho: 4x4 sympy Matrix
    # basis: |00>, |01>, |10>, |11>
    if keep == 0:
        # keep qubit 0, trace out qubit 1
        # ρ_A[i,j] = sum_k ρ[2*i+k, 2*j+k]
        return sp.Matrix([
            [rho[0,0] + rho[1,1], rho[0,2] + rho[1,3]],
            [rho[2,0] + rho[3,1], rho[2,2] + rho[3,3]]
        ])
    elif keep == 1:
        # keep qubit 1, trace out qubit 0
        # ρ_A[i,j] = sum_k ρ[k*2+i, k*2+j]
        return sp.Matrix([
            [rho[0,0] + rho[2,2], rho[0,1] + rho[2,3]],
            [rho[1,0] + rho[3,2], rho[1,1] + rho[3,3]]
        ])
    else:
        raise ValueError("keep must be 0 or 1 for 2-qubit system.")
    
def partial_trace_symbolic_3qubit(rho, keep=[0,1]):
    """
    Symbolic partial trace for 3-qubit density matrix.
    keep: list of qubit indices to keep (e.g. [0,1], [0,2], [1,2])
    Returns: 4x4 sympy Matrix (2 qubits)
    """
    if sorted(keep) == [0,1]:
        # trace out qubit 2
        # ρ_A[i,j] = sum_k ρ[4*i+k, 4*j+k], k=0,1,2,3
        return sp.Matrix([
            [rho[0,0]+rho[1,1]+rho[2,2]+rho[3,3], rho[0,4]+rho[1,5]+rho[2,6]+rho[3,7], rho[4,0]+rho[5,1]+rho[6,2]+rho[7,3], rho[4,4]+rho[5,5]+rho[6,6]+rho[7,7]],
            [rho[0,4]+rho[1,5]+rho[2,6]+rho[3,7], rho[0,0]+rho[1,1]+rho[2,2]+rho[3,3], rho[4,0]+rho[5,1]+rho[6,2]+rho[7,3], rho[4,4]+rho[5,5]+rho[6,6]+rho[7,7]],
            [rho[4,0]+rho[5,1]+rho[6,2]+rho[7,3], rho[4,0]+rho[5,1]+rho[6,2]+rho[7,3], rho[0,0]+rho[1,1]+rho[2,2]+rho[3,3], rho[0,4]+rho[1,5]+rho[2,6]+rho[3,7]],
            [rho[4,4]+rho[5,5]+rho[6,6]+rho[7,7], rho[4,4]+rho[5,5]+rho[6,6]+rho[7,7], rho[0,4]+rho[1,5]+rho[2,6]+rho[3,7], rho[0,0]+rho[1,1]+rho[2,2]+rho[3,3]]
        ])
    elif sorted(keep) == [0,2]:
        # trace out qubit 1
        # ρ_A[i,j] = sum_k ρ[2*i0+4*i2+k*2, 2*j0+4*j2+k*2], k=0,1
        # i0,i2 in {0,1}
        mat = sp.zeros(4,4)
        for i0 in range(2):
            for i2 in range(2):
                for j0 in range(2):
                    for j2 in range(2):
                        s = 0
                        for k in range(2):
                            row = i0*4 + k*2 + i2
                            col = j0*4 + k*2 + j2
                            s += rho[row, col]
                        mat[i0*2+i2, j0*2+j2] = s
        return mat
    elif sorted(keep) == [1,2]:
        # trace out qubit 0
        # ρ_A[i,j] = sum_k ρ[i1*2+i2+k*4, j1*2+j2+k*4], k=0,1
        mat = sp.zeros(4,4)
        for i1 in range(2):
            for i2 in range(2):
                for j1 in range(2):
                    for j2 in range(2):
                        s = 0
                        for k in range(2):
                            row = k*4 + i1*2 + i2
                            col = k*4 + j1*2 + j2
                            s += rho[row, col]
                        mat[i1*2+i2, j1*2+j2] = s
        return mat
    else:
        raise ValueError("keep must be two distinct qubit indices from [0,1,2]")    
    
def symbolic_schmidt_spectrum(psi, N, k):
    """
    Symbolic Schmidt spectrum for bipartition: first k qubits vs rest.
    psi: sympy column vector (2**N x 1)
    N: total qubits
    k: number of qubits in subsystem A
    Returns: list of symbolic Schmidt coefficients (not squared)
    """
    # reshape to (2**k, 2**(N-k))
    psi_mat = sp.Matrix(psi).reshape(2**k, 2**(N-k))
    # SVD: psi_mat = U * S * V^dagger
    # For symbolic, use .singular_values()
    s = psi_mat.singular_values()
    return s    