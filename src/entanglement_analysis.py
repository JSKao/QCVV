from state_utils import *
"""
batch_analysis.py
-----------------
High-level batch and global analysis tools for multipartite entanglement.
Includes:
- Batch Mermin witness checking for multiple states and system sizes
- Automated analysis of all bipartitions (entropy, spectrum, negativity)
- Genuine multipartite entanglement checks
- PPT and separability checks for all bipartitions
"""
import numpy as np
import cvxpy as cp

def analyze_all_bipartitions(state, N):
    """
    對所有 k|N-k 分割自動計算 Schmidt spectrum, 熵, negativity
    state: 1D numpy array, 純態
    N: qubit 數
    """
    print(f"=== All bipartitions for N={N} qubits ===")
    for k in range(1, N//2 + 1):
        # Schmidt spectrum
        psi = state.reshape([2]*N).reshape(2**k, 2**(N-k))
        u, s, vh = np.linalg.svd(psi, full_matrices=False)
        entropy = -np.sum(s**2 * np.log2(s**2 + 1e-12))
        print(f"Partition {k}|{N-k}: Entropy = {entropy:.4f}, Schmidt spectrum = {np.round(s,4)}")
        # Negativity (for bipartition)
        rho = np.outer(state, state.conj())
        dims = [2**k, 2**(N-k)]
        rho_reshaped = rho.reshape(dims[0], dims[1], dims[0], dims[1]).transpose(0,1,2,3).reshape(dims[0]*dims[1], dims[0]*dims[1])
        # Partial transpose on the second subsystem
        rho_pt = rho_reshaped.reshape(dims[0], dims[1], dims[0], dims[1]).transpose(0,3,2,1).reshape(dims[0]*dims[1], dims[0]*dims[1])
        eigs = np.linalg.eigvalsh(rho_pt)
        negativity = np.sum(np.abs(eigs[eigs < 0]))
        print(f"  Negativity: {negativity:.4f}")

def sdp_witness_all_bipartitions(state, N, verbose=True):
    """
    對所有 k|N-k 分割自動計算最優 witness，回傳 violation 結果。
    """
    results = []
    rho = np.outer(state, state.conj())  # 全系統密度矩陣
    for k in range(1, N//2 + 1):
        dims = [2**k, 2**(N-k)]
        print(f"Partition {k}|{N-k}, rho shape: {rho.shape}, dims: {dims}")
        try:
            W, val = optimal_entanglement_witness(rho, dims)
        except Exception as e:
            print("Exception:", e)
            val = None
        results.append((k, N-k, val))
        if verbose:
            print(f"Partition {k}|{N-k}: SDP witness value = {val}")
    return results

def batch_check_mermin_witness(state_fns, N_list, names):
    """
    批次計算多種態的 Mermin witness 期望值
    state_fns: list of state function, e.g. [ghz_state, w_state, dicke_state_k2]
    N_list: list of N, e.g. [3, 4, 5]
    names: list of state names, e.g. ["GHZ", "W", "Dicke"]
    """
    for state_fn, name in zip(state_fns, names):
        for N in N_list:
            if name == "Dicke":
                state = state_fn(N, 2)  # Dicke 需指定 k
            else:
                state = state_fn(N)
            rho = np.outer(state, state.conj())
            val = check_mermin_witness(rho, N)
            print(f"{name} (N={N}): Mermin witness = {val:.4f} ({'糾纏' if val<0 else '非糾纏'})")
            
def is_genuine_multipartite_entangled(psi, N, symbolic=False, tol=1e-8):
    """
    判斷 psi 是否為 genuine multipartite entangled state。
    方法：檢查所有非平凡 bipartition 的 Schmidt rank 是否都大於 1。
    psi: numpy array 或 sympy Matrix (2**N,)
    N: qubit 數
    symbolic: 是否用 symbolic 判斷（True 則用 sympy，False 則用 numpy）
    tol: 數值判斷零的容忍度
    回傳: True/False
    """
    from itertools import combinations
    for k in range(1, N//2 + 1):
        for A in combinations(range(N), k):
            # bipartition: A vs rest
            if symbolic:
                # symbolic reshape
                psi_mat = sp.Matrix(psi).reshape(2**k, 2**(N-k))
                s = psi_mat.singular_values()
                nonzero = [x for x in s if x != 0]
            else:
                psi_mat = psi.reshape([2]*N).reshape(2**k, 2**(N-k))
                s = np.linalg.svd(psi_mat, compute_uv=False)
                nonzero = [x for x in s if abs(x) > tol]
            if len(nonzero) == 1:
                # 有一個分割是 product state，則不是 GME
                return False
    return True

def all_bipartite_PPT_checks(rho, dims):
    """
    對 N qubit/mode 的 rho，自動檢查所有二體 reduced density matrix 的 PPT（分離性）性質。
    回傳：每一對 (i,j) 的 PPT 結果字典
    """
    N = len(dims)
    results = {}
    for i in range(N):
        for j in range(i+1, N):
            # trace out其他qubit，只保留i,j
            rho_ij = partial_trace(rho, [i, j], dims)
            ppt = is_separable_PPT(rho_ij, [dims[i], dims[j]])
            results[(i, j)] = ppt
    return results

def all_bipartition_PPT_checks(rho, dims):
    """
    對 N qubit/mode 的 rho，自動檢查所有 bipartition 的 PPT（分離性）性質。
    回傳：每個 bipartition (A, B) 的 PPT 結果字典
    """
    N = len(dims)
    results = {}
    for k in range(1, N//2 + 1):
        for A in combinations(range(N), k):
            B = tuple(i for i in range(N) if i not in A)
            # 對 A|B 做 partial trace，保留 A+B
            keep = list(A) + list(B)
            rho_AB = partial_trace(rho, keep, dims)
            # reshape為dA x dB
            dA = np.prod([dims[i] for i in A])
            dB = np.prod([dims[i] for i in B])
            rho_AB = rho_AB.reshape(dA, dB, dA, dB).transpose(0,1,2,3).reshape(dA*dB, dA*dB)
            # 對 B 做 partial transpose
            def partial_transpose_general(rho, dA, dB):
                rho = rho.reshape(dA, dB, dA, dB)
                rho_pt = np.transpose(rho, (0,3,2,1)).reshape(dA*dB, dA*dB)
                return rho_pt
            rho_pt = partial_transpose_general(rho_AB, dA, dB)
            eigs = np.linalg.eigvalsh(rho_pt)
            ppt = np.all(eigs >= -1e-8)
            results[(A, B)] = ppt
    return results

def partial_transpose_cvx(W, dims, subsystem=1):
    """
    Partial transpose for cvxpy Variable (SDP witness).
    """
    dA, dB = dims
    W = cp.reshape(W, (dA, dB, dA, dB), order='F')
    if subsystem == 0:
        W = cp.transpose(W, (2,1,0,3))
    else:
        W = cp.transpose(W, (0,3,2,1))
    return cp.reshape(W, (dA*dB, dA*dB), order='F')

def is_separable_SDP(rho, dims):
    """
    用半正定規劃(SDP)檢查二體密度矩陣是否可分離。
    這裡用最簡單的 PPT 條件（partial transpose 仍為正半定）。
    """
    dA, dB = dims
    # 定義變數
    X = cp.Variable((dA*dB, dA*dB), hermitian=True)
    constraints = [X >> 0, cp.trace(X) == 1]
    # Partial transpose on B
    def partial_transpose_cvx(X, dims, sys=1):
        X = cp.reshape(X, (dA, dB, dA, dB), order='F')
        if sys == 0:
            X = cp.transpose(X, (2,1,0,3))
        else:
            X = cp.transpose(X, (0,3,2,1))
        return cp.reshape(X, (dA*dB, dA*dB), order='F')
    X_pt = partial_transpose_cvx(X, dims, sys=1)
    constraints += [X_pt >> 0]
    # 目標：使 X 盡量接近 rho
    obj = cp.Minimize(cp.norm(X - rho, 'fro'))
    prob = cp.Problem(obj, constraints)
    prob.solve(solver=cp.SCS)
    return prob.status == cp.OPTIMAL and prob.value < 1e-6        

def is_separable_PPT(rho, dims):
    """
    用 partial transpose 判斷兩體密度矩陣是否為 PPT（必要但非充分條件）。
    rho: numpy array, density matrix
    dims: [dA, dB]
    回傳: True (PPT, 可能可分), False (NPT, 必定糾纏)
    """
    def partial_transpose(rho, dims, sys=1):
        # sys=0: transpose A, sys=1: transpose B
        dA, dB = dims
        rho = rho.reshape([dA, dB, dA, dB])
        if sys == 0:
            rho = np.transpose(rho, (2,1,0,3))
        else:
            rho = np.transpose(rho, (0,3,2,1))
        return rho.reshape(dA*dB, dA*dB)
    # 對 B 做 partial transpose
    rho_pt = partial_transpose(rho, dims, sys=1)
    eigs = np.linalg.eigvalsh(rho_pt)
    return np.all(eigs >= -1e-8)    

def trivial_witness_SDP(rho, dims):
    """
    WARNING: This is NOT a valid entanglement witness construction!

    This SDP only requires W to be Hermitian, positive semidefinite, and trace one (i.e., W is a valid quantum state).
    For any physical quantum state rho (which is also PSD), Tr[W rho] >= 0 always holds.
    Therefore, this cannot detect entanglement and is only a mathematical toy model or counterexample.
    Use only for illustration or as a baseline; do NOT use for real entanglement detection.
    """
    d = np.prod(dims)
    rho = np.asarray(rho, dtype=np.complex128)
    W = cp.Variable((d, d), hermitian=True)
    constraints = [cp.trace(W) == 1, W >> 0]  # 必須同時有 trace=1 和正半定
    prob = cp.Problem(cp.Minimize(cp.real(cp.trace(W @ rho))), constraints)
    prob.solve(solver=cp.SCS)
    print("SDP status:", prob.status)
    print("W.value is None?", W.value is None)
    W_val = W.value
    witness_val = np.trace(W_val @ rho) if W_val is not None else None
    return W_val, witness_val

def optimal_PPT_witness(rho, dims):
    """使用 PPT 條件的 entanglement witness SDP"""
    d = np.prod(dims)
    rho = np.asarray(rho, dtype=np.complex128)

    W = cp.Variable((d, d), hermitian=True)
    W_TB_expr = partial_transpose_cvx(W, dims, subsystem=1)  # 用cvxpy版本

    constraints = [
        cp.trace(W) == 1,
        W_TB_expr >> 0  # W^{T_B} ≥ 0
    ]

    prob = cp.Problem(cp.Minimize(cp.real(cp.trace(W @ rho))), constraints)
    prob.solve(solver=cp.SCS)

    print("SDP status:", prob.status)
    if W.value is not None:
        witness_val = np.real(np.trace(W.value @ rho))
        print("Witness value Tr(W ρ):", witness_val)
    else:
        witness_val = None
        print("No valid solution found.")
    return W.value, witness_val

def is_2_extendible(rho_AB, dA, dB):
    d = dA * dB * dB
    # cvxpy variable for rho_AB1B2
    rho_ext = cp.Variable((d, d), hermitian=True)
    constraints = [rho_ext >> 0]
    # partial trace over B2 equals rho_AB
    # (use numpy for index contraction)
    # For cvxpy, use a matrix representation for the partial trace
    # Build the partial trace operator
    # (This is a bit technical; for small dA, dB, can use explicit sum)
    # Here, just check symmetry constraint and trace constraint
    # Symmetry: rho_ext = swap_B1B2(rho_ext)
    swap = np.zeros((d, d))
    for a1 in range(dA):
        for b1 in range(dB):
            for b2 in range(dB):
                for a2 in range(dA):
                    for b1p in range(dB):
                        for b2p in range(dB):
                            i = ((a1*dB + b1)*dB + b2)
                            j = ((a2*dB + b1p)*dB + b2p)
                            i_swap = ((a1*dB + b2)*dB + b1)
                            j_swap = ((a2*dB + b2p)*dB + b1p)
                            swap[i, j] = 1 if (i_swap == i and j_swap == j) else 0
    constraints.append(rho_ext == (rho_ext + swap @ rho_ext @ swap.T) / 2)
    # Partial trace constraint (approximate, for small dA, dB)
    # For each (a1,b1,a2,b2), sum over b2=b2p
    partial_tr = np.zeros((dA*dB, d, d), dtype=complex)
    for a1 in range(dA):
        for b1 in range(dB):
            for a2 in range(dA):
                for b2 in range(dB):
                    row = a1*dB + b1
                    col = a2*dB + b2
                    for b2p in range(dB):
                        i = ((a1*dB + b1)*dB + b2p)
                        j = ((a2*dB + b2)*dB + b2p)
                        partial_tr[row, i, j] += 1
    # Impose: sum_b2 rho_ext = rho_AB
    constraints.append(cp.sum(cp.multiply(partial_tr, rho_ext)) == rho_AB)
    # Trace normalization
    constraints.append(cp.trace(rho_ext) == 1)
    # Solve (SDP formulation)
    prob = cp.Problem(cp.Minimize(0), constraints)
    prob.solve(solver=cp.SCS)
    return prob.status == 'optimal'

def partial_trace_B2(rho, dA, dB):
    # rho: (dA*dB*dB, dA*dB*dB)
    # trace over the last B (B2)
    rho = rho.reshape((dA, dB, dB, dA, dB, dB))
    # trace over axis 2 and 5 (B2)
    return np.einsum('ijklpq->ijlp', rho).reshape((dA*dB, dA*dB))

def symmetrize_B1B2(rho, dA, dB):
    # symmetrize over B1<->B2
    rho = rho.reshape((dA, dB, dB, dA, dB, dB))
    rho_swap = np.swapaxes(rho, 1, 2)
    rho_swap = np.swapaxes(rho_swap, 4, 5)
    rho_sym = (rho + rho_swap) / 2
    return rho_sym.reshape((dA*dB*dB, dA*dB*dB))

def is_3_extendible(rho_AB, dA, dB):
    import cvxpy as cp
    import numpy as np

    d = dA * dB * dB * dB
    rho_ext = cp.Variable((d, d), hermitian=True)
    constraints = [rho_ext >> 0]

    # --- Symmetrize over B1, B2, B3 ---
    def symmetrize_B123(rho, dA, dB):
        # rho: (dA*dB*dB*dB, dA*dB*dB*dB)
        rho = cp.reshape(rho, (dA, dB, dB, dB, dA, dB, dB, dB))
        # swap B1<->B2
        rho12 = cp.transpose(rho, (0,2,1,3,4,6,5,7))
        # swap B1<->B3
        rho13 = cp.transpose(rho, (0,3,2,1,4,7,6,5))
        # swap B2<->B3
        rho23 = cp.transpose(rho, (0,1,3,2,4,5,7,6))
        rho_sym = (rho + rho12 + rho13 + rho23) / 4
        return cp.reshape(rho_sym, (dA*dB*dB*dB, dA*dB*dB*dB))
    constraints.append(rho_ext == symmetrize_B123(rho_ext, dA, dB))

    # --- Partial trace over B2, B3 ---
    def partial_trace_B2B3(rho, dA, dB):
        # rho: (dA*dB*dB*dB, dA*dB*dB*dB)
        rho = cp.reshape(rho, (dA, dB, dB, dB, dA, dB, dB, dB))
        # trace over axis 2,3 and 6,7 (B2, B3)
        # 先 trace B3 (axis 3,7)
        rho = cp.sum(rho, axis=3)
        rho = cp.sum(rho, axis=6)
        # 再 trace B2 (axis 2,5)
        rho = cp.sum(rho, axis=2)
        rho = cp.sum(rho, axis=5)
        # reshape回 (dA*dB, dA*dB)
        return cp.reshape(rho, (dA*dB, dA*dB))
    constraints.append(partial_trace_B2B3(rho_ext, dA, dB) == rho_AB)

    # --- Trace normalization ---
    constraints.append(cp.trace(rho_ext) == 1)

    prob = cp.Problem(cp.Minimize(0), constraints)
    prob.solve(solver=cp.SCS)
    return prob.status == 'optimal'

