"""Linear-response (impurity / dual-number) TRG: carries T = T0 + eps*T1 with eps^2=0,
so f'(1) comes out with NO finite difference in q at all.  Truncation isometries are
fixed by T0 (standard fixed-isometry impurity scheme); the variation of the *factorization*
is kept exactly in a well-conditioned gauge:
    split  U s V^dag,  M = U^dag T1 V,   A_ij = M_ij sqrt(s_j)/(s_i+s_j),
                                          B_ij = M_ij sqrt(s_i)/(s_i+s_j)
    so that  L0 R1^T + L1 R0^T = P_U T1 P_V   with no s^{-1/2}."""
import numpy as np
from rc_tensor import square_tensors, triangular_block

LOST = [0.0]

def split_matrix_d(M0, M1, Qr, Qc, D):
    secs = []
    for Q in np.intersect1d(np.unique(Qr), np.unique(Qc)):
        ri = np.flatnonzero(Qr == Q); ci = np.flatnonzero(Qc == Q)
        s0 = M0[np.ix_(ri, ci)]; s1 = M1[np.ix_(ri, ci)]
        if not np.any(s0):
            LOST[0] = max(LOST[0], float(np.abs(s1).max()) if s1.size else 0.0); continue
        U, s, Vh = np.linalg.svd(s0, full_matrices=False)
        secs.append((int(Q), ri, ci, U, s, Vh, s1))
    allS = np.concatenate([x[4] for x in secs])
    allQ = np.concatenate([np.full(len(x[4]), x[0]) for x in secs])
    keep = np.argsort(-allS, kind="stable")[: min(D, len(allS))]
    nk = {x[0]: 0 for x in secs}
    for i in keep: nk[int(allQ[i])] += 1
    k = sum(nk.values())
    z = lambda n: np.zeros((n, k), complex)
    L0, L1, R0, R1 = z(M0.shape[0]), z(M0.shape[0]), z(M0.shape[1]), z(M0.shape[1])
    qm = np.zeros(k, int); pos = 0
    for (Q, ri, ci, U, s, Vh, s1) in secs:
        n = nk[Q]
        if n == 0: continue
        Un = U[:, :n]; Vhn = Vh[:n, :]; sn = s[:n]; sq = np.sqrt(sn)
        M = Un.conj().T @ s1 @ Vhn.conj().T                       # n x n
        den = sn[:, None] + sn[None, :]
        A = M * (sq[None, :] / den); B = M * (sq[:, None] / den)
        L0[ri, pos:pos+n] = Un*sq;        L1[ri, pos:pos+n] = Un @ A
        R0[ci, pos:pos+n] = Vhn.T*sq;     R1[ci, pos:pos+n] = (B @ Vhn).T
        qm[pos:pos+n] = Q; pos += n
    return L0, L1, R0, R1, qm

def split_A_d(T0, T1, q, D):
    qu, ql, qd, qr = q; du, dl, dd, dr = T0.shape
    L0,L1,R0,R1,qm = split_matrix_d(T0.reshape(du*dl, dd*dr), T1.reshape(du*dl, dd*dr),
        (qu[:,None]+ql[None,:]).ravel(), (qd[:,None]+qr[None,:]).ravel(), D)
    r = lambda X, a, b: X.reshape(a, b, -1)
    return r(L0,du,dl), r(L1,du,dl), r(R0,dd,dr), r(R1,dd,dr), qm

def split_B_d(T0, T1, q, D):
    qu, ql, qd, qr = q
    A0 = T0.transpose(1,2,3,0); A1 = T1.transpose(1,2,3,0); dl, dd, dr, du = A0.shape
    L0,L1,R0,R1,qm = split_matrix_d(A0.reshape(dl*dd, dr*du), A1.reshape(dl*dd, dr*du),
        (ql[:,None]-qd[None,:]).ravel(), (qr[:,None]-qu[None,:]).ravel(), D)
    r = lambda X, a, b: X.reshape(a, b, -1)
    return r(L0,dl,dd), r(L1,dl,dd), r(R0,dr,du), r(R1,dr,du), qm

def td(a, b, ax): return np.tensordot(a, b, axes=ax)

def step_d(TA0, TA1, TB0, TB1, q, D):
    S1_0,S1_1,S2_0,S2_1,qA = split_A_d(TA0, TA1, q, D)
    S3_0,S3_1,S4_0,S4_1,qB = split_B_d(TB0, TB1, q, D)
    X0 = td(S2_0,S3_0,([1],[0])); X1 = td(S2_1,S3_0,([1],[0])) + td(S2_0,S3_1,([1],[0]))
    Y0 = td(S1_0,S4_0,([1],[0])); Y1 = td(S1_1,S4_0,([1],[0])) + td(S1_0,S4_1,([1],[0]))
    P0 = td(X0,Y0,([0,2],[2,0])); P1 = td(X1,Y0,([0,2],[2,0])) + td(X0,Y1,([0,2],[2,0]))
    t = lambda Z: Z.transpose(0,3,2,1)
    return t(P0), t(P1), (qA, qB, qA, qB)

def lnZ_and_deriv(TA0, TA1, TB0, TB1, q, D, nsteps):
    cA, cB = np.abs(TA0).max(), np.abs(TB0).max()
    TA0, TA1, TB0, TB1 = TA0/cA, TA1/cA, TB0/cB, TB1/cB
    lnZ0 = 0.5*(np.log(cA)+np.log(cB)); lnZ1 = 0.0
    for k in range(1, nsteps+1):
        T0, T1, q = step_d(TA0, TA1, TB0, TB1, q, D)
        c = np.abs(T0).max(); T0, T1 = T0/c, T1/c
        lnZ0 += np.log(c)/2**k
        TA0 = TB0 = T0; TA1 = TB1 = T1
    tr0 = np.einsum('ulul->', T0); tr1 = np.einsum('ulul->', T1)
    lnZ0 += np.log(complex(tr0))/2**nsteps
    lnZ1 += (tr1/tr0)/2**nsteps
    return np.real(lnZ0), np.real(lnZ1)

def dtensor(builder, p, h=1e-6):
    """T0 and dT/dq at q=1 by central difference on the *tensor entries* (analytic, smooth:
    error ~1e-10, ten thousand times below the truncation error) -- no amplification here."""
    out0 = builder(p, 1.0); out_p = builder(p, 1.0+h); out_m = builder(p, 1.0-h)
    return out0, out_p, out_m

def kappa_lr(lat, p, D, nsteps=24, h=1e-6):
    LOST[0] = 0.0
    if lat == 'square':
        (A0,B0,q),(Ap,Bp,_),(Am,Bm,_) = dtensor(square_tensors, p, h)
        A1 = (Ap-Am)/(2*h); B1 = (Bp-Bm)/(2*h)
        z0, z1 = lnZ_and_deriv(A0, A1, B0, B1, q, D, nsteps)
        return 2*z0, 2*z1
    else:
        (T0,q),(Tp,_),(Tm,_) = dtensor(triangular_block, p, h)
        T1 = (Tp-Tm)/(2*h)
        return lnZ_and_deriv(T0, T1, T0, T1, q, D, nsteps)

if __name__ == '__main__':
    import sys
    lat = sys.argv[1]; p = float(sys.argv[2]); Ds = [int(x) for x in sys.argv[3].split(',')]
    print(f"linear-response kappa, {lat}, p={p}, nsteps=24 (no finite difference in q)")
    for D in Ds:
        f0, f1 = kappa_lr(lat, p, D)
        print(f"  D={D:3d}  f(1)={f0:+.3e}  kappa_LR={0.5+f1:.6f}  (dropped-sector max {LOST[0]:.1e})", flush=True)
