import numpy as np

def split_matrix(M, Qr, Qc, D):
    sectors = []
    for Q in np.intersect1d(np.unique(Qr), np.unique(Qc)):
        ri = np.flatnonzero(Qr == Q); ci = np.flatnonzero(Qc == Q)
        sub = M[np.ix_(ri, ci)]
        if not np.any(sub): continue
        U, s, Vh = np.linalg.svd(sub, full_matrices=False)
        sectors.append((int(Q), ri, ci, U, s, Vh))
    allS = np.concatenate([sec[4] for sec in sectors])
    allQ = np.concatenate([np.full(len(sec[4]), sec[0]) for sec in sectors])
    keep = np.argsort(-allS, kind="stable")[: min(D, len(allS))]
    nkeep = {sec[0]: 0 for sec in sectors}
    for idx in keep: nkeep[int(allQ[idx])] += 1
    k = sum(nkeep.values())
    L = np.zeros((M.shape[0], k), M.dtype); R = np.zeros((M.shape[1], k), M.dtype); qm = np.zeros(k, int)
    pos = 0
    for (Q, ri, ci, U, s, Vh) in sectors:
        n = nkeep[Q]
        if n == 0: continue
        sq = np.sqrt(s[:n])
        L[ri, pos:pos+n] = U[:, :n]*sq; R[ci, pos:pos+n] = Vh[:n, :].T*sq; qm[pos:pos+n] = Q
        pos += n
    return L, R, qm

def split_A(T, q, D):                      # T ~ S1[u,l,m] S2[d,r,m]
    qu, ql, qd, qr = q; du, dl, dd, dr = T.shape
    L, R, qm = split_matrix(T.reshape(du*dl, dd*dr),
                            (qu[:,None]+ql[None,:]).ravel(), (qd[:,None]+qr[None,:]).ravel(), D)
    return L.reshape(du, dl, -1), R.reshape(dd, dr, -1), qm

def split_B(T, q, D):                      # T ~ S3[l,d,m] S4[r,u,m]
    qu, ql, qd, qr = q; Tt = T.transpose(1,2,3,0); dl, dd, dr, du = Tt.shape
    L, R, qm = split_matrix(Tt.reshape(dl*dd, dr*du),
                            (ql[:,None]-qd[None,:]).ravel(), (qr[:,None]-qu[None,:]).ravel(), D)
    return L.reshape(dl, dd, -1), R.reshape(dr, du, -1), qm

def trg_step(TA, TB, q, D):
    S1, S2, qA = split_A(TA, q, D); S3, S4, qB = split_B(TB, q, D)
    X = np.tensordot(S2, S3, axes=([1],[0]))          # a,m1,c,m2
    Y = np.tensordot(S1, S4, axes=([1],[0]))          # c,m3,a,m4
    Tp = np.tensordot(X, Y, axes=([0,2],[2,0]))       # m1,m2,m3,m4 = NW,NE,SE,SW
    return Tp.transpose(0,3,2,1), (qA, qB, qA, qB)    # rotate: u=NW,l=SW,d=SE,r=NE

def free_energy(TA, TB, q, D, nsteps):
    cA, cB = np.abs(TA).max(), np.abs(TB).max(); TA, TB = TA/cA, TB/cB
    lnZ = 0.5*(np.log(cA)+np.log(cB))
    for k in range(1, nsteps+1):
        T, q = trg_step(TA, TB, q, D); c = np.abs(T).max(); T = T/c
        lnZ += np.log(c)/2**k; TA = TB = T
    tr = np.einsum('ulul->', T)
    if abs(tr) > 1e-300: lnZ += np.log(complex(tr))/2**nsteps
    return np.real(lnZ)


# ---- U(1) sanity: every nonzero element must satisfy qu+ql == qd+qr ----
def check_u1(T, q, tol=1e-13):
    qu, ql, qd, qr = q
    bad = (np.abs(T) > tol) & (
        (qu[:,None,None,None] + ql[None,:,None,None]) !=
        (qd[None,None,:,None] + qr[None,None,None,:]))
    return int(bad.sum()), float(np.abs(T)[bad].max() if bad.any() else 0.0)
