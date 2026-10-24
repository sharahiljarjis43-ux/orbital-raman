import numpy as np
SQ3 = np.sqrt(3.0)
def wrap(a): return (a+np.pi) % (2*np.pi) - np.pi

def vertex_tensor(dirs, t, is_head, lam, w_open, w_closed):
    dirs = np.asarray(dirs, float); t = np.asarray(t, float)
    cross = t[0]*dirs[:,1] - t[1]*dirs[:,0]; dot = dirs @ t
    def pairs_from(sign):
        pos = [i for i in range(4) if sign[i] > 0]; neg = [i for i in range(4) if sign[i] < 0]
        assert len(pos) == 2 and len(neg) == 2; return (tuple(pos), tuple(neg))
    open_pairs, closed_pairs = pairs_from(cross), pairs_from(dot)
    ang = np.arctan2(dirs[:,1], dirs[:,0])
    turn = lambda a, b: wrap(ang[b] - (ang[a] + np.pi))
    sig_in = [1 if h else 0 for h in is_head]; sig_out = [0 if h else 1 for h in is_head]
    T = np.zeros((2,2,2,2), complex)
    for pairs, w in ((open_pairs, w_open), (closed_pairs, w_closed)):
        (a,b),(c,d) = pairs
        for (i1,o1) in ((a,b),(b,a)):
            for (i2,o2) in ((c,d),(d,c)):
                s = [0]*4; s[i1]=sig_in[i1]; s[o1]=sig_out[o1]; s[i2]=sig_in[i2]; s[o2]=sig_out[o2]
                T[tuple(s)] += w*np.exp(1j*lam*(turn(i1,o1)+turn(i2,o2))/(2*np.pi))
    return T

def loop_params(p, q):
    n = np.sqrt(q); return np.arccos(n/2), p/n, 1-p

def square_tensors(p, q):          # legs (u,l,d,r) = original (NW,SW,SE,NE); f from TRG is per medial vertex (x2 per site)
    lam, wo, wc = loop_params(p, q); dirs = [(-1,1),(-1,-1),(1,-1),(1,1)]; ih = [True,True,False,False]
    q2 = np.array([0,1])
    return vertex_tensor(dirs,(1,0),ih,lam,wo,wc), vertex_tensor(dirs,(0,1),ih,lam,wo,wc), (q2,q2,q2,q2)

def triangular_block(p, q):        # cell owns B (bottom edge), L (left edge), Dg (diagonal) midpoints; P(x,y)=(x-y/2, y*sqrt3/2)
    lam, wo, wc = loop_params(p, q)
    P = lambda x, y: np.array([x - y/2, y*SQ3/2])
    B, L, Dg = (P(0,0)+P(1,0))/2, (P(0,0)+P(0,1))/2, (P(0,0)+P(1,1))/2
    E, N = P(1,0), P(0,1)
    L_E, B_N, B_W, Dg_W, L_S, Dg_S = L+E, B+N, B-E, Dg-E, L-N, Dg-N
    TBv = vertex_tensor([L_E-B, Dg-B, L_S-B, Dg_S-B], P(1,0)-P(0,0), [False,True,True,False], lam, wo, wc)   # e1,e3,e4',e5'
    TLv = vertex_tensor([B_N-L, Dg-L, B_W-L, Dg_W-L], P(0,1)-P(0,0), [False,True,True,False], lam, wo, wc)   # e4,e6,e1',e2'
    TDv = vertex_tensor([L_E-Dg, B-Dg, B_N-Dg, L-Dg], P(1,1)-P(0,0), [True,False,True,False], lam, wo, wc)   # e2,e3,e5,e6
    T = np.einsum('acgh,dfij,bcef->deijghab', TBv, TLv, TDv).reshape(4,4,4,4)   # u=(e4,e5) l=(e1',e2') d=(e4',e5') r=(e1,e2)
    s = np.array([[0,0],[0,1],[1,0],[1,1]])
    q_u = (1-s[:,0]) + s[:,1]; q_l = s[:,0] + (1-s[:,1]); q_d = q_u.copy(); q_r = q_l.copy()
    return T, (q_u, q_l, q_d, q_r)
