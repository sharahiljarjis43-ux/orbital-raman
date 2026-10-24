"""Critical-line finite-difference prototype for mean cluster density.

The exact normalized random-cluster model obeys f(p, 1) == 0, so moving
along any smooth p(q) through the target point preserves the exact derivative.
Choosing the critical line changes numerical truncation error; it does not
make a finite-D estimate exact or certify convergence. Both lattice critical
lines are checked at q=1 and q=2 below.
"""
import sys
import numpy as np
from run_kappa import f_of_q

def v_square(q): return np.sqrt(q)

def v_tri(q):
    r = np.roots([1.0, 3.0, 0.0, -q])
    r = [x.real for x in r if abs(x.imag) < 1e-12 and x.real > 0]
    assert len(r) == 1, r
    return r[0]

VC = {'square': v_square, 'triangular': v_tri}
def p_c(lat, q): v = VC[lat](q); return v/(1+v)

def verify_lines():
    ok = True
    # q=1: bond percolation thresholds (square 1/2 exact; triangular 2 sin(pi/18) exact)
    for lat, ref in (('square', 0.5), ('triangular', 2*np.sin(np.pi/18))):
        got = p_c(lat, 1.0); ok &= abs(got-ref) < 1e-12
        print(f"  {lat:11s} q=1: p_c={got:.12f} ref={ref:.12f} diff={got-ref:+.1e}")
    # q=2: v = e^{2K_Ising}-1 at the exact Ising critical couplings
    Kc_sq = 0.5*np.log(1+np.sqrt(2)); Kc_tri = 0.5*np.arcsinh(1/np.sqrt(3))
    for lat, K in (('square', Kc_sq), ('triangular', Kc_tri)):
        ref = (np.exp(2*K)-1)/(np.exp(2*K)); ref = 1-np.exp(-2*K)
        got = p_c(lat, 2.0); ok &= abs(got-ref) < 1e-12
        print(f"  {lat:11s} q=2: p_c={got:.12f} ref={ref:.12f} diff={got-ref:+.1e}")
    return ok

if __name__ == '__main__':
    print("critical lines vs independently known critical points:")
    assert verify_lines(), "critical line failed verification"
    lat = sys.argv[1]; delta = float(sys.argv[2]); Ds = [int(x) for x in sys.argv[3].split(',')]
    print(f"\ncritical-line kappa, {lat}, delta={delta}, nsteps=24")
    for D in Ds:
        g = {m: f_of_q(lat, p_c(lat, 1.0+m*delta), 1.0+m*delta, D) for m in (-2,-1,0,1,2)}
        k3 = 0.5 + (g[1]-g[-1])/(2*delta)
        k5 = 0.5 + (-g[2] + 8*g[1] - 8*g[-1] + g[-2])/(12*delta)
        print(f"  D={D:3d}  f(1)={g[0]:+.3e}  kappa3={k3:.7f}  kappa5={k5:.7f}", flush=True)
