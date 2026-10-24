"""Historical square-lattice self-dual-path comparison.

The unified critline.py driver also supports the triangular critical line.
This module is safe to import; running its command performs TRG calculations.
"""
import sys, json
import numpy as np
from run_kappa import f_of_q

def p_sd(q): r = np.sqrt(q); return r/(1+r)

def main():
    delta = float(sys.argv[1]); Ds = [int(x) for x in sys.argv[2].split(',')]
    print(f"self-dual line, delta={delta}, 5-point stencil")
    for D in Ds:
        g = {}
        for m in (-2,-1,0,1,2):
            q = 1.0 + m*delta
            g[m] = f_of_q('square', p_sd(q), q, D)
        k3 = 0.5 + (g[1]-g[-1])/(2*delta)
        k5 = 0.5 + (-g[2] + 8*g[1] - 8*g[-1] + g[-2])/(12*delta)
        print(f"  D={D:3d}  f(1)={g[0]:+.3e}  kappa3={k3:.6f}  kappa5={k5:.6f}", flush=True)

if __name__ == '__main__':
    main()
