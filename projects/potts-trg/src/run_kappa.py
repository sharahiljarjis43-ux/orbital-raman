"""Protocol driver (fixed before running): nsteps=24, central difference in q, no extrapolation.
usage: run_kappa.py <lattice> <p> <delta> <D,list> [--stencil5]"""
import sys, json, time
import numpy as np
from trg_u1 import free_energy
from rc_tensor import square_tensors, triangular_block

EDGES_PER_SITE = {'square': 2, 'triangular': 3}   # medial vertices per lattice site
NSTEPS = 24

def f_of_q(lat, p, q, D, nsteps=NSTEPS):
    """f(q) = ln Z_loop / N_site.  ln Z_RC/N = 0.5*ln q + f(q)."""
    if lat == 'square':
        TA, TB, qc = square_tensors(p, q)
        return 2.0*free_energy(TA, TB, qc, D, nsteps)     # 2 medial vertices per site
    elif lat == 'triangular':
        T, qc = triangular_block(p, q)
        return free_energy(T, T, qc, D, nsteps)           # block already = 1 site (3 medial vertices)
    raise ValueError(lat)

def main():
    lat = sys.argv[1]; p = float(sys.argv[2]); delta = float(sys.argv[3])
    Ds = [int(x) for x in sys.argv[4].split(',')]
    st5 = '--stencil5' in sys.argv
    out = f"kappa_{lat}_d{delta}{'_s5' if st5 else ''}.jsonl"
    with open(out, 'w') as fh:
        for D in Ds:
            t0 = time.time()
            f0 = f_of_q(lat, p, 1.0, D)
            fm = f_of_q(lat, p, 1.0-delta, D); fp = f_of_q(lat, p, 1.0+delta, D)
            rec = {'lattice': lat, 'p': p, 'delta': delta, 'D': D, 'nsteps': NSTEPS,
                   'f1': f0, 'f_minus': fm, 'f_plus': fp}
            rec['kappa3'] = 0.5 + (fp - fm)/(2*delta)
            if st5:
                fm2 = f_of_q(lat, p, 1.0-2*delta, D); fp2 = f_of_q(lat, p, 1.0+2*delta, D)
                rec['f_minus2'] = fm2; rec['f_plus2'] = fp2
                rec['kappa5'] = 0.5 + (-fp2 + 8*fp - 8*fm + fm2)/(12*delta)
            rec['secs'] = round(time.time()-t0, 1)
            fh.write(json.dumps(rec)+'\n'); fh.flush()
            k5 = f"  k5={rec['kappa5']:.6f}" if st5 else ""
            print(f"D={D:3d}  f(1)={f0:+.3e}  kappa3={rec['kappa3']:.6f}{k5}  [{rec['secs']}s]", flush=True)
    print("wrote", out)

if __name__ == '__main__':
    main()
