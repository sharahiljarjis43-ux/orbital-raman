import numpy as np

def onsager(K, n=200000):
    """square Ising ln Z / N, Baxter single-integral form (independent of the double-integral form)."""
    c = np.cosh(2*K); s = np.sinh(2*K); k = 2*s/c**2
    phi = (np.arange(n)+0.5)*np.pi/n
    integ = np.log(0.5*(1+np.sqrt(1-(k*np.sin(phi))**2)))
    return np.log(2*c) + integ.mean()/2

def houtappel(K, n=3000):
    """triangular Ising ln Z / N."""
    c = np.cosh(2*K); s = np.sinh(2*K)
    w = 2*np.pi*(np.arange(n)+0.5)/n
    W1, W2 = np.meshgrid(w, w, indexing='ij')
    br = c**3 + s**3 - s*(np.cos(W1)+np.cos(W2)+np.cos(W1+W2))
    return np.log(2) + 0.5*np.log(br).mean()

# --- independent validation of both closed forms: high-temperature series ---
# ln Z/N = ln 2 + (E/N) ln cosh K + sum over closed subgraphs t^|A| / N
def series_square(K):    # 4-cycles: 1/site (t^4); 6-cycles: 2/site (t^6)
    t = np.tanh(K); return np.log(2) + 2*np.log(np.cosh(K)) + t**4 + 2*t**6
def series_tri(K):       # triangles: 2/site (t^3); rhombi (4-cycles): 3/site (t^4)
    t = np.tanh(K); return np.log(2) + 3*np.log(np.cosh(K)) + 2*t**3 + 3*t**4

if __name__ == '__main__':
    print("closed-form vs high-T series (validates normalization + bracket):")
    for K in (0.05, 0.10):
        t = np.tanh(K)
        print(f"  K={K}: square  exact={onsager(K):.10f} series={series_square(K):.10f} "
              f"diff={onsager(K)-series_square(K):+.2e}  (next order ~{t**8:.1e})")
        print(f"  K={K}: tri     exact={houtappel(K):.10f} series={series_tri(K):.10f} "
              f"diff={houtappel(K)-series_tri(K):+.2e}  (next order ~{6*t**5:.1e})")
    Kc_sq = 0.5*np.log(1+np.sqrt(2)); Kc_tri = 0.5*np.arcsinh(1/np.sqrt(3))
    print(f"  square K_c={Kc_sq:.10f} onsager={onsager(Kc_sq):.10f}")
    print(f"  tri    K_c={Kc_tri:.10f} houtappel={houtappel(Kc_tri):.10f}")
