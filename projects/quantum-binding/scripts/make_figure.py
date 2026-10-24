"""Plot covariance aliasing and its ideal Ramsey witness; synthetic data only."""
from pathlib import Path
import argparse
import numpy as np
from scipy.linalg import expm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import forward_model as fm


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    t = np.linspace(0, 8, 180)
    fig, ax = plt.subplots(1, 2, figsize=(10.4, 3.8), constrained_layout=True)
    colors = ('#145A7A', '#BD5038')
    for h, label in enumerate(('Base network', 'Covariance alias')):
        q, pi = fm.generator(fm.TRUTH, h)
        cov = np.array([(pi*fm.F) @ expm(q*v) @ fm.F for v in t])
        rows = [fm.Setting('curve', 'science', 'Ramsey', float(v), .5, 0.) for v in t]
        w = fm.coherences(fm.TRUTH, h, rows, method='expm')
        style = '-' if h == 0 else '--'
        ax[0].plot(t, cov, style, color=colors[h], linewidth=2.2, label=label)
        ax[1].plot(t, w.imag, style, color=colors[h], linewidth=2.2, label=label)
    ax[0].set(title='Identical ordinary covariance', ylabel='C(t)', xlabel='Dimensionless lag t')
    ax[1].set(title='Different ideal Ramsey coherence', ylabel='Im W(T), g = 0.5', xlabel='Dimensionless duration T')
    for a in ax:
        a.spines[['top', 'right']].set_visible(False)
        a.grid(alpha=.15)
        a.legend(frameon=False, fontsize=9)
    fig.suptitle('One fixed three-partner model pair', fontsize=13)
    fig.savefig(args.output, dpi=170, facecolor='white')
    plt.close(fig)
    print(f'Wrote {args.output}')


if __name__ == '__main__':
    main()
