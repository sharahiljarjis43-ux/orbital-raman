"""
ORBITAL RAMAN FEASIBILITY STUDY
===============================
WITH REALISTIC SPACE ENVIRONMENT + EXTENDED ANALYSIS

Includes:
  - Surface incidence angle, shadow mapping, roughness, dust
  - Cosmic ray noise
  - Monte Carlo uncertainty quantification (10,000 trials)
  - Quantum squeezing analysis (analytical + QuTiP verification)
  - g^(2) photon correlation analysis

Usage: python src.py
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, LinearSegmentedColormap
from scipy import stats
import struct
import os
import warnings
# Keep runtime output readable without suppressing all warnings globally.

try:
    import qutip as qt
    QUTIP = True
except ImportError:
    QUTIP = False

# =============================================================================
# DATA LOADERS
# =============================================================================

def load_ovirs(filepath):
    print(f"  Loading OVIRS...")
    with open(filepath, 'rb') as f:
        f.seek(8640)
        raw = f.read(786432 * 24)
    lat, lon, val = [], [], []
    for i in range(786432):
        rec = raw[i*24:(i+1)*24]
        if len(rec) < 24: break
        lat.append(struct.unpack('>f', rec[4:8])[0])
        lon.append(struct.unpack('>f', rec[8:12])[0])
        val.append(struct.unpack('>f', rec[16:20])[0])
    lat, lon, val = np.array(lat), np.array(lon), np.array(val)
    valid = val > -9990
    organic_pct = np.clip(val[valid] * 15, 0, 50)
    print(f"    ✓ {valid.sum()} points, {organic_pct.mean():.1f}% mean organic")
    return {'lat': lat[valid], 'lon': lon[valid], 'organic_pct': organic_pct}

def load_otes(filepath):
    print(f"  Loading OTES...")
    with open(filepath, 'rb') as f:
        f.seek(62304)
        data = f.read(7264 * 4)
    temps = np.array(struct.unpack('>7264f', data))
    valid = (temps > 100) & (temps < 500)
    mean_T = temps[valid].mean() if valid.any() else 300.0
    print(f"    ✓ Temperature: {mean_T:.1f} K")
    return mean_T

def load_tsis1(filepath):
    print(f"  Loading TSIS-1...")
    try:
        import netCDF4 as nc
        ds = nc.Dataset(filepath)
        wl = ds.variables['Wavelength'][:]
        ssi = ds.variables['SSI'][:]
        if len(ssi.shape) > 1: ssi = np.nanmean(ssi, axis=0)
        ds.close()
        irr = float(ssi[np.argmin(np.abs(wl - 280))])
        print(f"    ✓ Irradiance @ 280nm: {irr:.4f} W/m²/nm")
        return irr
    except Exception as e:
        print(f"    TSIS-1 load failed: {e}")
        print(f"    Using default 0.015 W/m²/nm")
        return 0.015

def make_synthetic(seed=42):
    rng = np.random.default_rng(seed)
    lat, lon = np.meshgrid(np.linspace(-90,90,180), np.linspace(0,360,360), indexing='ij')
    organic = np.clip(5 + 3*rng.standard_normal(lat.shape), 1, 15)
    return {'lat': lat.flatten(), 'lon': lon.flatten(), 'organic_pct': organic.flatten()}


# =============================================================================
# SPACE ENVIRONMENT MODEL
# =============================================================================

class SpaceEnvironment:
    """Simulates realistic space environment effects."""
    
    def __init__(self, lat, lon, seed=42):
        self.rng = np.random.default_rng(seed)
        self.n = len(lat)
        self.lat = lat
        self.lon = lon
        
        # Incidence angle - but Bennu rotates, so we see different faces
        # Model as random distribution centered on 45° (average illumination)
        base_inc = 45 + 25 * self.rng.standard_normal(self.n)
        self.incidence = np.clip(base_inc, 0, 85)
        
        # Roughness factor (Bennu is rough but not catastrophically so)
        # Literature: ~10-30% signal reduction from roughness
        self.roughness = np.clip(0.8 + 0.1 * self.rng.standard_normal(self.n), 0.6, 1.0)
        
        # Shadow mask - only ~10% in deep shadow at any time
        self.illuminated = self.rng.random(self.n) > 0.10
        
    def get_factor(self, mission_day=100):
        """Combined environment factor (0-1)."""
        # Incidence: cos(45°) ≈ 0.7 average
        inc_factor = np.cos(np.radians(self.incidence))
        
        # Dust: ~1% loss per year = negligible for 100 days
        dust = 0.997  # ~0.3% loss at day 100
        
        # Combine: typical factor ~0.5-0.7 (not 0!)
        # Shadow sets some to 0, but most are illuminated
        factor = inc_factor * self.roughness * self.illuminated.astype(float) * dust
        
        # Floor at 0.1 for non-shadowed points (some signal always gets through)
        factor = np.where(self.illuminated, np.maximum(factor, 0.1), 0)
        
        return factor
    
    def summary(self):
        factors = self.get_factor()
        print(f"  Environment: {100*self.illuminated.mean():.1f}% illuminated, "
              f"mean factor {factors.mean():.2f}")


# =============================================================================
# PHYSICS MODEL
# =============================================================================

def calculate_snr(distance_m, organic_pct, wavelength_nm=248,
                  pulse_energy_J=0.050, aperture_m=0.30, integration_time_s=10,
                  temp_K=300, solar_irr=0.015, env_factor=1.0):
    """Calculate SNR with optional environment factor."""
    h, c = 6.626e-34, 3e8
    
    rep_rate, QE = 20, 0.30
    n_pulses = rep_rate * integration_time_s
    E_photon = h * c / (wavelength_nm * 1e-9)
    photons_per_pulse = pulse_energy_J / E_photon
    
    # K_R from organics
    K_R_1pct_532 = 1e-11
    lambda_factor = (532 / wavelength_nm) ** 4
    resonance = 10 if wavelength_nm < 260 else (5 if wavelength_nm < 300 else 1)
    K_R = K_R_1pct_532 * (organic_pct / 1.0) * lambda_factor * resonance * env_factor
    
    # Geometry
    A_tel = np.pi * (aperture_m / 2) ** 2
    solid_angle = A_tel / (distance_m ** 2)
    spot_area = np.pi * (distance_m * 0.5e-3) ** 2
    
    # Signal
    N_signal = (photons_per_pulse / spot_area) * K_R * solid_angle * spot_area * n_pulses * QE
    
    # Noise
    gate = n_pulses * 20e-9
    N_solar = solar_irr / 1.27 * 0.5 * A_tel * 0.044 * gate * QE / (h * c / 280e-9) * 0.1
    N_dark = 100 * np.exp((temp_K - 300) / 30) * integration_time_s
    N_read = 25 * n_pulses
    
    if N_signal <= 0: return 0.0
    return N_signal / np.sqrt(N_signal + N_solar + N_dark + N_read)


def find_min_organic(distance, snr_target=5, **kw):
    """Find minimum organic % needed for detection via binary search."""
    lo, hi = 0.0, 25.0
    if calculate_snr(distance, hi, **kw) < snr_target:
        return hi
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if calculate_snr(distance, mid, **kw) >= snr_target:
            hi = mid
        else:
            lo = mid
    return hi


# =============================================================================
# EXTENDED ANALYSIS: Helper functions
# =============================================================================

def calculate_snr_full(distance_m, organic_pct, **kwargs):
    """Returns full breakdown dict, not just SNR scalar."""
    h, c = 6.626e-34, 3e8
    wl = kwargs.get('wavelength_nm', 248)
    pe = kwargs.get('pulse_energy_J', 0.050)
    ap = kwargs.get('aperture_m', 0.30)
    ti = kwargs.get('integration_time_s', 10)
    tK = kwargs.get('temp_K', 300)
    si = kwargs.get('solar_irr', 0.015)
    ef = kwargs.get('env_factor', 1.0)
    rep_rate, QE = 20, 0.30
    n_pulses = rep_rate * ti
    E_photon = h * c / (wl * 1e-9)
    ppp = pe / E_photon
    K_R_1pct_532 = 1e-11
    lf = (532 / wl) ** 4
    res = 10 if wl < 260 else (5 if wl < 300 else 1)
    K_R = K_R_1pct_532 * (organic_pct / 1.0) * lf * res * ef
    A_tel = np.pi * (ap / 2) ** 2
    omega = A_tel / (distance_m ** 2)
    spot = np.pi * (distance_m * 0.5e-3) ** 2
    N_sig = (ppp / spot) * K_R * omega * spot * n_pulses * QE
    gate = n_pulses * 20e-9
    N_solar = si / 1.27 * 0.5 * A_tel * 0.044 * gate * QE / (h * c / 280e-9) * 0.1
    N_dark = 100 * np.exp((tK - 300) / 30) * ti
    N_read = 25 * n_pulses
    N_bg = N_solar + N_dark + N_read
    noise = np.sqrt(max(N_sig + N_bg, 1))
    snr = N_sig / noise if noise > 0 else 0
    return {'N_sig': N_sig, 'N_solar': N_solar, 'N_dark': N_dark,
            'N_read': N_read, 'N_bg': N_bg, 'snr': snr,
            'sig_rate': N_sig / ti, 'bg_rate': N_bg / ti}


def effective_squeezing_dB(eta, ideal_dB):
    """V_eff = eta * V_ideal + (1-eta). Returns effective dB."""
    V_ideal = 10 ** (-ideal_dB / 10)
    V_eff = eta * V_ideal + (1 - eta)
    return -10 * np.log10(V_eff)


def snr_with_squeezing(N_sig, N_bg, eta_total, squeeze_dB):
    """SNR with squeezed vacuum injection through lossy channel."""
    eff_dB = effective_squeezing_dB(eta_total, squeeze_dB)
    noise_factor = 10 ** (-eff_dB / 10)
    modified_noise = np.sqrt(N_sig * noise_factor + N_bg)
    return N_sig / modified_noise if modified_noise > 0 else 0


MC_RANGES = {
    'pulse_energy_J':     (0.030, 0.070),
    'aperture_m':         (0.25, 0.35),
    'integration_time_s': (8, 12),
    'temp_K':             (280, 350),
    'solar_irr':          (0.010, 0.020),
    'organic_pct':        (1.0, 15.0),
    'env_factor':         (0.3, 1.0),
}


def run_monte_carlo(N=10000, distance_m=680, seed=42):
    """Monte Carlo UQ: sample all uncertain params, propagate through photon budget."""
    rng = np.random.default_rng(seed)
    out = {'snr': np.zeros(N), 'alt': np.zeros(N),
           'N_sig': np.zeros(N), 'N_bg': np.zeros(N),
           'sampled': {k: np.zeros(N) for k in MC_RANGES}, 'N': N, 'distance': distance_m}
    for i in range(N):
        params = {}
        for k, (lo, hi) in MC_RANGES.items():
            v = rng.uniform(lo, hi); params[k] = v; out['sampled'][k][i] = v
        org = params.pop('organic_pct')
        ef = params.pop('env_factor'); params['env_factor'] = ef
        r = calculate_snr_full(distance_m, org, **params)
        out['snr'][i] = r['snr']; out['N_sig'][i] = r['N_sig']; out['N_bg'][i] = r['N_bg']
        for d in np.linspace(5000, 50, 200):
            if calculate_snr(d, org, **params) >= 5:
                out['alt'][i] = d; break
        else:
            out['alt'][i] = 50
    return out


def g2_func(source, tau, tau_c=1.0, n_mean=10):
    """g^(2)(tau) for different light sources."""
    if source == 'coherent': return np.ones_like(tau)
    elif source == 'thermal': return 1 + np.exp(-2 * np.abs(tau) / tau_c)
    elif source == 'raman': return 1 + np.exp(-2 * np.abs(tau) / (tau_c * 0.1))
    elif source == 'mixed':
        rho = n_mean / (n_mean + 100)
        g2_r = 1 + np.exp(-2 * np.abs(tau) / (tau_c * 0.1))
        g2_b = 1 + np.exp(-2 * np.abs(tau) / tau_c)
        return rho**2 * g2_r + (1 - rho)**2 * g2_b + 2 * rho * (1 - rho)
    return np.ones_like(tau)


# =============================================================================
# EXTENDED FIGURES: fig6 (MC), fig7 (Squeezing), fig8 (g²)
# =============================================================================

def fig6_monte_carlo():
    """Fig 6: Monte Carlo uncertainty quantification."""
    print("Fig 6: Monte Carlo UQ...")
    mc = run_monte_carlo(N=10000, distance_m=680)
    snr, alt = mc['snr'], mc['alt']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    
    # (a) SNR distribution
    ax = axes[0, 0]
    clip = snr[snr < np.percentile(snr, 99)]
    ax.hist(clip, bins=80, density=True, color='#2196F3', alpha=0.7, edgecolor='white', lw=0.5)
    med = np.median(snr); ci = np.percentile(snr, [2.5, 97.5])
    ax.axvline(med, color='#D32F2F', lw=2, label=f'Median = {med:.1f}')
    ax.axvline(ci[0], color='#D32F2F', lw=1.5, ls='--', label=f'95% CI = [{ci[0]:.1f}, {ci[1]:.1f}]')
    ax.axvline(ci[1], color='#D32F2F', lw=1.5, ls='--')
    ax.axvline(5, color='#4CAF50', lw=2, ls=':', label='SNR = 5 threshold')
    pct = np.mean(snr >= 5) * 100
    ax.set_xlabel('Signal-to-Noise Ratio'); ax.set_ylabel('Probability Density')
    ax.set_title(f'(a) SNR Distribution at 680m (Orbital B)\n{pct:.1f}% above detection threshold')
    ax.legend(fontsize=9)
    
    # (b) Max detection altitude
    ax = axes[0, 1]
    ax.hist(alt, bins=80, density=True, color='#FF9800', alpha=0.7, edgecolor='white', lw=0.5)
    ma = np.median(alt); ca = np.percentile(alt, [2.5, 97.5])
    ax.axvline(ma, color='#D32F2F', lw=2, label=f'Median = {ma:.0f} m')
    ax.axvline(ca[0], color='#D32F2F', lw=1.5, ls='--', label=f'95% CI = [{ca[0]:.0f}, {ca[1]:.0f}] m')
    ax.axvline(ca[1], color='#D32F2F', lw=1.5, ls='--')
    ax.axvline(375, color='cyan', lw=1.5, ls=':', alpha=0.8, label='Recon (375m)')
    ax.axvline(680, color='orange', lw=1.5, ls=':', alpha=0.8, label='Orbital B (680m)')
    ax.set_xlabel('Maximum Detection Altitude [m]'); ax.set_ylabel('Probability Density')
    ax.set_title('(b) Detection Range Distribution (SNR ≥ 5)'); ax.legend(fontsize=8)
    
    # (c) Sensitivity tornado
    ax = axes[1, 0]
    labels = {'pulse_energy_J': 'Pulse Energy', 'aperture_m': 'Aperture',
              'integration_time_s': 'Integration Time', 'temp_K': 'Surface Temp',
              'solar_irr': 'Solar Irradiance', 'organic_pct': 'Organic Concentration',
              'env_factor': 'Environment Factor'}
    corrs = {}
    for k, label in labels.items():
        if k in mc['sampled']:
            r, _ = stats.spearmanr(mc['sampled'][k], mc['snr']); corrs[label] = r
    sc = sorted(corrs.items(), key=lambda x: abs(x[1]), reverse=True)
    labs, vals = [x[0] for x in sc], [x[1] for x in sc]
    colors = ['#4CAF50' if v > 0 else '#F44336' for v in vals]
    ax.barh(range(len(labs)), vals, color=colors, alpha=0.8, edgecolor='white')
    ax.set_yticks(range(len(labs))); ax.set_yticklabels(labs, fontsize=10)
    ax.set_xlabel('Spearman ρ with SNR')
    ax.set_title('(c) Sensitivity Analysis\n(which parameters matter most?)'); ax.axvline(0, color='black', lw=0.8)
    
    # (d) Signal vs background
    ax = axes[1, 1]
    mask = mc['N_sig'] > 0
    if any(mask):
        sc2 = ax.scatter(mc['N_bg'][mask], mc['N_sig'][mask], c=mc['snr'][mask],
                         cmap='RdYlGn', s=3, alpha=0.5, vmin=0,
                         vmax=max(10, np.percentile(mc['snr'], 95)))
        plt.colorbar(sc2, ax=ax, label='SNR'); ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('Background Photons'); ax.set_ylabel('Signal Photons')
    ax.set_title('(d) Signal vs Background (color = SNR)')
    
    plt.tight_layout()
    plt.savefig('Outputs/figures/fig6_monte_carlo.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved")
    return mc


def fig7_squeezing():
    """Fig 7: Quantum squeezing analysis."""
    print("Fig 7: Squeezing Analysis...")
    r = calculate_snr_full(680, 5.0)
    N_sig, N_bg = r['N_sig'], r['N_bg']
    baseline_snr = r['snr']
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # (a) Effective squeezing vs optical efficiency
    ax = axes[0]
    eta = np.linspace(0.01, 0.80, 200)
    for sq_dB, color in [(3, '#2196F3'), (6, '#FF9800'), (10, '#4CAF50')]:
        eff = [effective_squeezing_dB(e, sq_dB) for e in eta]
        ax.plot(eta * 100, eff, lw=2, color=color, label=f'{sq_dB} dB ideal')
    
    if QUTIP:
        N_fock = 40
        eta_test = [0.045, 0.10, 0.20, 0.30, 0.50, 0.70]
        qt_eff = []
        for e in eta_test:
            r_param = 10 / (20 * np.log10(np.e))
            psi = qt.squeeze(N_fock, r_param) * qt.basis(N_fock, 0); psi = psi.unit()
            x = (qt.destroy(N_fock) + qt.create(N_fock)) / np.sqrt(2)
            var_in = qt.expect(x * x, psi) - qt.expect(x, psi) ** 2
            var_out = e * var_in + (1 - e) * 0.5
            eff_dB_qt = -10 * np.log10(var_out / 0.5) if var_out > 0 else 0
            qt_eff.append(max(eff_dB_qt, 0))
        ax.plot([e * 100 for e in eta_test], qt_eff, 'ko', ms=6, alpha=0.7,
                label='QuTiP verification', zorder=5)
    
    ax.axvline(4.5, color='red', ls=':', lw=2, label='Current (~4.5%)')
    ax.axvline(30, color='green', ls=':', lw=2, label='Target (~30%)')
    ax.set_xlabel('Total Optical Efficiency η [%]'); ax.set_ylabel('Effective Squeezing [dB]')
    ax.set_title('(a) Quantum Squeezing vs Optical Efficiency\n'
                 r'$V_{eff} = \eta \cdot V_{ideal} + (1-\eta) \cdot 1$')
    ax.legend(fontsize=9, loc='upper left'); ax.set_xlim([0, 80]); ax.set_ylim([0, 10])
    
    # (b) SNR improvement vs efficiency
    ax = axes[1]
    eta_sweep = np.linspace(0.01, 0.80, 200)
    for sq_dB, color, ls in [(3, '#2196F3', '--'), (6, '#FF9800', '--'), (10, '#4CAF50', '-')]:
        improvement = []
        for e in eta_sweep:
            snr_0 = snr_with_squeezing(N_sig, N_bg, e, 0)
            snr_sq = snr_with_squeezing(N_sig, N_bg, e, sq_dB)
            improvement.append(snr_sq / snr_0 if snr_0 > 0 else 1)
        ax.plot(eta_sweep * 100, improvement, color=color, lw=2, ls=ls, label=f'{sq_dB} dB ideal')
    
    ax.axhline(1.0, color='gray', ls=':', alpha=0.5)
    ax.axvline(4.5, color='red', ls=':', lw=2, label='Current (~4.5%)')
    ax.axvline(30, color='green', ls=':', lw=2, label='Target (~30%)')
    ax.set_xlabel('Total Optical Efficiency η [%]'); ax.set_ylabel('SNR Improvement Factor')
    ax.set_title(f'(b) SNR Gain from Squeezing at 680m\n(baseline SNR = {baseline_snr:.1f} at 5% organics)')
    ax.legend(fontsize=9); ax.set_xlim([0, 80]); ax.set_ylim([0.95, 2.5])
    
    plt.tight_layout()
    plt.savefig('Outputs/figures/fig7_squeezing.png', dpi=200, bbox_inches='tight')
    plt.close()
    
    imp = effective_squeezing_dB(0.045, 10)
    print(f"  At η=4.5%, 10dB ideal → {imp:.2f} dB effective")
    print("  ✓ Saved")


def fig8_g2():
    """Fig 8: g^(2) photon correlation analysis."""
    print("Fig 8: g² Correlation...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    tau = np.linspace(-5, 5, 1000)
    
    # (a) Source signatures
    ax = axes[0]
    ax.plot(tau, g2_func('coherent', tau), 'b-', lw=2, label='Coherent (laser)')
    ax.plot(tau, g2_func('thermal', tau), 'r-', lw=2, label='Thermal (solar/surface)')
    ax.plot(tau, g2_func('raman', tau), 'g-', lw=2, label='Raman (spontaneous)')
    ax.axhline(1, color='gray', ls=':', alpha=0.5)
    ax.set_xlabel('Delay τ [ns]'); ax.set_ylabel('g⁽²⁾(τ)')
    ax.set_title('(a) Photon Correlation Signatures')
    ax.legend(fontsize=9); ax.set_ylim(0.8, 2.2)
    ax.annotate('Bunching\n(photon pairs)', xy=(0, 2.0), fontsize=9, ha='center', color='red', alpha=0.7)
    ax.annotate('Poissonian\n(random)', xy=(3.5, 1.05), fontsize=9, ha='center', color='blue', alpha=0.7)
    
    # (b) Mixed signal
    ax = axes[1]
    fracs = [0.01, 0.05, 0.10, 0.25, 0.50]
    cols = plt.cm.plasma(np.linspace(0.1, 0.9, len(fracs)))
    for i, f in enumerate(fracs):
        ax.plot(tau, g2_func('mixed', tau, n_mean=f * 100), color=cols[i], lw=2, label=f'ρ = {f:.0%}')
    ax.axhline(1, color='gray', ls=':', alpha=0.5)
    ax.set_xlabel('Delay τ [ns]'); ax.set_ylabel('g⁽²⁾(τ)')
    ax.set_title('(b) Raman + Background Mix\n(signal fraction ρ varies)')
    ax.legend(fontsize=9); ax.set_ylim(0.95, 2.1)
    
    # (c) Integration time
    ax = axes[2]
    scenarios = [
        ('Orbital B (680m)', calculate_snr_full(680, 5.0), '#9C27B0'),
        ('Recon (375m)', calculate_snr_full(375, 5.0), '#2196F3'),
        ('Close (200m)', calculate_snr_full(200, 5.0), '#4CAF50'),
    ]
    for label, r, color in scenarios:
        sr, br = r['sig_rate'], r['bg_rate']; total = sr + br
        mults = np.logspace(-1, 3, 200); hours = []
        for m in mults:
            s = sr * m; tot = s + br; rho = s / tot if tot > 0 else 0; dg = rho ** 2
            if dg > 0 and tot > 0:
                hours.append(9 / (dg ** 2 * tot ** 2 * 0.1e-9) / 3600)
            else: hours.append(1e12)
        ax.semilogy(mults, hours, color=color, lw=2.5, label=label)
    ax.axhline(1, color='gray', ls=':', alpha=0.5, label='1 hour')
    ax.axhline(24, color='orange', ls=':', alpha=0.5, label='24 hours')
    ax.axvline(1, color='black', ls='--', lw=1, alpha=0.5, label='Baseline signal')
    ax.set_xlabel('Signal Rate Multiplier')
    ax.set_ylabel('Integration Time for 3σ g⁽²⁾ Detection [hours]')
    ax.set_title('(c) Required Time for Correlation Discrimination\n(100 ps detector resolution)')
    ax.legend(fontsize=8, loc='upper right'); ax.set_xscale('log'); ax.set_ylim(1e-2, 1e10)
    
    plt.tight_layout()
    plt.savefig('Outputs/figures/fig8_g2.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved")
# POSTER FIGURES (5 key figures)
# =============================================================================

def fig1_feasibility_map(temp_K, solar_irr):
    """Fig 1: Main feasibility map."""
    print("Fig 1: Feasibility Map...")
    
    conc = np.linspace(1, 20, 80)
    dist = np.linspace(50, 2000, 80)
    snr_map = np.array([[calculate_snr(d, c, temp_K=temp_K, solar_irr=solar_irr) 
                         for d in dist] for c in conc])
    
    fig, ax = plt.subplots(figsize=(10, 8))
    levels = [0.5, 1, 2, 3, 5, 10, 20, 50, 100]
    im = ax.contourf(dist, conc, snr_map, levels=levels, cmap='RdYlGn',
                     norm=LogNorm(vmin=0.5, vmax=100))
    ax.contour(dist, conc, snr_map, levels=[5], colors='black', linewidths=3)
    plt.colorbar(im, ax=ax, label='SNR')
    
    ax.axvline(375, color='cyan', ls='--', lw=2, label='Recon (375m)')
    ax.axvline(680, color='orange', ls='--', lw=2, label='Orbital B (680m)')
    ax.set_xlabel('Distance (m)', fontsize=12)
    ax.set_ylabel('Organic Concentration (%)', fontsize=12)
    ax.set_title('Raman Detection Feasibility (248nm)\nBlack line = SNR=5 threshold', fontsize=13)
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig('Outputs/figures/fig1_feasibility_map.png', dpi=200)
    plt.close()
    print("  ✓ Saved")


def fig2_organic_threshold(temp_K, solar_irr):
    """Fig 2: Minimum organic % required vs distance."""
    print("Fig 2: Organic Threshold...")
    
    dists = np.linspace(100, 1500, 50)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    configs = [
        ('Baseline (50mJ, 30cm, 10s)', 0.05, 0.30, 10, 'blue'),
        ('Medium (100mJ, 40cm, 30s)', 0.10, 0.40, 30, 'green'),
        ('Optimized (200mJ, 50cm, 60s)', 0.20, 0.50, 60, 'red'),
    ]
    
    for name, E, A, t, color in configs:
        min_org = [find_min_organic(
            d,
            pulse_energy_J=E,
            aperture_m=A,
            integration_time_s=t,
            temp_K=temp_K,
            solar_irr=solar_irr,
        ) for d in dists]
        ax.plot(dists, min_org, '-', lw=2, color=color, label=name)
    
    ax.axhspan(3, 10, alpha=0.2, color='orange', label='Bennu organic range (3-10%)')
    ax.axvline(375, color='cyan', ls=':', alpha=0.7)
    ax.axvline(680, color='orange', ls=':', alpha=0.7)
    
    ax.set_xlabel('Distance (m)', fontsize=12)
    ax.set_ylabel('Minimum Organic % Required (SNR≥5)', fontsize=12)
    ax.set_title('Detection Threshold: What organic concentration is needed?', fontsize=13)
    ax.legend(loc='upper left')
    ax.set_xlim([100, 1500]); ax.set_ylim([0, 20])
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('Outputs/figures/fig2_organic_threshold.png', dpi=200)
    plt.close()
    print("  ✓ Saved")


def fig3_quantum_threshold():
    """Fig 3: Quantum squeezing threshold."""
    print("Fig 3: Quantum Threshold...")
    
    eta = np.linspace(0.01, 0.80, 100)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    for sq_dB in [3, 6, 10]:
        V_ideal = 10 ** (-sq_dB / 10)
        eff_dB = [-10 * np.log10(e * V_ideal + (1 - e)) for e in eta]
        ax.plot(eta * 100, eff_dB, lw=2, label=f'{sq_dB} dB ideal')
    
    ax.axvline(4.5, color='red', ls=':', lw=2, label='Current (~4.5%)')
    ax.axvline(30, color='green', ls=':', lw=2, label='Target (~30%)')
    ax.set_xlabel('Total Optical Efficiency (%)')
    ax.set_ylabel('Effective Squeezing (dB)')
    ax.set_title('Quantum Squeezing vs Optical Efficiency\n(Loss degrades quantum advantage)')
    ax.legend(loc='upper left')
    ax.set_xlim([0, 80]); ax.set_ylim([0, 10])
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('Outputs/figures/fig3_quantum_threshold.png', dpi=200)
    plt.close()
    print("  ✓ Saved")


def fig4_feasibility_matrix(temp_K, solar_irr):
    """Fig 4: Mission phase × organic % matrix."""
    print("Fig 4: Feasibility Matrix...")
    
    phases = ['Sample\nCollection', 'Recon', 'Orbital B', 'Detailed\nSurvey', 'Orbital A']
    distances = [50, 375, 680, 750, 1600]
    organics = ['10%', '5%', '3%', '1%']
    org_vals = [10, 5, 3, 1]
    
    snr_matrix = np.array([[calculate_snr(d, o, temp_K=temp_K, solar_irr=solar_irr) 
                            for d in distances] for o in org_vals])
    
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#d73027', '#fc8d59', '#fee08b', '#d9ef8b', '#91cf60', '#1a9850']
    cmap = LinearSegmentedColormap.from_list('feas', colors)
    
    im = ax.imshow(snr_matrix, cmap=cmap, aspect='auto', norm=LogNorm(vmin=0.1, vmax=500))
    for i in range(len(org_vals)):
        for j in range(len(distances)):
            snr = snr_matrix[i, j]
            color = 'white' if snr < 3 or snr > 50 else 'black'
            ax.text(j, i, f'{snr:.1f}', ha='center', va='center', fontsize=12, 
                   fontweight='bold', color=color)
    
    ax.set_xticks(range(len(phases))); ax.set_xticklabels(phases)
    ax.set_yticks(range(len(organics))); ax.set_yticklabels(organics)
    ax.set_xlabel('Mission Phase'); ax.set_ylabel('Organic %')
    ax.set_title('Detection Feasibility Matrix (SNR)\nGreen≥5: Feasible | Yellow 3-5: Marginal | Red<3: No')
    plt.colorbar(im, ax=ax, label='SNR', shrink=0.8)
    
    plt.tight_layout()
    plt.savefig('Outputs/figures/fig4_feasibility_matrix.png', dpi=200)
    plt.close()
    print("  ✓ Saved")


def fig5_bennu_realistic(ovirs_data, temp_K, solar_irr):
    """Fig 5: Bennu with realistic environment effects."""
    print("Fig 5: Bennu Realistic...")
    
    lat, lon, org = ovirs_data['lat'], ovirs_data['lon'], ovirs_data['organic_pct']
    env = SpaceEnvironment(lat, lon)
    env.summary()
    
    env_factors = env.get_factor(mission_day=100)
    
    print("  Computing SNR maps...")
    
    # Compare at 375m (Recon) where detection should work
    snr_ideal_375 = np.array([calculate_snr(375, o, temp_K=temp_K, solar_irr=solar_irr) 
                              for o in org])
    snr_real_375 = np.array([calculate_snr(375, o, temp_K=temp_K, solar_irr=solar_irr, 
                                           env_factor=ef) 
                             for o, ef in zip(org, env_factors)])
    
    # Also show 680m for comparison
    snr_ideal_680 = np.array([calculate_snr(680, o, temp_K=temp_K, solar_irr=solar_irr) 
                              for o in org])
    snr_real_680 = np.array([calculate_snr(680, o, temp_K=temp_K, solar_irr=solar_irr, 
                                           env_factor=ef) 
                             for o, ef in zip(org, env_factors)])
    
    cov_ideal_375 = 100 * (snr_ideal_375 >= 5).sum() / len(snr_ideal_375)
    cov_real_375 = 100 * (snr_real_375 >= 5).sum() / len(snr_real_375)
    cov_ideal_680 = 100 * (snr_ideal_680 >= 5).sum() / len(snr_ideal_680)
    cov_real_680 = 100 * (snr_real_680 >= 5).sum() / len(snr_real_680)
    
    fig = plt.figure(figsize=(18, 10))
    
    # Row 1: Inputs
    ax1 = fig.add_subplot(2, 3, 1)
    sc = ax1.scatter(lon, lat, c=org, s=1, cmap='YlOrRd', vmin=0, vmax=15)
    plt.colorbar(sc, ax=ax1, label='Organic %')
    ax1.set_xlim([0, 360]); ax1.set_ylim([-90, 90])
    ax1.set_title('Organic Distribution (OVIRS)')
    ax1.set_xlabel('Longitude (°)'); ax1.set_ylabel('Latitude (°)')
    
    ax2 = fig.add_subplot(2, 3, 2)
    sc = ax2.scatter(lon, lat, c=env.incidence, s=1, cmap='coolwarm', vmin=0, vmax=90)
    plt.colorbar(sc, ax=ax2, label='Angle (°)')
    ax2.set_xlim([0, 360]); ax2.set_ylim([-90, 90])
    ax2.set_title('Solar Incidence Angle')
    ax2.set_xlabel('Longitude (°)'); ax2.set_ylabel('Latitude (°)')
    
    ax3 = fig.add_subplot(2, 3, 3)
    sc = ax3.scatter(lon, lat, c=env_factors, s=1, cmap='viridis', vmin=0, vmax=1)
    plt.colorbar(sc, ax=ax3, label='Factor')
    ax3.set_xlim([0, 360]); ax3.set_ylim([-90, 90])
    ax3.set_title('Combined Environment Factor\n(incidence × roughness × shadow × dust)')
    ax3.set_xlabel('Longitude (°)'); ax3.set_ylabel('Latitude (°)')
    
    # Row 2: Results at 375m (where it works)
    ax4 = fig.add_subplot(2, 3, 4)
    sc = ax4.scatter(lon, lat, c=snr_ideal_375, s=1, cmap='RdYlGn', norm=LogNorm(0.5, 100))
    plt.colorbar(sc, ax=ax4, label='SNR')
    ax4.set_xlim([0, 360]); ax4.set_ylim([-90, 90])
    ax4.set_title(f'IDEAL SNR @ 375m\n{cov_ideal_375:.1f}% detectable')
    ax4.set_xlabel('Longitude (°)'); ax4.set_ylabel('Latitude (°)')
    
    ax5 = fig.add_subplot(2, 3, 5)
    sc = ax5.scatter(lon, lat, c=snr_real_375, s=1, cmap='RdYlGn', norm=LogNorm(0.5, 100))
    plt.colorbar(sc, ax=ax5, label='SNR')
    ax5.set_xlim([0, 360]); ax5.set_ylim([-90, 90])
    ax5.set_title(f'REALISTIC SNR @ 375m\n{cov_real_375:.1f}% detectable')
    ax5.set_xlabel('Longitude (°)'); ax5.set_ylabel('Latitude (°)')
    
    # Coverage comparison at multiple distances
    ax6 = fig.add_subplot(2, 3, 6)
    dists = [200, 300, 375, 500, 680, 800, 1000]
    cov_i, cov_r = [], []
    for d in dists:
        si = np.array([calculate_snr(d, o, temp_K=temp_K, solar_irr=solar_irr) for o in org])
        sr = np.array([calculate_snr(d, o, temp_K=temp_K, solar_irr=solar_irr, env_factor=ef) 
                      for o, ef in zip(org, env_factors)])
        cov_i.append(100 * (si >= 5).sum() / len(si))
        cov_r.append(100 * (sr >= 5).sum() / len(sr))
    
    ax6.fill_between(dists, cov_r, cov_i, alpha=0.3, color='blue', label='Environment penalty')
    ax6.plot(dists, cov_i, 'g--o', lw=2, ms=6, label='Ideal')
    ax6.plot(dists, cov_r, 'r-o', lw=2, ms=8, label='Realistic')
    ax6.axvline(375, color='cyan', ls=':', alpha=0.7, label='Recon (375m)')
    ax6.axvline(680, color='orange', ls=':', alpha=0.7, label='Orbital B (680m)')
    ax6.set_xlabel('Distance (m)')
    ax6.set_ylabel('Detectable Surface (%)')
    ax6.set_title('Coverage: Ideal vs Realistic')
    ax6.legend(loc='upper right', fontsize=8)
    ax6.set_xlim([150, 1050]); ax6.set_ylim([0, 105])
    ax6.grid(True, alpha=0.3)
    
    plt.suptitle('Realistic Orbital Raman on Bennu (248nm, mission day 100)\n'
                 f'375m: {cov_ideal_375:.0f}%→{cov_real_375:.0f}% | 680m: {cov_ideal_680:.0f}%→{cov_real_680:.0f}%',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig('Outputs/figures/fig5_bennu_realistic.png', dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved")
    print(f"  375m: Ideal={cov_ideal_375:.1f}%, Realistic={cov_real_375:.1f}%")
    print(f"  680m: Ideal={cov_ideal_680:.1f}%, Realistic={cov_real_680:.1f}%")
    
    with open('Outputs/results/coverage.csv', 'w') as f:
        f.write('distance_m,ideal_pct,realistic_pct\n')
        for d, ci, cr in zip(dists, cov_i, cov_r):
            f.write(f'{d},{ci:.1f},{cr:.1f}\n')


# =============================================================================
# MAIN
# =============================================================================

def main():
    os.makedirs('Outputs/figures', exist_ok=True)
    os.makedirs('Outputs/results', exist_ok=True)

    print("="*60)
    print("ORBITAL RAMAN FEASIBILITY STUDY")
    print("With Realistic Space Environment")
    print("="*60)
    
    print("\nLOADING DATASETS:")
    ovirs_path = 'Data/l_1600mm_sp_ovirs_reca_bandarea3200to3600nm_allsites_wavc_0000n00000.fits'
    ovirs = load_ovirs(ovirs_path) if os.path.exists(ovirs_path) else make_synthetic()
    
    data_dir = 'Data'
    if os.path.isdir(data_dir):
        otes_files = [f for f in os.listdir(data_dir) if 'ote' in f.lower() and f.endswith('.hdf5')]
        tsis_files = [f for f in os.listdir(data_dir) if 'tsis' in f.lower() and f.endswith('.nc')]
    else:
        otes_files = []
        tsis_files = []

    temp_K = load_otes(f'{data_dir}/{otes_files[0]}') if otes_files else 300.0
    solar_irr = load_tsis1(f'{data_dir}/{tsis_files[0]}') if tsis_files else 0.015
    
    print("\nMODEL CHECK:")
    print(f"  5% @ 500m (ideal): SNR = {calculate_snr(500, 5):.1f}")
    print(f"  5% @ 500m (50% env): SNR = {calculate_snr(500, 5, env_factor=0.5):.1f}")
    
    print("\nGENERATING FIGURES:")
    fig1_feasibility_map(temp_K, solar_irr)
    fig2_organic_threshold(temp_K, solar_irr)
    fig3_quantum_threshold()
    fig4_feasibility_matrix(temp_K, solar_irr)
    fig5_bennu_realistic(ovirs, temp_K, solar_irr)
    
    print("\nEXTENDED ANALYSIS:")
    mc = fig6_monte_carlo()
    fig7_squeezing()
    fig8_g2()
    
    print("\n" + "="*60)
    print("DONE! 8 figures:")
    print("  fig1: Feasibility map")
    print("  fig2: Organic threshold")
    print("  fig3: Quantum threshold")  
    print("  fig4: Mission matrix")
    print("  fig5: Bennu realistic (6 panels)")
    print("  fig6: Monte Carlo UQ (4 panels)")
    print("  fig7: Squeezing analysis (2 panels)")
    print("  fig8: g² correlation (3 panels)")
    print("="*60)


if __name__ == "__main__":
    main()