import { useState, useCallback, useEffect, useRef } from "react";

// ============================================================================
// PHYSICS ENGINE — exact copy of src.py calculate_snr()
// ============================================================================
function calculateSNR(distance_m, organic_pct, params = {}) {
  const {
    wavelength_nm = 248,
    pulse_energy_J = 0.050,
    aperture_m = 0.30,
    integration_time_s = 10,
    temp_K = 300,
    solar_irr = 0.015,
    env_factor = 1.0,
  } = params;

  const h = 6.626e-34, c = 3e8;
  const rep_rate = 20, QE = 0.30;
  const n_pulses = rep_rate * integration_time_s;
  const E_photon = h * c / (wavelength_nm * 1e-9);
  const photons_per_pulse = pulse_energy_J / E_photon;

  const K_R_1pct_532 = 1e-11;
  const lambda_factor = Math.pow(532 / wavelength_nm, 4);
  const resonance = wavelength_nm < 260 ? 10 : (wavelength_nm < 300 ? 5 : 1);
  const K_R = K_R_1pct_532 * (organic_pct / 1.0) * lambda_factor * resonance * env_factor;

  const A_tel = Math.PI * Math.pow(aperture_m / 2, 2);
  const solid_angle = A_tel / Math.pow(distance_m, 2);
  const spot_area = Math.PI * Math.pow(distance_m * 0.5e-3, 2);

  const N_signal = (photons_per_pulse / spot_area) * K_R * solid_angle * spot_area * n_pulses * QE;

  const gate = n_pulses * 20e-9;
  const N_solar = solar_irr / 1.27 * 0.5 * A_tel * 0.044 * gate * QE / (h * c / 280e-9) * 0.1;
  const N_dark = 100 * Math.exp((temp_K - 300) / 30) * integration_time_s;
  const N_read = 25 * n_pulses;

  if (N_signal <= 0) return { snr: 0, N_signal: 0, N_solar, N_dark, N_read, N_bg: N_solar + N_dark + N_read };
  const N_bg = N_solar + N_dark + N_read;
  const snr = N_signal / Math.sqrt(N_signal + N_bg);
  return { snr, N_signal, N_solar, N_dark, N_read, N_bg };
}

function findMaxRange(params, snrThreshold = 5) {
  for (let d = 5000; d >= 50; d -= 10) {
    if (calculateSNR(d, params.organic_pct || 5, params).snr >= snrThreshold) return d;
  }
  return 50;
}

function effectiveSqueezing(eta, idealDB) {
  const V_ideal = Math.pow(10, -idealDB / 10);
  const V_eff = eta * V_ideal + (1 - eta);
  return -10 * Math.log10(V_eff);
}

// Monte Carlo (simplified for browser — 2000 trials)
function runMonteCarlo(baseParams, N = 2000) {
  const ranges = {
    pulse_energy_J: [0.030, 0.070],
    aperture_m: [0.25, 0.35],
    integration_time_s: [8, 12],
    temp_K: [280, 350],
    solar_irr: [0.010, 0.020],
    organic_pct: [1.0, 15.0],
    env_factor: [0.3, 1.0],
  };
  const results = [];
  for (let i = 0; i < N; i++) {
    const trial = { ...baseParams };
    for (const [key, [lo, hi]] of Object.entries(ranges)) {
      trial[key] = lo + Math.random() * (hi - lo);
    }
    const org = trial.organic_pct;
    delete trial.organic_pct;
    const r = calculateSNR(baseParams.distance || 680, org, trial);
    results.push({ snr: r.snr, organic: org, env: trial.env_factor });
  }
  results.sort((a, b) => a.snr - b.snr);
  const snrs = results.map(r => r.snr);
  const median = snrs[Math.floor(N / 2)];
  const ci_lo = snrs[Math.floor(N * 0.025)];
  const ci_hi = snrs[Math.floor(N * 0.975)];
  const pDetect = snrs.filter(s => s >= 5).length / N * 100;
  return { results, median, ci_lo, ci_hi, pDetect, snrs };
}

// Known asteroids with approximate parameters
const ASTEROID_DB = {
  "Bennu": { albedo: 0.044, temp_K: 300, solar_irr: 0.015, type: "B-type carbonaceous", diameter_m: 490 },
  "Ryugu": { albedo: 0.045, temp_K: 300, solar_irr: 0.014, type: "Cb-type carbonaceous", diameter_m: 900 },
  "Itokawa": { albedo: 0.53, temp_K: 330, solar_irr: 0.020, type: "S-type siliceous", diameter_m: 350 },
  "Eros": { albedo: 0.25, temp_K: 280, solar_irr: 0.018, type: "S-type siliceous", diameter_m: 16840 },
  "Ceres": { albedo: 0.09, temp_K: 168, solar_irr: 0.005, type: "C-type carbonaceous", diameter_m: 946000 },
  "Vesta": { albedo: 0.42, temp_K: 190, solar_irr: 0.006, type: "V-type basaltic", diameter_m: 525400 },
  "Psyche": { albedo: 0.15, temp_K: 170, solar_irr: 0.004, type: "M-type metallic", diameter_m: 226000 },
  "Apophis": { albedo: 0.30, temp_K: 310, solar_irr: 0.020, type: "Sq-type", diameter_m: 370 },
  "Didymos": { albedo: 0.15, temp_K: 290, solar_irr: 0.017, type: "S-type", diameter_m: 780 },
  "Custom": { albedo: 0.10, temp_K: 300, solar_irr: 0.015, type: "Custom", diameter_m: 500 },
};

// ============================================================================
// MINI CHART COMPONENTS (pure SVG, no dependencies)
// ============================================================================

function SNRCurve({ params, width = 500, height = 280 }) {
  const pad = { t: 30, r: 20, b: 45, l: 55 };
  const w = width - pad.l - pad.r, h = height - pad.t - pad.b;
  const dists = Array.from({ length: 80 }, (_, i) => 50 + i * (2000 - 50) / 79);
  const organics = [1, 3, 5, 10];
  const colors = ["#ef4444", "#f59e0b", "#22c55e", "#06b6d4"];

  const maxSNR = 100;
  const toX = d => pad.l + (d - 50) / (2000 - 50) * w;
  const toY = s => pad.t + h - Math.min(s, maxSNR) / maxSNR * h;

  return (
    <svg width={width} height={height} style={{ background: "transparent" }}>
      <rect x={pad.l} y={pad.t} width={w} height={h} fill="rgba(0,0,0,0.03)" rx={4} />
      {/* SNR=5 threshold */}
      <line x1={pad.l} y1={toY(5)} x2={pad.l + w} y2={toY(5)}
        stroke="#888" strokeWidth={1.5} strokeDasharray="6,4" />
      <text x={pad.l + w - 4} y={toY(5) - 5} fill="#888" fontSize={10} textAnchor="end">SNR=5</text>

      {organics.map((org, idx) => {
        const points = dists.map(d => {
          const r = calculateSNR(d, org, params);
          return `${toX(d)},${toY(r.snr)}`;
        }).join(" ");
        return <polyline key={org} points={points} fill="none"
          stroke={colors[idx]} strokeWidth={2} opacity={0.9} />;
      })}

      {/* Axes */}
      <line x1={pad.l} y1={pad.t + h} x2={pad.l + w} y2={pad.t + h} stroke="#666" />
      <line x1={pad.l} y1={pad.t} x2={pad.l} y2={pad.t + h} stroke="#666" />
      {[0, 500, 1000, 1500, 2000].map(d => (
        <text key={d} x={toX(d)} y={pad.t + h + 16} fill="#666" fontSize={10} textAnchor="middle">{d}</text>
      ))}
      {[0, 20, 40, 60, 80, 100].map(s => (
        <text key={s} x={pad.l - 8} y={toY(s) + 4} fill="#666" fontSize={10} textAnchor="end">{s}</text>
      ))}
      <text x={pad.l + w / 2} y={height - 4} fill="#999" fontSize={11} textAnchor="middle">Distance (m)</text>
      <text x={14} y={pad.t + h / 2} fill="#999" fontSize={11} textAnchor="middle"
        transform={`rotate(-90, 14, ${pad.t + h / 2})`}>SNR</text>

      {/* Legend */}
      {organics.map((org, i) => (
        <g key={org} transform={`translate(${pad.l + 10}, ${pad.t + 8 + i * 16})`}>
          <line x1={0} y1={0} x2={14} y2={0} stroke={colors[i]} strokeWidth={2} />
          <text x={18} y={4} fill="#ccc" fontSize={10}>{org}%</text>
        </g>
      ))}
    </svg>
  );
}

function FeasibilityMatrix({ params }) {
  const phases = [
    { name: "Sample\nCollection", dist: 50 },
    { name: "Recon", dist: 375 },
    { name: "Orbital B", dist: 680 },
    { name: "Survey", dist: 750 },
    { name: "Orbital A", dist: 1600 },
  ];
  const organics = [10, 5, 3, 1];

  const getColor = (snr) => {
    if (snr >= 5) return `rgba(34, 197, 94, ${Math.min(snr / 50, 1) * 0.7 + 0.3})`;
    if (snr >= 3) return `rgba(245, 158, 11, ${0.5 + snr / 10})`;
    return `rgba(239, 68, 68, ${Math.max(0.3, snr / 5)})`;
  };

  return (
    <div style={{ overflowX: "auto" }}>
      <table style={{ borderCollapse: "collapse", width: "100%", fontSize: 12 }}>
        <thead>
          <tr>
            <th style={{ padding: "6px 8px", color: "#999", fontWeight: 500 }}>Organic %</th>
            {phases.map(p => (
              <th key={p.name} style={{ padding: "6px 8px", color: "#999", fontWeight: 500, textAlign: "center" }}>
                {p.name.split("\n").map((l, i) => <div key={i}>{l}</div>)}
                <div style={{ fontSize: 9, color: "#666" }}>{p.dist}m</div>
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {organics.map(org => (
            <tr key={org}>
              <td style={{ padding: "6px 8px", color: "#ccc", fontWeight: 600 }}>{org}%</td>
              {phases.map(p => {
                const r = calculateSNR(p.dist, org, params);
                return (
                  <td key={p.name} style={{
                    padding: "8px 6px", textAlign: "center", fontWeight: 700,
                    background: getColor(r.snr), color: r.snr > 50 || r.snr < 1 ? "#fff" : "#000",
                    borderRadius: 4, fontSize: 13,
                  }}>
                    {r.snr < 0.1 ? "<0.1" : r.snr.toFixed(1)}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function MCHistogram({ mcData, width = 480, height = 200 }) {
  if (!mcData) return null;
  const { snrs, median, ci_lo, ci_hi, pDetect } = mcData;
  const pad = { t: 25, r: 15, b: 35, l: 45 };
  const w = width - pad.l - pad.r, h = height - pad.t - pad.b;

  const maxSNR = Math.min(Math.max(...snrs), 30);
  const nBins = 40;
  const bins = Array(nBins).fill(0);
  snrs.forEach(s => {
    const idx = Math.min(Math.floor(s / maxSNR * nBins), nBins - 1);
    if (idx >= 0) bins[idx]++;
  });
  const maxCount = Math.max(...bins);
  const bw = w / nBins;

  const toX = v => pad.l + v / maxSNR * w;
  const toY = c => pad.t + h - c / maxCount * h;

  return (
    <svg width={width} height={height}>
      {bins.map((count, i) => (
        <rect key={i} x={pad.l + i * bw} y={toY(count)} width={bw - 1} height={h - (toY(count) - pad.t)}
          fill="#3b82f6" opacity={0.7} rx={1} />
      ))}
      <line x1={toX(5)} y1={pad.t} x2={toX(5)} y2={pad.t + h}
        stroke="#22c55e" strokeWidth={2} strokeDasharray="4,3" />
      <line x1={toX(median)} y1={pad.t} x2={toX(median)} y2={pad.t + h}
        stroke="#ef4444" strokeWidth={2} />
      <text x={toX(median) + 4} y={pad.t + 12} fill="#ef4444" fontSize={9}>med={median.toFixed(1)}</text>
      <text x={toX(5) + 4} y={pad.t + 24} fill="#22c55e" fontSize={9}>SNR=5</text>
      <line x1={pad.l} y1={pad.t + h} x2={pad.l + w} y2={pad.t + h} stroke="#666" />
      <text x={pad.l + w / 2} y={height - 4} fill="#888" fontSize={10} textAnchor="middle">SNR</text>
      <text x={pad.l + w} y={pad.t + 12} fill="#999" fontSize={10} textAnchor="end">
        P(detect) = {pDetect.toFixed(1)}%
      </text>
    </svg>
  );
}

function SqueezePlot({ width = 480, height = 220 }) {
  const pad = { t: 25, r: 15, b: 35, l: 50 };
  const w = width - pad.l - pad.r, h = height - pad.t - pad.b;
  const etas = Array.from({ length: 80 }, (_, i) => 0.01 + i * 0.79 / 79);
  const dBs = [3, 6, 10];
  const colors = ["#3b82f6", "#f59e0b", "#22c55e"];

  const toX = e => pad.l + (e - 0.01) / 0.79 * w;
  const toY = d => pad.t + h - d / 10 * h;

  return (
    <svg width={width} height={height}>
      {dBs.map((db, idx) => {
        const pts = etas.map(e => `${toX(e)},${toY(effectiveSqueezing(e, db))}`).join(" ");
        return <polyline key={db} points={pts} fill="none" stroke={colors[idx]} strokeWidth={2} />;
      })}
      <line x1={toX(0.045)} y1={pad.t} x2={toX(0.045)} y2={pad.t + h}
        stroke="#ef4444" strokeWidth={1.5} strokeDasharray="4,3" />
      <line x1={toX(0.30)} y1={pad.t} x2={toX(0.30)} y2={pad.t + h}
        stroke="#22c55e" strokeWidth={1.5} strokeDasharray="4,3" />
      <text x={toX(0.045) + 3} y={pad.t + 12} fill="#ef4444" fontSize={9}>4.5%</text>
      <text x={toX(0.30) + 3} y={pad.t + 12} fill="#22c55e" fontSize={9}>30%</text>
      <line x1={pad.l} y1={pad.t + h} x2={pad.l + w} y2={pad.t + h} stroke="#666" />
      <line x1={pad.l} y1={pad.t} x2={pad.l} y2={pad.t + h} stroke="#666" />
      <text x={pad.l + w / 2} y={height - 4} fill="#888" fontSize={10} textAnchor="middle">
        Total Optical Efficiency η (%)
      </text>
      <text x={10} y={pad.t + h / 2} fill="#888" fontSize={10} textAnchor="middle"
        transform={`rotate(-90, 10, ${pad.t + h / 2})`}>Effective dB</text>
      {dBs.map((db, i) => (
        <g key={db} transform={`translate(${pad.l + w - 80}, ${pad.t + 8 + i * 14})`}>
          <line x1={0} y1={0} x2={12} y2={0} stroke={colors[i]} strokeWidth={2} />
          <text x={16} y={4} fill="#bbb" fontSize={9}>{db} dB ideal</text>
        </g>
      ))}
      {[0, 20, 40, 60, 80].map(p => (
        <text key={p} x={toX(p / 100)} y={pad.t + h + 14} fill="#666" fontSize={9} textAnchor="middle">{p}</text>
      ))}
    </svg>
  );
}

// ============================================================================
// MAIN APP
// ============================================================================

export default function OrbitalRamanSimulator() {
  const [asteroid, setAsteroid] = useState("Bennu");
  const [params, setParams] = useState({
    wavelength_nm: 248, pulse_energy_J: 0.050, aperture_m: 0.30,
    integration_time_s: 10, temp_K: 300, solar_irr: 0.015,
    env_factor: 1.0, distance: 680, organic_pct: 5,
  });
  const [tab, setTab] = useState("snr");
  const [mcData, setMcData] = useState(null);
  const [jplQuery, setJplQuery] = useState("");
  const [jplResults, setJplResults] = useState(null);
  const [jplLoading, setJplLoading] = useState(false);

  // Update params when asteroid changes
  useEffect(() => {
    const a = ASTEROID_DB[asteroid];
    if (a) {
      setParams(p => ({ ...p, temp_K: a.temp_K, solar_irr: a.solar_irr }));
    }
  }, [asteroid]);

  const updateParam = (key, val) => {
    setParams(p => ({ ...p, [key]: parseFloat(val) || 0 }));
    setMcData(null);
  };

  const runMC = useCallback(() => {
    setMcData(runMonteCarlo(params, 2000));
  }, [params]);

  // JPL Small Body Database lookup
  const searchJPL = async () => {
    if (!jplQuery.trim()) return;
    setJplLoading(true);
    try {
      const res = await fetch(
        `https://ssd-api.jpl.nasa.gov/sbdb.api?sstr=${encodeURIComponent(jplQuery)}&phys-par=true`
      );
      const data = await res.json();
      if (data.phys_par) {
        const findParam = (name) => {
          const p = data.phys_par.find(pp => pp.name === name);
          return p ? parseFloat(p.value) : null;
        };
        const albedo = findParam("albedo") || 0.1;
        const diameter = findParam("diameter");
        setJplResults({
          name: data.object?.fullname || jplQuery,
          albedo,
          diameter: diameter ? `${diameter} km` : "unknown",
          spkid: data.object?.spkid,
          orbit_class: data.object?.orbit_class?.name || "unknown",
        });
        // Estimate temp from albedo (rough: lower albedo = more absorption = hotter for similar distance)
        const estTemp = 250 + (1 - albedo) * 150;
        setParams(p => ({ ...p, temp_K: Math.round(estTemp), solar_irr: 0.015 }));
        setAsteroid("Custom");
      } else {
        setJplResults({ error: "Asteroid not found or no physical data available" });
      }
    } catch {
      setJplResults({ error: "Failed to reach JPL API" });
    }
    setJplLoading(false);
  };

  // Current SNR
  const current = calculateSNR(params.distance, params.organic_pct, params);
  const maxRange = findMaxRange({ ...params, organic_pct: params.organic_pct });

  const tabs = [
    { id: "snr", label: "SNR Curves" },
    { id: "matrix", label: "Mission Matrix" },
    { id: "mc", label: "Monte Carlo" },
    { id: "squeeze", label: "Squeezing" },
  ];

  const inputStyle = {
    background: "rgba(255,255,255,0.06)", border: "1px solid rgba(255,255,255,0.12)",
    borderRadius: 6, padding: "6px 10px", color: "#e2e8f0", fontSize: 13, width: "100%",
    outline: "none", transition: "border 0.2s",
  };
  const labelStyle = { color: "#94a3b8", fontSize: 11, marginBottom: 2, display: "block" };
  const cardStyle = {
    background: "rgba(255,255,255,0.03)", border: "1px solid rgba(255,255,255,0.08)",
    borderRadius: 10, padding: 16, marginBottom: 12,
  };

  return (
    <div style={{
      fontFamily: "'JetBrains Mono', 'SF Mono', 'Fira Code', monospace",
      background: "linear-gradient(145deg, #0a0a0f 0%, #0d1117 50%, #0a0f1a 100%)",
      color: "#e2e8f0", minHeight: "100vh", padding: "16px 12px",
    }}>
      {/* Header */}
      <div style={{ textAlign: "center", marginBottom: 20 }}>
        <div style={{ fontSize: 10, letterSpacing: 4, color: "#64748b", textTransform: "uppercase", marginBottom: 4 }}>
          Orbital Spectroscopy Lab
        </div>
        <h1 style={{
          fontSize: 22, fontWeight: 700, margin: 0,
          background: "linear-gradient(135deg, #60a5fa, #34d399)",
          WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent",
        }}>
          Raman Detection Simulator
        </h1>
        <div style={{ fontSize: 11, color: "#64748b", marginTop: 4 }}>
          248nm DUV · Resonance-Enhanced · Pulsed Time-Gated
        </div>
      </div>

      {/* Live SNR readout */}
      <div style={{
        display: "flex", justifyContent: "center", gap: 24, marginBottom: 16,
        flexWrap: "wrap",
      }}>
        <div style={{ textAlign: "center" }}>
          <div style={{ fontSize: 36, fontWeight: 800, color: current.snr >= 5 ? "#22c55e" : current.snr >= 3 ? "#f59e0b" : "#ef4444" }}>
            {current.snr.toFixed(1)}
          </div>
          <div style={{ fontSize: 10, color: "#64748b" }}>SNR @ {params.distance}m</div>
        </div>
        <div style={{ textAlign: "center" }}>
          <div style={{ fontSize: 36, fontWeight: 800, color: "#60a5fa" }}>
            {maxRange < 5000 ? `${maxRange}` : ">5k"}
          </div>
          <div style={{ fontSize: 10, color: "#64748b" }}>Max Range (m)</div>
        </div>
        <div style={{ textAlign: "center" }}>
          <div style={{ fontSize: 36, fontWeight: 800, color: "#a78bfa" }}>
            {current.N_signal.toFixed(0)}
          </div>
          <div style={{ fontSize: 10, color: "#64748b" }}>Signal γ</div>
        </div>
      </div>

      {/* Asteroid selector + JPL lookup */}
      <div style={cardStyle}>
        <div style={{ display: "flex", gap: 8, flexWrap: "wrap", alignItems: "end" }}>
          <div style={{ flex: "1 1 140px" }}>
            <label style={labelStyle}>Target Asteroid</label>
            <select value={asteroid} onChange={e => setAsteroid(e.target.value)}
              style={{ ...inputStyle, cursor: "pointer" }}>
              {Object.keys(ASTEROID_DB).map(a => <option key={a} value={a}>{a}</option>)}
            </select>
          </div>
          <div style={{ flex: "2 1 200px" }}>
            <label style={labelStyle}>Or search NASA JPL database</label>
            <div style={{ display: "flex", gap: 6 }}>
              <input value={jplQuery} onChange={e => setJplQuery(e.target.value)}
                onKeyDown={e => e.key === "Enter" && searchJPL()}
                placeholder="e.g. Phaethon, 2024 YR4..."
                style={{ ...inputStyle, flex: 1 }} />
              <button onClick={searchJPL} disabled={jplLoading} style={{
                ...inputStyle, width: "auto", cursor: "pointer", padding: "6px 14px",
                background: "rgba(59,130,246,0.2)", border: "1px solid rgba(59,130,246,0.4)",
                color: "#60a5fa", fontWeight: 600,
              }}>
                {jplLoading ? "..." : "Search"}
              </button>
            </div>
          </div>
        </div>
        {jplResults && !jplResults.error && (
          <div style={{ marginTop: 8, fontSize: 11, color: "#94a3b8", background: "rgba(59,130,246,0.08)", padding: "8px 12px", borderRadius: 6 }}>
            <strong style={{ color: "#60a5fa" }}>{jplResults.name}</strong> · albedo: {jplResults.albedo} · diameter: {jplResults.diameter} · class: {jplResults.orbit_class}
          </div>
        )}
        {jplResults?.error && (
          <div style={{ marginTop: 8, fontSize: 11, color: "#f59e0b", padding: "6px 12px" }}>{jplResults.error}</div>
        )}
        {ASTEROID_DB[asteroid] && asteroid !== "Custom" && (
          <div style={{ marginTop: 6, fontSize: 10, color: "#64748b" }}>
            {ASTEROID_DB[asteroid].type} · albedo {ASTEROID_DB[asteroid].albedo} · Ø {(ASTEROID_DB[asteroid].diameter_m / 1000).toFixed(1)} km
          </div>
        )}
      </div>

      {/* Parameter controls */}
      <div style={cardStyle}>
        <div style={{ fontSize: 11, color: "#64748b", marginBottom: 8, fontWeight: 600, letterSpacing: 1, textTransform: "uppercase" }}>
          Instrument & Target
        </div>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(130px, 1fr))", gap: 8 }}>
          {[
            ["distance", "Distance (m)", params.distance, 50, 5000, 10],
            ["organic_pct", "Organic %", params.organic_pct, 0.1, 50, 0.1],
            ["pulse_energy_J", "Pulse (mJ)", params.pulse_energy_J * 1000, 5, 500, 5],
            ["aperture_m", "Aperture (cm)", params.aperture_m * 100, 10, 100, 5],
            ["integration_time_s", "Int. Time (s)", params.integration_time_s, 1, 120, 1],
            ["wavelength_nm", "λ (nm)", params.wavelength_nm, 200, 600, 1],
            ["temp_K", "Temp (K)", params.temp_K, 100, 500, 5],
            ["env_factor", "Environment", params.env_factor, 0.1, 1, 0.05],
          ].map(([key, label, val, min, max, step]) => (
            <div key={key}>
              <label style={labelStyle}>{label}</label>
              <input type="number" value={val} min={min} max={max} step={step}
                onChange={e => {
                  let v = parseFloat(e.target.value);
                  if (key === "pulse_energy_J") v /= 1000;
                  if (key === "aperture_m") v /= 100;
                  updateParam(key, v);
                }}
                style={inputStyle} />
            </div>
          ))}
        </div>
      </div>

      {/* Tabs */}
      <div style={{ display: "flex", gap: 4, marginBottom: 12 }}>
        {tabs.map(t => (
          <button key={t.id} onClick={() => setTab(t.id)} style={{
            flex: 1, padding: "8px 4px", borderRadius: 6, border: "none", cursor: "pointer",
            fontSize: 11, fontWeight: 600, fontFamily: "inherit", letterSpacing: 0.5,
            background: tab === t.id ? "rgba(59,130,246,0.2)" : "rgba(255,255,255,0.04)",
            color: tab === t.id ? "#60a5fa" : "#64748b",
            transition: "all 0.2s",
          }}>
            {t.label}
          </button>
        ))}
      </div>

      {/* Tab content */}
      <div style={cardStyle}>
        {tab === "snr" && (
          <div>
            <div style={{ fontSize: 12, color: "#94a3b8", marginBottom: 8, fontWeight: 600 }}>
              SNR vs Distance — {asteroid} at {params.wavelength_nm}nm
            </div>
            <SNRCurve params={params} width={Math.min(window.innerWidth - 60, 520)} />
            <div style={{ fontSize: 10, color: "#64748b", marginTop: 6 }}>
              Each curve shows a different organic concentration (1%, 3%, 5%, 10%).
              Dashed line = SNR=5 detection threshold.
            </div>
          </div>
        )}

        {tab === "matrix" && (
          <div>
            <div style={{ fontSize: 12, color: "#94a3b8", marginBottom: 8, fontWeight: 600 }}>
              Mission Phase × Organic % Feasibility
            </div>
            <FeasibilityMatrix params={params} />
            <div style={{ display: "flex", gap: 12, marginTop: 8, fontSize: 10 }}>
              <span style={{ color: "#22c55e" }}>■ SNR ≥ 5: Feasible</span>
              <span style={{ color: "#f59e0b" }}>■ SNR 3–5: Marginal</span>
              <span style={{ color: "#ef4444" }}>■ SNR {"<"} 3: Not feasible</span>
            </div>
          </div>
        )}

        {tab === "mc" && (
          <div>
            <div style={{ fontSize: 12, color: "#94a3b8", marginBottom: 8, fontWeight: 600 }}>
              Monte Carlo Uncertainty (2,000 trials at {params.distance}m)
            </div>
            {!mcData ? (
              <div style={{ textAlign: "center", padding: 20 }}>
                <button onClick={runMC} style={{
                  padding: "10px 24px", borderRadius: 8, border: "none", cursor: "pointer",
                  background: "linear-gradient(135deg, #3b82f6, #8b5cf6)",
                  color: "white", fontWeight: 700, fontSize: 13, fontFamily: "inherit",
                }}>
                  Run Monte Carlo
                </button>
                <div style={{ fontSize: 10, color: "#64748b", marginTop: 6 }}>
                  Samples instrument & target uncertainties through the full photon budget
                </div>
              </div>
            ) : (
              <div>
                <MCHistogram mcData={mcData} width={Math.min(window.innerWidth - 60, 520)} />
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8, marginTop: 8 }}>
                  <div style={{ background: "rgba(59,130,246,0.1)", padding: 10, borderRadius: 6 }}>
                    <div style={{ fontSize: 10, color: "#64748b" }}>Median SNR</div>
                    <div style={{ fontSize: 20, fontWeight: 700, color: "#60a5fa" }}>
                      {mcData.median.toFixed(1)}
                    </div>
                  </div>
                  <div style={{ background: "rgba(34,197,94,0.1)", padding: 10, borderRadius: 6 }}>
                    <div style={{ fontSize: 10, color: "#64748b" }}>P(Detect)</div>
                    <div style={{ fontSize: 20, fontWeight: 700, color: "#22c55e" }}>
                      {mcData.pDetect.toFixed(1)}%
                    </div>
                  </div>
                  <div style={{ background: "rgba(239,68,68,0.1)", padding: 10, borderRadius: 6 }}>
                    <div style={{ fontSize: 10, color: "#64748b" }}>95% CI</div>
                    <div style={{ fontSize: 14, fontWeight: 600, color: "#f87171" }}>
                      [{mcData.ci_lo.toFixed(1)}, {mcData.ci_hi.toFixed(1)}]
                    </div>
                  </div>
                  <div style={{ background: "rgba(167,139,250,0.1)", padding: 10, borderRadius: 6 }}>
                    <div style={{ fontSize: 10, color: "#64748b" }}>Signal Photons</div>
                    <div style={{ fontSize: 14, fontWeight: 600, color: "#a78bfa" }}>
                      {current.N_signal.toFixed(0)} ± {(current.N_signal * 0.4).toFixed(0)}
                    </div>
                  </div>
                </div>
                <button onClick={() => setMcData(null)} style={{
                  marginTop: 8, padding: "4px 12px", borderRadius: 4, border: "1px solid rgba(255,255,255,0.1)",
                  background: "transparent", color: "#64748b", fontSize: 10, cursor: "pointer", fontFamily: "inherit",
                }}>Reset</button>
              </div>
            )}
          </div>
        )}

        {tab === "squeeze" && (
          <div>
            <div style={{ fontSize: 12, color: "#94a3b8", marginBottom: 4, fontWeight: 600 }}>
              Quantum Squeezing: Why It Doesn't Help
            </div>
            <div style={{ fontSize: 10, color: "#64748b", marginBottom: 8 }}>
              V_eff = η · V_ideal + (1 − η) · 1 — loss injects vacuum noise that drowns out squeezing
            </div>
            <SqueezePlot width={Math.min(window.innerWidth - 60, 520)} />
            <div style={{
              marginTop: 10, padding: 10, borderRadius: 6,
              background: "rgba(239,68,68,0.08)", border: "1px solid rgba(239,68,68,0.15)",
              fontSize: 11, color: "#f87171",
            }}>
              At 4.5% optical efficiency, 10 dB ideal squeezing yields only{" "}
              <strong>{effectiveSqueezing(0.045, 10).toFixed(2)} dB</strong> effective — negligible.
              You need η {">"} 30% for squeezing to matter.
            </div>
          </div>
        )}
      </div>

      {/* Footer */}
      <div style={{ textAlign: "center", padding: "12px 0 4px", fontSize: 9, color: "#475569" }}>
        Photon budget model from OSIRIS-REx OVIRS/OTES data · K_R scaling: (532/λ)⁴ × resonance
        <br />Built for orbital Raman spectroscopy feasibility assessment
      </div>
    </div>
  );
}
