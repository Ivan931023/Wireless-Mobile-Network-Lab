# Wireless Mobile Network Lab

Simulation project for a **four-layer hexagonal cellular network** with base station sleep scheduling and DTX (Discontinuous Transmission). The goal is to find the optimal trade-off between packet success rate and energy consumption by varying the fraction of active base stations and enabling DTX.

---

## Project Structure

```
Wireless-Mobile-Network-Lab/
└── Four-layer Cellular Base Station Layout Model with DTX/
    ├── Hex_grid.py              # Standalone hexagonal grid visualizer
    ├── Hex_grid.ipynb           # Notebook version of Hex_grid.py
    ├── model.ipynb              # Interactive notebook for model exploration
    ├── model_Shannon.py         # Baseline model using Shannon capacity
    ├── model_MCS.py             # MCS-based model (no DTX)
    ├── model_MCS_DTX.py         # MCS-based model with DTX (scalar loops)
    └── model_MCS_DTX_new.py     # MCS + DTX, vectorized & optimized (recommended)
```

---

## Models Overview

### `model_Shannon.py` — Baseline
The simplest model. Uses Shannon's capacity formula `C = B · log₂(1 + SINR) · η` to compute per-BS throughput. BS activation follows a **ring-based** strategy (nearest BSes are turned on first). No DTX support.

| Parameter | Value |
|---|---|
| λ (packet arrival rate) | 0.1 pkts/ms/UE |
| Timeslot duration | 5 ms |
| SINR threshold | 0 dB |
| Capacity model | Shannon + 0.7 efficiency factor |

### `model_MCS.py` — MCS-Based, No DTX
Replaces Shannon with a **24-entry MCS table** (QPSK → 256-QAM) mapping SINR to discrete spectral efficiencies based on LTE/5G standards. BS activation follows a ring-based strategy. Produces MCS distribution statistics.

| Parameter | Value |
|---|---|
| λ | 0.01 pkts/ms/UE |
| Timeslot duration | 1 ms |
| SINR threshold | −6 dB (MCS 0) |
| Capacity model | MCS table (24 levels) |

### `model_MCS_DTX.py` — MCS + DTX (scalar)
Adds **DTX (Discontinuous Transmission)** to the MCS model. Each BS independently toggles its transmitter off when it has no queued requests, subject to minimum on/off dwell constraints. BS activation is randomized per experiment.

| Parameter | Value |
|---|---|
| λ | 0.05 pkts/ms/UE |
| DTX enabled | True |
| DTX min off slots | 1 |
| DTX min on slots | 1 |

### `model_MCS_DTX_new.py` — MCS + DTX, Vectorized (recommended)
Functionally equivalent to `model_MCS_DTX.py` but **pre-computes the full UE×BS received-power matrix** before the simulation loop, replacing per-UE Python loops with NumPy matrix operations. Significantly faster for large UE/BS counts.

| Parameter | Value |
|---|---|
| λ | 0.006 pkts/ms/UE |
| DTX enabled | False (toggle `ENABLE_DTX = True`) |
| Serving BS selection | Fixed for entire simulation (RSSI-based) |

---

## System Model

### Network Topology

37 base stations arranged in a **4-ring hexagonal grid** (axial radius R = 3) with inter-site distance of 500 m, centered at the origin. The cell layout uses pointy-top hexagons with axial-to-Cartesian conversion:

```
x = 1.5 · s · q
y = √3 · s · (r + 0.5 · q)       where s = RADIUS / √3
```

### Radio Model

**Path loss** (log-distance model):
```
PL(d) = 10 · α · log₁₀(d)    [dB],    α = 3.5
```

**Received power** at UE from BS:
```
P_rx [dBm] = P_tx [dBm] − PL(d)    (P_tx = 43 dBm)
```

**Noise power**:
```
N [dBm] = N₀ [dBm/Hz] + 10·log₁₀(B) + NF    (N₀ = −174, NF = 5 dB, B = 10 MHz)
```

**SINR** (UE connects to strongest BS):
```
SINR = P_signal / (P_interference + N)
```

### Capacity Model (MCS)

24 MCS levels from QPSK to 256-QAM:

| MCS | SINR threshold (dB) | Spectral Efficiency (bits/s/Hz) | Modulation |
|-----|---------------------|--------------------------------|------------|
| 0   | −6.0                | 0.15                           | QPSK       |
| 6   | 6.0                 | 1.48                           | QPSK       |
| 7   | 8.0                 | 1.91                           | 16-QAM     |
| 11  | 13.0                | 3.03                           | 64-QAM     |
| 21  | 24.0                | 5.89                           | 256-QAM    |
| 23  | 28.0                | 7.16                           | 256-QAM    |

**BS capacity** (shared among connected UEs):
```
C_bs = B · mean(SE_i)    for all UEs i assigned to BS
Max packets/slot = floor(C_bs · T_slot / packet_size)
```

### DTX (Discontinuous Transmission)

Each BS tracks desired transmission state per slot. State transitions are hysteresis-controlled:

- **ON → OFF**: requires `DTX_MIN_OFF_SLOTS` consecutive idle slots
- **OFF → ON**: requires `DTX_MIN_ON_SLOTS` consecutive busy slots

Serving BS assignment is fixed for the entire simulation (RSSI-based at start); DTX only affects interference computation.

### Energy Model

Three-level power model:

| State | Power |
|---|---|
| Deep sleep (hardware off) | 10 W |
| DTX idle (transmitter off) | 30 W |
| Active (transmitter on) | 50 + 100 · load W |

With DTX, per-BS power is weighted by transmit duty cycle:
```
P_bs = duty · P_active(load) + (1 − duty) · P_idle
```

---

## Key Metrics

| Metric | Description |
|---|---|
| **Success Rate** | Fraction of arriving packets successfully delivered (SINR above threshold + capacity available) |
| **Power Consumption** | Total network power in Watts |
| **Energy Efficiency** | Success Rate / Power (packets per Joule, normalized) |
| **MCS Distribution** | Histogram of MCS levels used, weighted by packet count |
| **Active BS Ratio** | Fraction of the 37 BSes that are powered on |

---

## Dependencies

```
numpy
scipy
matplotlib
tqdm
```

Install with:
```bash
pip install numpy scipy matplotlib tqdm
```

---

## How to Run

```bash
cd "Four-layer Cellular Base Station Layout Model with DTX"

# Baseline (Shannon capacity)
python model_Shannon.py

# MCS-based model without DTX
python model_MCS.py

# MCS-based model with DTX (scalar)
python model_MCS_DTX.py

# Vectorized MCS + DTX model (recommended, fastest)
python model_MCS_DTX_new.py
```

Each script:
1. Generates and plots the hexagonal BS layout
2. Generates and plots random UE positions
3. Plots RSSI and SINR heat maps
4. Runs Monte Carlo simulation over 10 active-BS ratio levels (10% → 100%), 10 experiments each
5. Plots success rate, power consumption, MCS distribution, and energy efficiency
6. Prints a decision analysis identifying the most energy-efficient operating point that meets a 90% success rate QoS requirement

### Key Parameters to Tune

In the `SystemParameters` class at the top of each file:

```python
params.LAMBDA               # Packet arrival rate (pkts/ms/UE)
params.UE_COUNT             # Number of users
params.ENABLE_DTX           # Toggle DTX on/off (DTX files only)
params.DTX_MIN_OFF_SLOTS    # Min consecutive idle slots to switch TX off
params.DTX_MIN_ON_SLOTS     # Min consecutive busy slots to switch TX on
params.CAPACITY_MODE        # 'share' | 'avg' | 'sum'
params.SHARE_OVER           # 'all' | 'busy' (who shares bandwidth)
```

---

## Simulation Flow

```
For each active_ratio in [0.1, 0.2, ..., 1.0]:
  Randomly select n_active BSes
  For each of 10 experiments:
    Generate random UE positions
    Pre-compute UE×BS received power matrix
    Fix UE → BS serving assignment (strongest RSSI)
    For each timeslot t = 1..1000:
      Generate Poisson packet arrivals per UE
      [DTX] Decide per-BS TX state based on request load
      Compute SINR for all UEs (vectorized)
      Map SINR → MCS → spectral efficiency
      Compute per-BS capacity and allocate packets
      Count successful transmissions
  Average results across experiments
Report best operating point
```

---

## Results Interpretation

The four output plots show:

- **Success Rate vs Active BS Ratio**: Higher ratios generally give better coverage; diminishing returns past ~0.7.
- **Power vs Active BS Ratio**: Nearly linear increase; each inactive BS saves ~40 W (active baseline minus sleep).
- **MCS Distribution**: Indicates link quality distribution. High MCS usage means good SINR; large "NC" bar means poor coverage.
- **Energy Efficiency**: Typically peaks at intermediate ratios where coverage is adequate but fewer BSes are on.
