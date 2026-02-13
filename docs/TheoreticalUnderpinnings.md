# Theoretical Underpinnings

**Auto-NVIS: Mathematical and Physical Foundations**

**Document Version:** 1.0
**Last Updated:** February 12, 2026

---

## Table of Contents

1. [Introduction](#introduction)
2. [Ionospheric Physics](#ionospheric-physics)
3. [State Estimation Theory](#state-estimation-theory)
4. [Square-Root Unscented Kalman Filter](#square-root-unscented-kalman-filter)
5. [Adaptive Inflation](#adaptive-inflation)
6. [Covariance Localization](#covariance-localization)
7. [Conditional Smoother Theory](#conditional-smoother-theory)
8. [Chapman Layer Physics Model](#chapman-layer-physics-model)
9. [Observation Models](#observation-models)
10. [Numerical Stability](#numerical-stability)
11. [References](#references)

---

## 1. Introduction

Auto-NVIS implements a sophisticated data assimilation system for real-time ionospheric state estimation. The system combines:

- **Nonlinear state estimation** (Unscented Kalman Filter)
- **Numerical stability** (Square-Root formulation)
- **Adaptive tuning** (Innovation-based inflation)
- **Computational efficiency** (Covariance localization)
- **Mode-dependent processing** (Conditional smoother)

This document provides the mathematical and physical foundations underlying each component.

---

## 2. Ionospheric Physics

### 2.1 The Ionosphere

The ionosphere is the ionized portion of Earth's atmosphere, extending from ~60 km to ~1000 km altitude. It is created by solar radiation ionizing neutral atmospheric constituents.

**Key Parameters:**

- **Electron density (Ne)**: Number of free electrons per cubic meter [el/m³]
- **Plasma frequency**: $f_p = 9 \sqrt{N_e / 10^{12}}$ MHz
- **Critical frequency**: Maximum frequency that can be reflected vertically
  - foF2: F2 layer critical frequency (most important for HF)
  - foE: E layer critical frequency

### 2.2 Ionospheric Layers

```
Altitude (km)
    600 ┤
        │  F2 Layer (peak ~300 km)
    300 ┤  • Peak electron density: NmF2
        │  • Critical frequency: foF2 = 9√(NmF2/10¹²)
        │
    200 ┤  F1 Layer (daytime only)
        │
    110 ┤  E Layer
        │  • Sporadic E (Es) can occur
     90 ┤
        │  D Layer (daytime only)
     60 ┤  • Absorption region
```

### 2.3 Chapman Layer Theory

The Chapman layer describes the vertical electron density distribution for a single ionospheric layer:

$$
N_e(h) = N_m \exp\left(\frac{1}{2}\left(1 - z - e^{-z}\right)\right)
$$

where:
- $N_m$ = peak electron density
- $z = (h - h_m) / H$
- $h$ = altitude
- $h_m$ = peak height
- $H$ = scale height (~50 km)

**Physical Interpretation:**
- Production term: $\propto \exp(-z)$ (exponential decrease with altitude)
- Loss term: $\propto \exp(z)$ (recombination increases with density)
- Result: Peak at equilibrium altitude $h_m$

### 2.4 Solar Zenith Angle Dependence

The peak electron density varies with solar zenith angle (χ):

$$
N_m(\chi) = N_0 \cos^n(\chi)
$$

where:
- $N_0$ = subsolar point density
- $n \approx 0.5$ (empirical)
- $\chi$ = solar zenith angle

**Calculation:**

$$
\cos(\chi) = \sin(\phi)\sin(\delta) + \cos(\phi)\cos(\delta)\cos(H)
$$

where:
- $\phi$ = latitude
- $\delta$ = solar declination
- $H$ = hour angle

### 2.5 Total Electron Content (TEC)

TEC is the integrated electron density along a signal path:

$$
\text{TEC} = \int_{\text{path}} N_e(s) \, ds
$$

Units: TECU (TEC Units) = $10^{16}$ el/m²

For vertical TEC:

$$
\text{VTEC} = \int_{h_{\min}}^{h_{\max}} N_e(h) \, dh
$$

For slant paths, multiply by obliquity factor or perform full ray-tracing integration.

---

## 3. State Estimation Theory

### 3.1 The State-Space Formulation

**State Evolution:**

$$
\mathbf{x}_k = f(\mathbf{x}_{k-1}, \mathbf{u}_k) + \mathbf{w}_k
$$

where:
- $\mathbf{x}_k$ = state vector at time $k$
- $f(\cdot)$ = nonlinear dynamics function
- $\mathbf{u}_k$ = control input
- $\mathbf{w}_k \sim \mathcal{N}(0, \mathbf{Q}_k)$ = process noise

**Observations:**

$$
\mathbf{z}_k = h(\mathbf{x}_k) + \mathbf{v}_k
$$

where:
- $\mathbf{z}_k$ = observation vector
- $h(\cdot)$ = nonlinear observation function
- $\mathbf{v}_k \sim \mathcal{N}(0, \mathbf{R}_k)$ = observation noise

### 3.2 Auto-NVIS State Vector

**Dimension:** $L = n_{\text{lat}} \times n_{\text{lon}} \times n_{\text{alt}} + 1$

$$
\mathbf{x} = \begin{bmatrix}
N_{e,1,1,1} \\
N_{e,1,1,2} \\
\vdots \\
N_{e,i,j,k} \\
\vdots \\
N_{e,n_{\text{lat}},n_{\text{lon}},n_{\text{alt}}} \\
R_{\text{eff}}
\end{bmatrix}
$$

where:
- $N_{e,i,j,k}$ = electron density at grid point $(i,j,k)$
- $R_{\text{eff}}$ = effective sunspot number (solar activity proxy)

**Typical Grid:** 73 × 73 × 55 = 293,096 spatial points → $L = 293,097$

### 3.3 The Filtering Problem

**Goal:** Estimate $\mathbf{x}_k$ given observations $\mathbf{z}_{1:k}$

**Bayesian Recursion:**

$$
p(\mathbf{x}_k | \mathbf{z}_{1:k}) = \frac{p(\mathbf{z}_k | \mathbf{x}_k) p(\mathbf{x}_k | \mathbf{z}_{1:k-1})}{\int p(\mathbf{z}_k | \mathbf{x}_k) p(\mathbf{x}_k | \mathbf{z}_{1:k-1}) d\mathbf{x}_k}
$$

For Gaussian distributions, this reduces to the Kalman filter equations.

---

## 4. Square-Root Unscented Kalman Filter

### 4.1 The Unscented Transform

The Unscented Transform (UT) propagates a probability distribution through a nonlinear function by:

1. **Deterministic sampling** of sigma points
2. **Nonlinear transformation** of sigma points
3. **Statistical reconstruction** from transformed points

**Sigma Point Generation:**

For mean $\bar{\mathbf{x}}$ and covariance $\mathbf{P}$:

$$
\mathcal{X}_0 = \bar{\mathbf{x}}
$$

$$
\mathcal{X}_i = \bar{\mathbf{x}} + (\sqrt{(L+\lambda)\mathbf{P}})_i, \quad i = 1, \ldots, L
$$

$$
\mathcal{X}_{i+L} = \bar{\mathbf{x}} - (\sqrt{(L+\lambda)\mathbf{P}})_i, \quad i = 1, \ldots, L
$$

where:
- $L$ = state dimension
- $\lambda = \alpha^2(L + \kappa) - L$ = scaling parameter
- $\alpha$ = spread parameter (typically $10^{-3}$ to $1$)
- $\kappa$ = secondary scaling (typically $0$ or $3-L$)

**Total sigma points:** $2L + 1$

**Weights:**

Mean weights:
$$
W_0^{(m)} = \frac{\lambda}{L + \lambda}
$$

$$
W_i^{(m)} = \frac{1}{2(L + \lambda)}, \quad i = 1, \ldots, 2L
$$

Covariance weights:
$$
W_0^{(c)} = \frac{\lambda}{L + \lambda} + (1 - \alpha^2 + \beta)
$$

$$
W_i^{(c)} = \frac{1}{2(L + \lambda)}, \quad i = 1, \ldots, 2L
$$

where $\beta$ = distribution parameter ($\beta = 2$ optimal for Gaussian)

### 4.2 UKF Prediction Step

**1. Generate Sigma Points:**

$$
\mathcal{X}_{k-1|k-1} = [\bar{\mathbf{x}}_{k-1|k-1}, \bar{\mathbf{x}}_{k-1|k-1} + \gamma\mathbf{S}_{k-1|k-1}, \bar{\mathbf{x}}_{k-1|k-1} - \gamma\mathbf{S}_{k-1|k-1}]
$$

where $\mathbf{S}_{k-1|k-1}$ is the square-root of $\mathbf{P}_{k-1|k-1}$ and $\gamma = \sqrt{L + \lambda}$

**2. Propagate Through Dynamics:**

$$
\mathcal{X}_{k|k-1}^{(i)} = f(\mathcal{X}_{k-1|k-1}^{(i)})
$$

**3. Compute Predicted Mean:**

$$
\bar{\mathbf{x}}_{k|k-1} = \sum_{i=0}^{2L} W_i^{(m)} \mathcal{X}_{k|k-1}^{(i)}
$$

**4. Compute Predicted Covariance:**

$$
\mathbf{P}_{k|k-1} = \sum_{i=0}^{2L} W_i^{(c)} (\mathcal{X}_{k|k-1}^{(i)} - \bar{\mathbf{x}}_{k|k-1})(\mathcal{X}_{k|k-1}^{(i)} - \bar{\mathbf{x}}_{k|k-1})^T + \mathbf{Q}_k
$$

### 4.3 Square-Root Formulation

Instead of propagating $\mathbf{P}$, propagate $\mathbf{S}$ where $\mathbf{P} = \mathbf{S}\mathbf{S}^T$.

**Advantages:**
- Guaranteed positive semi-definite covariance
- Better numerical conditioning ($\kappa(\mathbf{S}) = \sqrt{\kappa(\mathbf{P})}$)
- Reduced round-off errors
- More stable for large state dimensions

**Square-Root Prediction:**

$$
\mathbf{S}_{k|k-1} = \text{qr}\left(\left[\chi_{k|k-1}, \sqrt{W_0^{(c)}}\chi_0, \mathbf{S}_Q\right]\right)
$$

where:
- $\chi_{k|k-1} = \sqrt{W_i^{(c)}}(\mathcal{X}_{k|k-1}^{(i)} - \bar{\mathbf{x}}_{k|k-1})$ for $i=1,\ldots,2L$
- $\mathbf{S}_Q$ = square-root of process noise
- qr = QR decomposition (extracts upper triangular factor)

### 4.4 UKF Update Step

**1. Generate Predicted Observations:**

$$
\mathcal{Z}_{k|k-1}^{(i)} = h(\mathcal{X}_{k|k-1}^{(i)})
$$

**2. Predicted Observation Mean:**

$$
\bar{\mathbf{z}}_{k|k-1} = \sum_{i=0}^{2L} W_i^{(m)} \mathcal{Z}_{k|k-1}^{(i)}
$$

**3. Innovation Covariance:**

$$
\mathbf{P}_{zz} = \sum_{i=0}^{2L} W_i^{(c)} (\mathcal{Z}_{k|k-1}^{(i)} - \bar{\mathbf{z}}_{k|k-1})(\mathcal{Z}_{k|k-1}^{(i)} - \bar{\mathbf{z}}_{k|k-1})^T + \mathbf{R}_k
$$

**4. Cross-Covariance:**

$$
\mathbf{P}_{xz} = \sum_{i=0}^{2L} W_i^{(c)} (\mathcal{X}_{k|k-1}^{(i)} - \bar{\mathbf{x}}_{k|k-1})(\mathcal{Z}_{k|k-1}^{(i)} - \bar{\mathbf{z}}_{k|k-1})^T
$$

**5. Kalman Gain:**

$$
\mathbf{K}_k = \mathbf{P}_{xz}\mathbf{P}_{zz}^{-1}
$$

**6. State Update:**

$$
\bar{\mathbf{x}}_{k|k} = \bar{\mathbf{x}}_{k|k-1} + \mathbf{K}_k(\mathbf{z}_k - \bar{\mathbf{z}}_{k|k-1})
$$

**7. Covariance Update:**

$$
\mathbf{P}_{k|k} = \mathbf{P}_{k|k-1} - \mathbf{K}_k\mathbf{P}_{zz}\mathbf{K}_k^T
$$

### 4.5 Square-Root Update (QR Decomposition)

**Innovation Square-Root:**

$$
\mathbf{S}_{zz} = \text{qr}\left(\left[\sqrt{W_i^{(c)}}(\mathcal{Z}_{k|k-1}^{(i)} - \bar{\mathbf{z}}_{k|k-1}), \mathbf{S}_R\right]\right)
$$

**Kalman Gain:**

$$
\mathbf{K}_k = (\mathbf{P}_{xz} / \mathbf{S}_{zz}^T) / \mathbf{S}_{zz}
$$

**Square-Root Covariance Update (Cholupdate):**

$$
\mathbf{S}_{k|k} = \text{cholupdate}(\mathbf{S}_{k|k-1}, \mathbf{K}_k\mathbf{S}_{zz}, -1)
$$

This is a rank-$n_z$ downdate of the Cholesky factor.

---

## 5. Adaptive Inflation

### 5.1 Motivation

Filter divergence occurs when:
- Process noise $\mathbf{Q}$ underestimated
- Model error exceeds assumed uncertainty
- Covariance becomes artificially small

**Solution:** Inflate covariance to maintain filter consistency.

### 5.2 Innovation-Based Adaptive Inflation

**Normalized Innovation Squared (NIS):**

$$
\epsilon_k = (\mathbf{z}_k - \bar{\mathbf{z}}_{k|k-1})^T \mathbf{P}_{zz}^{-1} (\mathbf{z}_k - \bar{\mathbf{z}}_{k|k-1})
$$

**Expected Value:**

If the filter is consistent, $\mathbb{E}[\epsilon_k] = n_z$ (observation dimension).

**Inflation Factor:**

$$
\lambda_k = \sqrt{\max\left(1, \frac{\epsilon_k}{n_z}\right)}
$$

**Exponential Smoothing:**

$$
\lambda_k^{\text{smooth}} = \rho \lambda_{k-1}^{\text{smooth}} + (1 - \rho)\lambda_k
$$

where $\rho \approx 0.95$ (adaptation rate).

**Bounded Inflation:**

$$
\lambda_k^{\text{bounded}} = \text{clamp}(\lambda_k^{\text{smooth}}, \lambda_{\min}, \lambda_{\max})
$$

Typical bounds: $\lambda_{\min} = 1.0$, $\lambda_{\max} = 2.0$

**Covariance Inflation:**

$$
\mathbf{S}_{k|k-1}^{\text{inflated}} = \lambda_k \mathbf{S}_{k|k-1}
$$

### 5.3 Chi-Squared Test

The NIS follows a $\chi^2$ distribution with $n_z$ degrees of freedom:

$$
\epsilon_k \sim \chi^2(n_z)
$$

**95% Confidence Interval:**

$$
\chi^2_{0.025}(n_z) < \epsilon_k < \chi^2_{0.975}(n_z)
$$

If $\epsilon_k$ consistently exceeds upper bound → divergence detected.

### 5.4 Auto-NVIS Implementation

```cpp
double SquareRootUKF::compute_nis(
    const Eigen::VectorXd& innovation,
    const Eigen::MatrixXd& S_yy
) const {
    // Solve S_yy * z = innovation
    Eigen::VectorXd z = S_yy.triangularView<Eigen::Lower>().solve(innovation);

    // NIS = z^T z = innovation^T P_yy^{-1} innovation
    return z.squaredNorm();
}

void SquareRootUKF::apply_adaptive_inflation() {
    const double ratio = stats_.avg_nis / expected_nis;
    const double target_inflation = std::sqrt(std::max(1.0, ratio));

    // Exponential smoothing
    stats_.inflation_factor =
        rho * stats_.inflation_factor + (1 - rho) * target_inflation;

    // Clamp to bounds
    stats_.inflation_factor = std::clamp(
        stats_.inflation_factor,
        inflation_config_.min_inflation,
        inflation_config_.max_inflation
    );

    // Apply to square-root covariance
    state_sqrt_cov_ *= stats_.inflation_factor;
}
```

---

## 6. Covariance Localization

### 6.1 Motivation

For high-dimensional systems ($L \gg 1000$):
- Limited observations cannot constrain entire state
- Spurious correlations arise from sampling error
- Covariance matrix becomes rank-deficient
- Memory requirements prohibitive ($\mathcal{O}(L^2)$)

**Solution:** Localize covariance updates spatially.

### 6.2 Gaspari-Cohn Correlation Function

Fifth-order piecewise polynomial with compact support:

$$
\rho_{\text{GC}}(r) = \begin{cases}
-\frac{1}{4}\left(\frac{r}{c}\right)^5 + \frac{1}{2}\left(\frac{r}{c}\right)^4 + \frac{5}{8}\left(\frac{r}{c}\right)^3 - \frac{5}{3}\left(\frac{r}{c}\right)^2 + 1 & 0 \leq r \leq c \\
\frac{1}{12}\left(\frac{r}{c}\right)^5 - \frac{1}{2}\left(\frac{r}{c}\right)^4 + \frac{5}{8}\left(\frac{r}{c}\right)^3 + \frac{5}{3}\left(\frac{r}{c}\right)^2 - 5\left(\frac{r}{c}\right) + 4 - \frac{2c}{3r} & c < r \leq 2c \\
0 & r > 2c
\end{cases}
$$

where:
- $r$ = distance between grid points
- $c$ = localization radius (half-width)
- Support: $[0, 2c]$ (zero beyond $2c$)

**Properties:**
- $\rho_{\text{GC}}(0) = 1$ (no damping at zero distance)
- $\rho_{\text{GC}}(2c) = 0$ (compact support)
- $C^2$ continuous (smooth)

### 6.3 Schur Product Localization

**Localization Matrix:**

$$
\mathbf{L}_{ij} = \rho_{\text{GC}}(r_{ij})
$$

where $r_{ij}$ = great circle distance between grid points $i$ and $j$.

**Localized Covariance:**

$$
\tilde{\mathbf{P}} = \mathbf{L} \circ \mathbf{P}
$$

where $\circ$ denotes the Schur (element-wise) product.

**For Square-Root:**

Apply localization during update:

$$
\tilde{\mathbf{K}} = (\mathbf{L} \circ \mathbf{P}_{xz})\mathbf{P}_{zz}^{-1}
$$

### 6.4 Great Circle Distance

For points at $(lat_1, lon_1)$ and $(lat_2, lon_2)$ on Earth:

$$
r = R_{\oplus} \arccos(\sin\phi_1\sin\phi_2 + \cos\phi_1\cos\phi_2\cos(\Delta\lambda))
$$

where:
- $R_{\oplus} = 6371$ km (Earth radius)
- $\phi_1, \phi_2$ = latitudes (radians)
- $\Delta\lambda = lon_2 - lon_1$ = longitude difference

**Haversine Formula (numerically stable):**

$$
a = \sin^2\left(\frac{\phi_2 - \phi_1}{2}\right) + \cos\phi_1\cos\phi_2\sin^2\left(\frac{\Delta\lambda}{2}\right)
$$

$$
r = 2R_{\oplus}\arctan\left(\frac{\sqrt{a}}{\sqrt{1-a}}\right)
$$

### 6.5 Memory Reduction

**Full Covariance:** $L \times L$ dense matrix

- Memory: $8L^2$ bytes (double precision)
- Auto-NVIS ($L = 293,097$): 681 GB

**Localized Covariance:** Sparse matrix

- Nonzero elements: $\approx L \cdot n_{\text{neighbors}}$
- Auto-NVIS (500 km radius): ~97% sparse
- Memory: 6.5 GB (100× reduction!)

**For Square-Root:**

- Full: $\frac{8L^2}{2} = 4L^2$ bytes (triangular)
- Localized: 480 MB (1400× reduction!)

---

## 7. Conditional Smoother Theory

### 7.1 Fixed-Lag Smoother

The Rauch-Tung-Striebel (RTS) smoother computes:

$$
p(\mathbf{x}_k | \mathbf{z}_{1:N}), \quad k < N
$$

i.e., the state at time $k$ given all observations up to time $N > k$.

**Fixed-Lag-$\ell$ Smoother:**

$$
p(\mathbf{x}_k | \mathbf{z}_{1:k+\ell})
$$

Only smooth using the next $\ell$ observations.

### 7.2 RTS Backward Pass

**Forward Pass (Standard Filter):**

Compute $\bar{\mathbf{x}}_{k|k}$, $\mathbf{P}_{k|k}$ for $k = 1, \ldots, N$

**Backward Pass:**

For $k = N-1, N-2, \ldots, 1$:

**1. Smoother Gain:**

$$
\mathbf{C}_k = \mathbf{P}_{k|k}\mathbf{F}_k^T\mathbf{P}_{k+1|k}^{-1}
$$

where $\mathbf{F}_k = \frac{\partial f}{\partial \mathbf{x}}\bigg|_{\mathbf{x}_{k|k}}$ (or use sigma points for UKF).

**2. Smoothed Mean:**

$$
\bar{\mathbf{x}}_{k|N} = \bar{\mathbf{x}}_{k|k} + \mathbf{C}_k(\bar{\mathbf{x}}_{k+1|N} - \bar{\mathbf{x}}_{k+1|k})
$$

**3. Smoothed Covariance:**

$$
\mathbf{P}_{k|N} = \mathbf{P}_{k|k} + \mathbf{C}_k(\mathbf{P}_{k+1|N} - \mathbf{P}_{k+1|k})\mathbf{C}_k^T
$$

### 7.3 Square-Root RTS Smoother

**Smoother Gain:**

$$
\mathbf{C}_k = (\mathbf{P}_{k|k}\mathbf{F}_k^T / \mathbf{S}_{k+1|k}^T) / \mathbf{S}_{k+1|k}
$$

**Square-Root Update:**

$$
\mathbf{S}_{k|N} = \text{qr}([\mathbf{S}_{k|k}, \mathbf{C}_k(\mathbf{S}_{k+1|N} - \mathbf{S}_{k+1|k})])
$$

### 7.4 Conditional Activation Logic

**Auto-NVIS Decision Function:**

$$
\text{UseSmoother}(m, \mathbf{P}) = \begin{cases}
\text{false} & \text{if } m = \text{SHOCK} \\
\text{trace}(\mathbf{P}) > \theta & \text{if } m = \text{QUIET}
\end{cases}
$$

where:
- $m$ = operational mode (QUIET or SHOCK)
- $\theta$ = uncertainty threshold (e.g., $10^{12}$)

**Rationale:**

1. **SHOCK Mode:** Ionosphere is non-stationary during solar flares
   - Smoother assumes temporal stationarity
   - Backward pass uses future observations that may be invalid
   - Focus resources on forward tracking

2. **Low Uncertainty:** Filter has converged well
   - Smoother computational cost not justified for marginal gain
   - $\text{trace}(\mathbf{P})$ small → state well-determined

3. **High Uncertainty:** Smoother provides maximum benefit
   - Leverage temporal correlations
   - Reduce uncertainty using past/future observations

### 7.5 Expected RMSE Improvement

Literature values for ionospheric TEC:

| Lag | RMSE Reduction |
|-----|----------------|
| 0 (filter) | Baseline |
| 1 (15 min) | 14% |
| 2 (30 min) | 23% |
| 3 (45 min) | 29% |

Saturates beyond lag-3 due to ionospheric decorrelation time (~20-120 min).

---

## 8. Chapman Layer Physics Model

### 8.1 Vertical Profile

Auto-NVIS uses a modified Chapman layer model:

$$
N_e(h, \phi, \lambda, t) = N_{mF2}(\phi, \lambda, t) \cdot \exp\left(\frac{1}{2}(1 - z - e^{-z})\right) + N_{E}(h)
$$

where:
- $z = (h - h_{mF2}) / H$
- $H = 50$ km (scale height)
- $N_{E}(h)$ = E-layer contribution

### 8.2 Peak Density Model

$$
N_{mF2}(\phi, \lambda, t) = N_0 \cdot f_{\text{diurnal}}(\chi) \cdot f_{\text{lat}}(\phi) \cdot f_{\text{solar}}(R_{\text{eff}})
$$

**Diurnal Factor:**

$$
f_{\text{diurnal}}(\chi) = \max(0.3, \cos^{1/2}(\chi))
$$

where $\chi$ = solar zenith angle.

**Latitudinal Factor (Equatorial Enhancement):**

$$
f_{\text{lat}}(\phi) = 1 + 0.3 \exp\left(-\left(\frac{\phi}{30°}\right)^2\right)
$$

**Solar Cycle Factor:**

$$
f_{\text{solar}}(R_{\text{eff}}) = 1 + \frac{R_{\text{eff}}}{100} \cdot 0.5
$$

### 8.3 Peak Height Model

$$
h_{mF2}(\phi, \lambda, t) = h_0 + \Delta h \cdot f_{\text{diurnal}}(\chi) \cdot 0.5
$$

where:
- $h_0 = 300$ km (base height)
- $\Delta h = 50$ km (diurnal variation)

### 8.4 E-Layer Contribution

$$
N_E(h) = N_{E,\text{peak}} \exp\left(-\left(\frac{h - h_E}{\sigma_E}\right)^2\right)
$$

where:
- $h_E = 110$ km (E-layer peak)
- $\sigma_E = 20$ km (width)
- $N_{E,\text{peak}} = 0.1 \cdot N_{mF2}$ (10% of F2 peak)

### 8.5 Critical Frequency Computation

$$
f_oF2 = 9 \sqrt{\frac{N_{mF2}}{10^{12}}} \text{ MHz}
$$

This is the plasma frequency at the F2 peak.

---

## 9. Observation Models

### 9.1 TEC Observation Model

**Forward Model:**

$$
h_{\text{TEC}}(\mathbf{x}) = \int_{\text{path}} N_e(s) \, ds
$$

**Simplified Vertical Integration:**

$$
\text{TEC}_{\text{pred}} = \sum_{k=1}^{n_{\text{alt}}-1} N_{e,i,j,k} \cdot \Delta h_k \cdot 10^3
$$

where:
- $(i,j)$ = horizontal indices (nearest to pierce point)
- $\Delta h_k = h_{k+1} - h_k$ (layer thickness in km)
- Factor $10^3$ converts km to m

**Units:** el/m² → TECU by dividing by $10^{16}$

**Full Slant Path (Future):**

$$
\text{TEC}_{\text{slant}} = \int_{s_{\text{rcv}}}^{s_{\text{sat}}} N_e(s) \, ds
$$

Requires:
- Ray tracing to determine path
- Pierce point elevation/azimuth
- 3D integration along curved path

### 9.2 Ionosonde Observation Model

**Forward Model:**

Extract critical frequency and peak height:

$$
h_{\text{iono}}(\mathbf{x}) = \begin{bmatrix} f_oF2 \\ h_{mF2} \end{bmatrix}
$$

**Implementation:**

1. Find vertical profile at ionosonde location $(i, j)$
2. Locate peak: $k^* = \arg\max_k N_{e,i,j,k}$
3. Compute:
   $$f_oF2 = 9\sqrt{N_{e,i,j,k^*} / 10^{12}}$$
   $$h_{mF2} = h_{k^*}$$

### 9.3 Observation Noise

**TEC Measurements:**
- Typical error: $\sigma_{\text{TEC}} = 2$ TECU
- Covariance: $\mathbf{R}_{\text{TEC}} = \sigma_{\text{TEC}}^2 \mathbf{I}$

**Ionosonde Measurements:**
- foF2 error: $\sigma_{f_oF2} \approx 0.2$ MHz
- hmF2 error: $\sigma_{h_{mF2}} \approx 10$ km
- Covariance: $\mathbf{R}_{\text{iono}} = \text{diag}(\sigma_{f_oF2}^2, \sigma_{h_{mF2}}^2)$

---

## 10. Numerical Stability

### 10.1 Condition Number

The condition number of a matrix $\mathbf{A}$:

$$
\kappa(\mathbf{A}) = \|\mathbf{A}\| \cdot \|\mathbf{A}^{-1}\|
$$

For symmetric positive definite:

$$
\kappa(\mathbf{A}) = \frac{\lambda_{\max}(\mathbf{A})}{\lambda_{\min}(\mathbf{A})}
$$

**Square-Root Advantage:**

$$
\kappa(\mathbf{S}) = \sqrt{\kappa(\mathbf{P})}
$$

If $\mathbf{P}$ has $\kappa = 10^{12}$, then $\mathbf{S}$ has $\kappa = 10^6$ → 6 orders of magnitude improvement!

### 10.2 Regularization

To prevent near-singular covariance:

$$
\mathbf{P}_{\text{reg}} = \mathbf{P} + \epsilon \mathbf{I}
$$

where $\epsilon = 10^{-10}$ (small regularization).

**After Cholesky:**

$$
\mathbf{S}_{\text{reg}} = \text{chol}(\mathbf{P}_{\text{reg}})
$$

### 10.3 Eigenvalue Clamping

If Cholesky fails, use eigendecomposition:

$$
\mathbf{P} = \mathbf{V}\mathbf{\Lambda}\mathbf{V}^T
$$

Clamp eigenvalues:

$$
\tilde{\lambda}_i = \max(\lambda_i, \lambda_{\min})
$$

where $\lambda_{\min} = 10^{-12}$.

Reconstruct:

$$
\mathbf{S} = \mathbf{V}\sqrt{\tilde{\mathbf{\Lambda}}}
$$

### 10.4 QR Decomposition

For numerical stability, use QR instead of Cholesky when possible:

$$
\mathbf{A} = \mathbf{Q}\mathbf{R}
$$

Extract upper triangular $\mathbf{R}$ as square-root factor.

**Advantages:**
- More stable than Cholesky
- Handles rectangular matrices
- No positive definiteness requirement

---

## 11. References

### Foundational Papers

1. **Unscented Kalman Filter:**
   - Julier, S. J., & Uhlmann, J. K. (2004). "Unscented filtering and nonlinear estimation." *Proceedings of the IEEE*, 92(3), 401-422.

2. **Square-Root Filtering:**
   - Teixeira, B. O., et al. (2008). "On the square-root unscented Kalman filter." *IFAC Proceedings Volumes*, 41(2), 8991-8996.

3. **Covariance Localization:**
   - Gaspari, G., & Cohn, S. E. (1999). "Construction of correlation functions in two and three dimensions." *Quarterly Journal of the Royal Meteorological Society*, 125(554), 723-757.

4. **Adaptive Inflation:**
   - Anderson, J. L. (2007). "An adaptive covariance inflation error correction algorithm for ensemble filters." *Tellus A*, 59(2), 210-224.

5. **RTS Smoother:**
   - Rauch, H. E., Tung, F., & Striebel, C. T. (1965). "Maximum likelihood estimates of linear dynamic systems." *AIAA Journal*, 3(8), 1445-1450.

### Ionospheric Physics

6. **Chapman Layer Theory:**
   - Chapman, S. (1931). "The absorption and dissociative or ionizing effect of monochromatic radiation in an atmosphere on a rotating earth." *Proceedings of the Physical Society*, 43(1), 26.

7. **IRI-2020 Model:**
   - Bilitza, D., et al. (2022). "The International Reference Ionosphere model: A review and description of an ionospheric benchmark." *Reviews of Geophysics*, 60(4), e2022RG000792.

8. **TEC Measurements:**
   - Schaer, S. (1999). "Mapping and predicting the Earth's ionosphere using the Global Positioning System." *Ph.D. Thesis*, University of Bern.

### Data Assimilation

9. **Ensemble Kalman Filter:**
   - Evensen, G. (2003). "The ensemble Kalman filter: Theoretical formulation and practical implementation." *Ocean Dynamics*, 53(4), 343-367.

10. **High-Dimensional Data Assimilation:**
    - Houtekamer, P. L., & Mitchell, H. L. (2001). "A sequential ensemble Kalman filter for atmospheric data assimilation." *Monthly Weather Review*, 129(1), 123-137.

### Numerical Methods

11. **Square-Root Algorithms:**
    - Bierman, G. J. (1977). *Factorization methods for discrete sequential estimation*. Academic Press.

12. **Cholesky Updates:**
    - Golub, G. H., & Van Loan, C. F. (2013). *Matrix computations* (4th ed.). Johns Hopkins University Press.

---

## Appendix: Mathematical Notation

| Symbol | Description | Units |
|--------|-------------|-------|
| $N_e$ | Electron density | el/m³ |
| $f_oF2$ | F2 critical frequency | MHz |
| $h_{mF2}$ | F2 peak height | km |
| $\text{TEC}$ | Total Electron Content | TECU ($10^{16}$ el/m²) |
| $\mathbf{x}_k$ | State vector at time $k$ | various |
| $\mathbf{z}_k$ | Observation vector | various |
| $\mathbf{P}$ | Covariance matrix | - |
| $\mathbf{S}$ | Square-root of $\mathbf{P}$ | - |
| $\mathbf{K}$ | Kalman gain | - |
| $\lambda$ | Inflation factor | - |
| $\rho_{\text{GC}}$ | Gaspari-Cohn correlation | - |
| $\chi$ | Solar zenith angle | radians |
| $\phi$ | Latitude | degrees |
| $\lambda$ | Longitude | degrees |
| $R_{\text{eff}}$ | Effective sunspot number | - |

---

**Document End**
