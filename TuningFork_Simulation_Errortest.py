import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from TuningFork_Simulation_Function import calcX
import TuningFork_Simulation_Parameters as TP

# --- User settings ----------------------------------------------------------

i = 1
mode = 'Normal'   # 'Normal' or 'Shear'
eta = 0

# Build the filename based on how you named your output in Scan()
fname = f"Simulation_Scan_{eta}_{mode}_i_{i}_Number_of_D_{len(TP.D)}_of_F_{len(TP.Fn)}.TXT"

# --- 1) Load data -----------------------------------------------------------

# Read CSV, stripping any spaces after commas in the header
df = pd.read_csv(fname, skipinitialspace=True)

# Check your columns:
# print(df.columns.tolist())
# Expect: ['D [mm]', 'Frequency[Hz]', 'test Frequency', 'Amplitude Scan', 'Quality factor']

# --- 2) Define the analytic model ------------------------------------------

# Resonant angular frequency (normal mode)
omega0 = 2 * np.pi * TP.Fn_0

def A_model(omega, C, gamma_sys):
    """
    Analytical amplitude vs. ω for a driven damped harmonic oscillator
      A(ω) = C / sqrt(ω0^4 - 2 ω0^2 ω^2 + ω^4 + γ_sys^2 ω^2)
    """
    denom = np.sqrt(
        omega0**4
        - 2 * omega0**2 * omega**2
        + omega**4
        + gamma_sys**2 * omega**2
    )
    return C / denom

# --- 3) Fit each distance ---------------------------------------------------

results = []
for d in sorted(df['D [mm]'].unique()):
    sub = df[df['D [mm]'] == d]
    # Convert drive frequency (Hz) to angular frequency (rad/s)
    omega = 2 * np.pi * sub['Frequency[Hz]'].values
    A_obs = sub['Amplitude Scan'].values

    # Initial guesses
    C0_guess     = np.max(A_obs) * TP.gamma_sys * omega0
    gamma0_guess = TP.gamma_sys
    p0 = [C0_guess, gamma0_guess]

    # Bound gamma_sys >= 0, C >= 0
    bounds = ([0, 0], [np.inf, np.inf])

    popt, pcov = curve_fit(A_model, omega, A_obs, p0=p0, bounds=bounds)
    C_fit, gamma_fit = popt
    sigma_C, sigma_g = np.sqrt(np.diag(pcov))

    results.append((d, C_fit, gamma_fit, sigma_C, sigma_g))

    print(f"D = {d:.3f} mm:")
    print(f"  C_fit       = {C_fit:.3e} ± {sigma_C:.3e}")
    print(f"  gamma_sys   = {gamma_fit:.3f} ± {sigma_g:.3f}")
    print()

# Build a summary DataFrame
res_df = pd.DataFrame(results, columns=['D_mm','C_fit','gamma_sys','σ_C','σ_gamma'])

# --- 4) Compute “true” parameters ------------------------------------------

# Drive amplitude for normal/shear
a_drive = TP.An if mode == 'Normal' else TP.As
# h0 at resonance (μm)
h0_res = calcX(a_drive, TP.Fn_0) * 1e3
# True C and gamma_sys
C_true     = TP.K0 * h0_res
gamma_true = TP.gamma_sys

# --- 5) Plot recovered parameters vs. gap -------------------------------

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Panel a: C_fit vs D
ax1.errorbar(res_df['D_mm'], res_df['C_fit'], yerr=res_df['σ_C'],
             fmt='o', capsize=4, label=r'$C_{\rm fit}$')
ax1.axhline(C_true, color='gray', linestyle='--', label=r'True $C$')
ax1.set_xlabel('Gap $D$ (mm)', fontsize=12)
ax1.set_ylabel(r'$C_{\rm fit}$', fontsize=12)
ax1.set_title(r'Recovered $C$ vs.\ Gap', fontsize=13)
ax1.legend()

# Panel b: gamma_sys vs D
ax2.errorbar(res_df['D_mm'], res_df['gamma_sys'], yerr=res_df['σ_gamma'],
             fmt='o', capsize=4, label=r'$\gamma_{\rm sys,fit}$')
ax2.axhline(gamma_true, color='gray', linestyle='--',
            label=r'True $\gamma_{\mathrm{sys}}$')
ax2.set_xlabel('Gap $D$ (mm)', fontsize=12)
ax2.set_ylabel(r'$\gamma_{\mathrm{sys}}$ (s$^{-1}$)', fontsize=12)
ax2.set_title(r'Recovered $\gamma_{\mathrm{sys}}$ vs.\ Gap', fontsize=13)
ax2.legend()

plt.tight_layout()
plt.show()
