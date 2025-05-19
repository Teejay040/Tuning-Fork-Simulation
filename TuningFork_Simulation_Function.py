import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import hilbert, argrelmax, argrelmin, find_peaks, correlate, correlation_lags, argrelextrema
from scipy.integrate import odeint
from numpy.fft import fft, fftfreq
import TuningFork_Simulation_Parameters as TP

def calcX(V, f):
    """Convert drive voltage and frequency to displacement X (m)."""
    return ((V * np.sqrt(2)) * 9.80665e9) / (0.22 * (2 * np.pi * f)**2)

def Width(x, y, t):
    """Compute width at fraction t of the peak height."""
    Z = max(y) * t
    idx = np.where(y >= Z)[0]
    if len(idx) < 2:
        return 0.0
    return x[idx[-1]] - x[idx[0]]

def _1Lorentzian(x, amp1, cen1, wid1):
    """Single Lorentzian peak."""
    return amp1 * wid1**2 / ((x - cen1)**2 + wid1**2)

# Load parameters
An = TP.An
As = TP.As
Fn_0 = TP.Fn_0
omega_0_N = 2 * np.pi * Fn_0
Fs_0 = TP.Fs_0
omega_0_S = 2 * np.pi * Fs_0
L1 = TP.L1
L2 = TP.L2

def Scan(Freq, Dist, i, s):
    """
    Scan over distances Dist and drive frequencies Freq.
    i = simulation case (1–4), s = 'Normal' or 'Shear'.
    Returns: [amplitude_list, frequency_list, Q_list]

    Damping now has two pieces:
      • γ_sys = TP.gamma_sys            (distance‐independent)
      • γ_visc = π·R²·η / D             (distance‐dependent, geometry)
      → total_gamma = γ_sys + γ_visc
    Assumes:
      - TP.R is probe radius in mm
      - Dist entries are in mm
      - TP.eta is dynamic viscosity in Pa·s
    """

    import numpy as np
    from scipy.integrate import odeint
    from scipy.signal import find_peaks
    from numpy.fft import fft, fftfreq
    import TuningFork_Simulation_Parameters as TP

    # System constants
    K0        = TP.K0             # fork stiffness (N/m)
    gamma_sys = TP.gamma_sys      # systemic damping [1/s]
    eta       = TP.eta            # viscosity [Pa·s]
    phase     = TP.phase
    amp_noise = TP.Amplitude_noise

    # Probe geometry: convert mm → m
    R_m = TP.R * 1e-3

    # Choose resonance & drive amplitude
    if s == "Shear":
        omega_0 = 2*np.pi * TP.Fs_0 # is this conversion necassery? gives omega ~5000
        a_drive = TP.As
    else:
        omega_0 = 2*np.pi * TP.Fn_0
        a_drive = TP.An

    amplitude_list = []
    frequency_list = []
    Q_list         = []

    for d in Dist:
        print(f"Distance = {d} mm")
        # convert gap mm → m
        d_m = d * 1e-3

        # compute viscous damping for this gap
        gamma_visc = np.pi * R_m**2 * eta / d_m
        total_gamma = gamma_sys + gamma_visc
        # odeint uses damp = total_gamma/2 so that -2*damp*v = -total_gamma*v
        damp = total_gamma / 2.0

        # constant stiffness
        K = K0

        TF_array        = np.zeros(len(Freq))
        Test_omega_array = np.zeros(len(Freq))

        for n, f_drive in enumerate(Freq):
            print(f"  Drive freq: {f_drive} Hz")
            ω_drive = 2*np.pi * f_drive

            # time array
            n_period, n_points = 8000, 100
            t = np.linspace(1/f_drive,
                            n_period/f_drive,
                            (n_period-1)*n_points)
            dt = t[1] - t[0]

            # drive displacement amplitude (μm → code units)
            h0 = ((a_drive * np.sqrt(2) * 9.80665e9)
                  /(0.22*(2*np.pi*f_drive)**2)) * 1e3

            # noise if needed
            noise = None
            if i in (2,3):
                noise = amp_noise * np.random.normal(size=len(t))

            # ODE RHS
            def rhs(y, tt):
                h_t = h0 * np.cos(ω_drive*tt + phase)

                if i == 1:
                    accel = K*h_t - total_gamma*y[1] - omega_0**2 * y[0]

                elif i == 2:
                    idx = min(int((tt - t[0])/dt), len(noise)-1)
                    accel = (K*h_t
                             - total_gamma*(1+noise[idx])*y[1]
                             - omega_0**2 * y[0])

                elif i == 3:
                    idx = min(int((tt - t[0])/dt), len(noise)-1)
                    accel = (K*h_t
                             - total_gamma*y[1]
                             - omega_0**2*(1+noise[idx])*y[0])

                elif i == 4:
                    # memory‐kernel branch unchanged
                    x = np.linspace(0, tt, 100)
                    H = lambda tt: d + h0*np.cos(ω_drive*tt + phase)
                    Hd = lambda tt: -ω_drive*h0*np.sin(ω_drive*tt + phase)
                    f_int = (Hd(x)/H(x)**2) * np.exp((tt-x)/TP.L1)
                    M = ((1 - TP.L2/TP.L1)*H(tt)**2
                         /(TP.L1*Hd(tt))*trapz(f_int, x)
                         + TP.L2/TP.L1)
                    accel = (K*h_t
                             - 2*damp*M*y[1]
                             - omega_0**2*y[0])
                else:
                    raise ValueError("Invalid simulation case i")

                return [y[1], accel]

            # initial conditions & solve
            y0  = [K*h0, K*h0/dt]
            sol = odeint(rhs, y0, t)
            y_sol = sol[:,0]

            # trim transients if not case 4
            if i != 4:
                cut = -100*n_points - 1
                y_sol = y_sol[cut:]
            # FFT & peak pick
            X     = fft(y_sol)
            freqs = fftfreq(len(y_sol), d=dt)
            Npos  = len(y_sol)//2
            Xn    = np.abs(X[:Npos])/Npos
            fp    = freqs[:Npos]

            peaks, _ = find_peaks(Xn)
            peaks    = [p for p in peaks if fp[p]>0]
            if len(peaks)>1:
                peaks = [p for p in peaks if fp[p]>=400]
            peak_idx = max(peaks, key=lambda p: Xn[p]) if peaks else np.argmax(Xn)

            TF_array[n]        = Xn[peak_idx]
            Test_omega_array[n] = fp[peak_idx]

        amplitude_list.append(TF_array)
        frequency_list.append(Test_omega_array)

        # compute Q at half-height
        Δω = Width(Test_omega_array, TF_array, 0.5)
        ω_peak = Test_omega_array[np.argmax(TF_array)]
        Q_list.append(ω_peak/Δω)

    # Write results once, after loops
    if i in (1, 4):
        fname = f"Simulation_Scan_{s}_i_{i}_Number_of_D_{len(Dist)}_of_F_{len(Freq)}.TXT"
    else:
        fname = (f"Simulation_Scan_{s}_i_{i}_Number_of_D_{len(Dist)}_of_F_{len(Freq)}"
                 f"_Amplitude_Noise_{TP.Amplitude_noise}.TXT")
    with open(fname, 'w') as f:
        f.write('D [mm], Frequency[Hz], test Frequency, Amplitude Scan, Quality factor\n')
        for idx_d, d in enumerate(Dist):
            TF = amplitude_list[idx_d]
            FR = frequency_list[idx_d]
            Qv = Q_list[idx_d]
            for v in range(len(Freq)):
                f.write(f"{d},{Freq[v]},{FR[v]},{TF[v]},{Qv}\n")

    return [amplitude_list, frequency_list, Q_list]
