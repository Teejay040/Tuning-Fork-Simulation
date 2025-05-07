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
    """
    amplitude_list = []
    frequency_list = []
    Q_list = []

    # Select resonant omega
    if s == "Shear":
        omega_0 = omega_0_S
    elif s == "Normal":
        omega_0 = omega_0_N
    else:
        raise ValueError("Mode s must be 'Shear' or 'Normal'")

    phase = TP.phase
    amplitude_noise = TP.Amplitude_noise

    for d in Dist:
        print(f"Processing distance: {d}")
        n_omega = len(Freq)
        TF_array = np.zeros(n_omega)
        Test_omega_array = np.zeros(n_omega)
        K = 1.0 / d
        damp = 1.0 / d

        for n, freq_val in enumerate(Freq):
            print(f"  Frequency: {freq_val}")
            omega_n = 2 * np.pi * freq_val
            n_period = 8000
            n_points = 100
            t = np.linspace(1 / freq_val, n_period / freq_val, (n_period - 1) * n_points)
            dt = t[1] - t[0]

            # Drive amplitude and displacement
            a = An if s == "Normal" else As
            h_0 = calcX(a, freq_val) * 1e3  # μm

            # Precompute distance-to-surface array (optional)
            H_array = d + h_0 * np.cos(omega_n * t + phase)
            H_array[H_array < 0] = 0

            def H_func(tt):
                return d + h_0 * np.cos(omega_n * tt + phase)

            def H_dot(tt):
                return -omega_n * h_0 * np.sin(omega_n * tt + phase)

            noise = None
            if i in (2, 3):
                noise = amplitude_noise * np.random.normal(0, 1, len(t))

            def rhs(y, tt):
                h = h_0 * np.cos(omega_n * tt + phase)
                if i == 1:
                    return [y[1], K*h - 2*damp*y[1] - omega_0**2*y[0]]
                elif i == 2:
                    idx = min(int((tt - t[0]) / dt), len(noise)-1)
                    return [y[1], K*h - 2*damp*(1+noise[idx])*y[1] - omega_0**2*y[0]]
                elif i == 3:
                    idx = min(int((tt - t[0]) / dt), len(noise)-1)
                    return [y[1], K*h - 2*damp*y[1] - omega_0**2*(1+noise[idx])*y[0]]
                elif i == 4:
                    x = np.linspace(0, tt, 10)
                    f_int = (H_dot(x) / H_func(x)**2) * np.exp((tt - x) / L1)
                    M_val = (1 - L2/L1) * H_func(tt)**2 / (L1*H_dot(tt)) * trapz(f_int, x) + L2/L1
                    return [y[1], K*h - 2*damp*M_val*y[1] - omega_0**2*y[0]]
                else:
                    raise ValueError("Invalid simulation case i")

            # Initial conditions
            y0 = [K*h_0, K*h_0/dt]
            sol = odeint(rhs, y0, t)
            y_sol = sol[:, 0]

            # Remove transient for cases 1–3
            if i != 4:
                t_trunc = t[-100*n_points-1:]
                y_sol = y_sol[-100*n_points-1:]
                t_sol = t_trunc - t_trunc[0]
            else:
                t_sol = t

            # FFT analysis
            X = fft(y_sol)
            freq_comp = fftfreq(len(y_sol), d=dt)
            N = len(y_sol) // 2
            X_abs = np.abs(X[:N])
            X_norm = X_abs / N
            freq_pos = freq_comp[:N]

            # find_peaks instead of argrelmax(…, np.greater)
            peaks, _ = find_peaks(X_norm)
            # filter out the zero‐frequency peak if needed
            peaks = peaks[freq_pos[peaks] > 0]
            if len(peaks) > 1:
                peaks = peaks[freq_pos[peaks] >= 400]
            if len(peaks) > 0:
                peak_idx = peaks[np.argmax(X_norm[peaks])]
            else:
                peak_idx = np.argmax(X_norm)

            TF_array[n] = X_norm[peak_idx]
            Test_omega_array[n] = freq_pos[peak_idx]

        amplitude_list.append(TF_array)
        frequency_list.append(Test_omega_array)

        # Compute Q from first threshold width
        thresholds = np.linspace(0.5, 0.99, 50)
        widths = [float(Width(Test_omega_array, TF_array, th)) for th in thresholds]
        Q_list.append(Test_omega_array[np.argmax(TF_array)] / widths[0])

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
