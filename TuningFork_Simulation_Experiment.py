import time
import re
import numpy as np
import matplotlib.pyplot as plt

import TuningFork_Simulation_Function as TF
import TuningFork_Simulation_Parameters as TP

Make_one_simulation = 'Yes'
i = 1
s = 'Normal'

Read_Data = 'No'
fname = 'Simulation_Scan_Normal_Zero.txt'

# Fluid & probe params
eta = TP.eta
D = TP.D
R = TP.R
Fn = TP.Fn
Fs = TP.Fs
K0 = TP.K0

if Make_one_simulation == 'Yes':
    # choose drive frequencies
    if s == "Normal":
        F_drive = Fn
    else:
        F_drive = Fs

    # run scan
    Amp, Freq_test, QFactor = TF.Scan(F_drive, D, i, s)

    # output file
    outname = f"Simulation_Scan_{eta}_{s}_i_{i}_Number_of_D_{len(D)}_of_F_{len(F_drive)}.TXT"
    with open(outname, 'w') as f:
        f.write('D [mm], Frequency[Hz], test Frequency, Amplitude Scan, Quality factor\n')
        for idx_d, d in enumerate(D):
            for v in range(len(F_drive)):
                f.write(f"{d},{F_drive[v]},{Freq_test[idx_d][v]},{Amp[idx_d][v]},{QFactor[idx_d]}\n")

# (optional) reading back in...
if Read_Data == 'Yes':
    # … your existing read‐in code (or consider pandas.read_csv) …
    pass

# ---- consolidated plotting ----

# 1) Amplitude vs. ω
plt.figure('Scan Amplitude')
for idx_d, d in enumerate(D):
    omega = 2 * np.pi * np.array(F_drive)
    plt.plot(omega, Amp[idx_d], 'o-', markersize=6, label=f'D = {d} mm')
plt.xlabel(r'$\omega$ [Hz]', fontsize=15)
plt.ylabel('A (a.u.)',     fontsize=15)
plt.tick_params(labelsize=12)
plt.legend(fontsize=12)
plt.tight_layout()

# 2) Test frequency vs. drive frequency
plt.figure('Frequency Test')
for idx_d, d in enumerate(D):
    plt.plot(F_drive, Freq_test[idx_d], 'o-', markersize=6, label=f'D = {d} mm')
plt.xlabel('F drive [Hz]',      fontsize=15)
plt.ylabel('F found [Hz]',      fontsize=15)
plt.tick_params(labelsize=12)
plt.legend(fontsize=12)
plt.tight_layout()

# 3) Quality factor vs. D
plt.figure('Quality Factor')
plt.plot(D, QFactor, 'o-', markersize=8)
plt.xlabel('D [mm]',  fontsize=15)
plt.ylabel('Q',       fontsize=15)
plt.tick_params(labelsize=12)
plt.tight_layout()

plt.show()