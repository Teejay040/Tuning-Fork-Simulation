import numpy as np

#######################
### Parameters Scan ###
#######################


An = 0.5                                       # Normal driving amplitude in Volt
As = 0.5                                       # Shear driving ampliude in Volt

Fn_0 = 840                                     # Resonance normal frequency in Hz
Fs_0 = 375                                     # Resonance shear frequency in Hz

L1 = 0.6                                       # Damping or energy dissipation related? (???)
L2 = 0.1                                       # Damping or energy dissipation related? (???)

phase = np.pi/2                                # Phase shift of the driving signal (w.r.t. resonance frequency?) (???)

Amplitude_noise = 0.025                        # Noise on the output amplitude 

#############################
### Parameters experiment ###
#############################

eta_c = 500*10**(-4)                           # m^2/s
density = 760                                  # kg/m^3
# eta = eta_c * density                        # kg/m*s
eta = 0
gamma_sys = 5.0

#D = np.array([50, 40, 30, 20, 10])*10**(-3)    # Distance between probe and surface in millimeters (mm)
D = np.array([60])*10**(-3)    # mm
R = (2.78/2)                                   # Radius of the probe in millimeter

# Fn = np.array(np.linspace(800,900,60))         # Range of normal-frequencies to be tested in Hertz (Hz)
Fn = np.array(np.linspace(830,850,201))

Fs = np.array(np.linspace(320,440,80))         # Range of shear-frequencies to be tested in Hertz (Hz)
# Fs = np.array([300])          

K0 = 2.35*10**5                                # Effective stiffness of the tuning fork prong with 
                                               # K0 = (Young's Modulus * area moment of inertia) / length of prong (???)

print("Loading TP from:", __file__)