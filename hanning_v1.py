from pathlib import Path
import wfdb
import numpy as np
import matplotlib.pyplot as plt

# File path
base_dir = Path(__file__).resolve().parent
data_dir = base_dir.parent / "ptb-diagnostic-ecg-database-1.0.0"
record_path = data_dir / "patient003" / "s0017lre"

# Load ECG (lead V3)
record = wfdb.rdrecord(str(record_path), sampfrom=0, sampto=4000)
ecg = record.p_signal[:, 8]

# Parameters
fs = 1000
N = len(ecg)
n = np.arange(N)
M = 32

# FFT and Hanning window
a = np.fft.fft(ecg) / N
window = np.hanning(2*M + 1)

# Reconstruct T-wave using windowed truncated Fourier series
t_wave = np.zeros(N, dtype=complex)
for idx, k in enumerate(range(-M, M+1)):
    coef_idx = k if k >= 0 else N + k
    t_wave += window[idx] * a[coef_idx] * np.exp(1j * 2 * np.pi * k * n / N)
t_wave = t_wave.real

# QRS complex
qrs = ecg - t_wave

# Plot
plt.figure(figsize=(14, 4))
plt.plot(n, ecg, label='ECG', color='dodgerblue')
plt.plot(n, t_wave, label='T-wave', color='orangered')
plt.plot(n, qrs, label='QRS complex', color='goldenrod')
plt.title(r"Truncated $\ell_2$ Fourier-series expansion (Hanning window)")
plt.xlabel("Samples")
plt.ylabel("Amplitude (mV)")
plt.xlim(0, N)
plt.xticks(np.arange(0, N+1, 500))
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
