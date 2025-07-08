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
ecg -= np.mean(ecg)

# Normalize amplitude
ecg *= 2.5 / np.max(ecg)

# Parameters
N = len(ecg)
x = np.arange(N)
fs = 1000

# FFT and frequency truncation
fft_ecg = np.fft.fft(ecg)
freqs = np.fft.fftfreq(N, d=1/fs)
cutoff = 8

# Low-frequency part (T-wave)
t_fft = np.zeros_like(fft_ecg)
t_fft[np.abs(freqs) <= cutoff] = fft_ecg[np.abs(freqs) <= cutoff]
t_wave = np.fft.ifft(t_fft).real

# High-frequency part (QRS complex)
qrs_fft = np.zeros_like(fft_ecg)
qrs_fft[np.abs(freqs) > cutoff] = fft_ecg[np.abs(freqs) > cutoff]
qrs = np.fft.ifft(qrs_fft).real

# Plot
plt.figure(figsize=(14, 4))
plt.plot(x, ecg, label='ECG', color='#0072BD', linewidth=1.5)
plt.plot(x, t_wave, label='T-wave', color='orangered', linewidth=1)
plt.plot(x, qrs, label='QRS complex', color='goldenrod', linewidth=1)
plt.xlabel("Samples", fontsize=13)
plt.ylabel("Amplitude (mV)", fontsize=13)
plt.title("Truncated ℓ₂ Fourier-series expansion", fontsize=15)
plt.xlim(0, N)
plt.ylim(-2, 3)
plt.xticks(np.arange(0, N+1, 500))
plt.yticks(np.arange(-2, 3.5, 1))
plt.grid(True, which='both', linestyle=':', linewidth=0.6, alpha=0.6)
plt.legend(loc='upper right', fontsize=11, frameon=True)
plt.tight_layout()
plt.show()
