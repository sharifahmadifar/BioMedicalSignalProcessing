from pathlib import Path
import wfdb
import numpy as np
import matplotlib.pyplot as plt

# === Set up file paths ===
# Determine the base directory and construct path to the ECG record
base_dir = Path(__file__).resolve().parent
data_dir = base_dir.parent / "ptb-diagnostic-ecg-database-1.0.0"
patient_id = "patient003"
record_name = "s0017lre"
record_path = data_dir / patient_id / record_name

# === Load ECG signal ===
# Read 4000 samples from the specified ECG record
record = wfdb.rdrecord(str(record_path), sampfrom=0, sampto=4000)

# Extract Lead V3 (channel index 8) signal
ecg = record.p_signal[:, 8]

# Define sampling rate and sample index array
fs = 1000  # Hz
N = len(ecg)  # Total number of samples
n = np.arange(N)

# === Set number of Fourier coefficients to retain (±M) ===
M = 32

# === Perform FFT and normalize ===
# Compute the Fourier Transform of the ECG signal
a = np.fft.fft(ecg) / N

# === Blackman-windowed truncated Fourier series reconstruction ===
# Create Blackman window of length 2M+1 for smoothing
window = np.blackman(2*M + 1)

# Initialize reconstructed T-wave signal (complex, same length as ECG)
t_wave = np.zeros(N, dtype=complex)

# Reconstruct the signal using truncated and windowed Fourier series
for idx, k in enumerate(range(-M, M+1)):
    coef_idx = k if k >= 0 else N + k  # Wrap negative indices
    w = window[idx]  # Window coefficient
    t_wave += w * a[coef_idx] * np.exp(1j * 2 * np.pi * k * n / N)

# Keep only the real part (imaginary part should be negligible)
t_wave = t_wave.real

# === Isolate QRS complex ===
# Subtract reconstructed T-wave from original ECG to isolate the QRS complex
qrs = ecg - t_wave

# === Plot ECG, T-wave, and QRS complex ===
plt.figure(figsize=(14, 4))
plt.plot(n, ecg, label='ECG', color='dodgerblue')
plt.plot(n, t_wave, label='T-wave', color='orangered')
plt.plot(n, qrs, label='QRS complex', color='goldenrod')
plt.title(r" Truncated $\ell_2$ Fourier-series expansion (Blackman window)")
plt.xlabel("Samples")
plt.ylabel("Amplitude (mV)")
plt.xlim(0, N)
plt.xticks(np.arange(0, N+1, 500))
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
