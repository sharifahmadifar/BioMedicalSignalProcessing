from pathlib import Path
import wfdb
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, detrend

# ℓ₁-based truncated Fourier series approximation
def l1_fourier_mm(x, M, n_iter=30, eps=1e-6):
    N = len(x)
    n = np.arange(N)
    k = np.arange(-M, M+1)
    Phi = np.exp(1j * 2 * np.pi * np.outer(n, k) / N)
    c = np.linalg.pinv(Phi) @ x
    for _ in range(n_iter):
        xM = np.real(Phi @ c)
        e = x - xM
        Lambda_inv = np.diag(1 / (np.abs(e) + eps))
        A = Phi.conj().T @ Lambda_inv @ Phi
        b = Phi.conj().T @ Lambda_inv @ x
        c = np.linalg.pinv(A) @ b
    return np.real(Phi @ c)

# File path
base_dir = Path(__file__).resolve().parent
data_dir = base_dir.parent / "ptb-diagnostic-ecg-database-1.0.0"
record_path = data_dir / "patient003" / "s0017lre"

# Load ECG (lead V3)
record = wfdb.rdrecord(str(record_path), sampfrom=0, sampto=4000)
ecg = record.p_signal[:, 8]

# Normalize and detrend
ecg -= np.mean(ecg)
ecg *= 2.5 / np.max(ecg)
ecg = detrend(ecg, type='linear')

# Parameters
N = len(ecg)
x = np.arange(N)
M = 32

# Extract T-wave
t_wave_raw = l1_fourier_mm(ecg, M)
t_wave = savgol_filter(t_wave_raw, 17, 3)

# Isolate QRS
qrs = ecg - t_wave
qrs = detrend(qrs, type='linear')
qrs += (ecg[0] - qrs[0])

# Plot
plt.figure(figsize=(14, 4))
plt.plot(x, ecg, label='ECG', color='#00BCD4', linewidth=1.5)
plt.plot(x, t_wave, label='T-wave', color='crimson', linewidth=1.3)
plt.plot(x, qrs, label='QRS complex', color='#FF9800', linewidth=1.1)
plt.xlabel("Samples", fontsize=13)
plt.ylabel("Amplitude (mV)", fontsize=13)
plt.title("Truncated ℓ₁ Fourier-series expansion", fontsize=13)
plt.xlim(0, N)
plt.ylim(-2, 3)
plt.xticks(np.arange(0, N+1, 500))
plt.yticks(np.arange(-2, 3.5, 1))
plt.grid(True, linestyle=':', linewidth=0.5, alpha=0.5)
plt.legend(loc='upper right', fontsize=11, frameon=True)
plt.tight_layout()
plt.show()
