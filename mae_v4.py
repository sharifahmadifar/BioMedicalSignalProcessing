from pathlib import Path
import random
import wfdb
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# Base directory of the ECG dataset
base_dir = Path(__file__).resolve().parent.parent / "ptb-diagnostic-ecg-database-1.0.0"

# Collect all record paths (excluding file extensions)
record_paths = []
for path in base_dir.rglob("*.hea"):
    record_paths.append(str(path.with_suffix('')))

# Shuffle for random selection
random.shuffle(record_paths)

# L2 Fourier (truncated low-pass reconstruction)
def l2_fourier(ecg, cutoff=8, fs=1000):
    N = len(ecg)
    fft_ecg = np.fft.fft(ecg)
    freqs = np.fft.fftfreq(N, d=1/fs)
    t_fft = np.zeros_like(fft_ecg)
    t_fft[np.abs(freqs) <= cutoff] = fft_ecg[np.abs(freqs) <= cutoff]
    return np.fft.ifft(t_fft).real

# L1 Fourier (IRLS-based smoothed reconstruction)
def l1_fourier_mm(x, M, n_iter=30, eps=1e-6):
    N = len(x)
    n = np.arange(N)
    k = np.arange(-M, M+1)
    Phi = np.exp(1j * 2 * np.pi * np.outer(n, k) / N)
    c = np.linalg.pinv(Phi) @ x
    for r in range(n_iter):
        xM = np.real(Phi @ c)
        e = x - xM
        Lambda_inv = np.diag(1 / (np.abs(e) + eps))
        A = Phi.conj().T @ Lambda_inv @ Phi
        b = Phi.conj().T @ Lambda_inv @ x
        c = np.linalg.pinv(A) @ b
    return np.real(Phi @ c)

# Storage for MAE results
valid_mae_l2 = []
valid_mae_l1 = []
valid_rec_names = []

i = 0
# Evaluate until 20 valid records are collected
while len(valid_mae_l2) < 20 and i < len(record_paths):
    rec_path = record_paths[i]
    try:
        record = wfdb.rdrecord(rec_path, sampfrom=0, sampto=4000)
        lead_idx = 7 if record.p_signal.shape[1] > 7 else 0
        ecg = record.p_signal[:, lead_idx]

        # Normalize signal
        ecg = ecg - np.mean(ecg)
        ecg *= 2.5 / np.max(ecg)

        # Reconstruct T-wave using both methods
        t_wave_l2 = l2_fourier(ecg, cutoff=8)
        t_wave_l1 = l1_fourier_mm(ecg, M=32, n_iter=30)

        # Compute Mean Absolute Error
        mae_l2 = np.mean(np.abs(ecg - t_wave_l2))
        mae_l1 = np.mean(np.abs(ecg - t_wave_l1))

        # Save results
        valid_mae_l2.append(mae_l2)
        valid_mae_l1.append(mae_l1)
        valid_rec_names.append(Path(rec_path).name)

        print(f"{Path(rec_path).name}: L2 MAE={mae_l2:.3f}, L1 MAE={mae_l1:.3f}")
    except Exception as e:
        print(f"Error processing {rec_path}: {e}")
    i += 1

# Summary statistics
mean_l2 = np.mean(valid_mae_l2)
std_l2 = np.std(valid_mae_l2)
mean_l1 = np.mean(valid_mae_l1)
std_l1 = np.std(valid_mae_l1)

# Print table of results
print("\nRecord\t\tL2 MAE\tL1 MAE")
for name, l2, l1 in zip(valid_rec_names, valid_mae_l2, valid_mae_l1):
    print(f"{name}\t{l2:.3f}\t{l1:.3f}")

# Print overall mean and std
print(f"\nSummary (mean ± std):")
print(f"L2 Fourier MAE = {mean_l2:.3f} ± {std_l2:.3f}")
print(f"L1 Fourier MAE = {mean_l1:.3f} ± {std_l1:.3f}")

# Plot comparison
x = np.arange(len(valid_rec_names))
plt.figure(figsize=(10, 4))
plt.plot(x, valid_mae_l2, marker='>', label='L2 Fourier', color='#1976D2')
plt.plot(x, valid_mae_l1, marker='o', label='L1 Fourier', color='#FFA000')
plt.xticks(x, valid_rec_names, rotation=45, ha='right', fontsize=9)
plt.ylabel("MAE", fontsize=13)
plt.xlabel("ECG Record", fontsize=13)
plt.title("Comparison of MAE for L2 vs L1 Fourier (20 valid random records)", fontsize=14)
plt.grid(True, which='both', linestyle='--', alpha=0.4)
plt.legend(fontsize=12)
plt.tight_layout()
plt.show()
