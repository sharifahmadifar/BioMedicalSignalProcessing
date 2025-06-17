from pathlib import Path
import wfdb
import numpy as np
import matplotlib.pyplot as plt

# Set base directory and data path
base_dir = Path(__file__).resolve().parent
data_dir = base_dir.parent / "ptb-diagnostic-ecg-database-1.0.0"

# Specifying the ECG record to load
patient_id = "patient003"
record_name = "s0017lre"

# Build the full path to the specific ECG record
record_path = data_dir / patient_id / record_name

# Load 4000 samples from the ECG record
record = wfdb.rdrecord(str(record_path), sampfrom=0, sampto=4000)

# Extract Lead V4
ecg = record.p_signal[:, 8]

# Defining signal and model parameters
N = len(ecg)        # Number of samples
fs = 1000           # Sampling rate (Hz)
M = 70              # Number of Fourier harmonics to use in the model
epsilon = 1e-8      # Small constant to avoid divide-by-zero errors
num_iterations = 200  # Number of IRLS iterations

# Generate the time and frequency index arrays
n = np.arange(N).reshape(-1, 1)    # Sample index column vector
k = np.arange(M)                   # Harmonic index row vector

# Build the complex Fourier basis matrix (N x M)
Φ = np.exp(1j * 2 * np.pi * k * n / N)

# Initialize the complex coefficients with small random values
c = np.zeros(M, dtype=complex) + 1e-5 * np.random.randn(M) * 1j

# Iterative reweighted least squares (IRLS) to minimize ℓ₁ reconstruction error
for r in range(num_iterations):
    x_M = Φ @ c                 # Reconstruct signal using current coefficients
    e_r = ecg - x_M             # Compute residual (error)

    weights = np.abs(e_r) + epsilon     # ℓ₁-like weights (inverse of residual magnitude)
    Λ_r_inv = np.diag(1 / weights)      # Diagonal weighting matrix

    # Weighted least squares update step
    A = Φ.conj().T @ Λ_r_inv @ Φ
    B = Φ.conj().T @ Λ_r_inv @ ecg
    c = np.linalg.solve(A, B)          # Update Fourier coefficients

# After learning the coefficients, extract low-frequency (T-wave) components
freq_step = fs / N                    # Frequency resolution per FFT bin
cutoff_idx = int(15 / freq_step)      # Use harmonics below 15 Hz for T-wave
t_wave = np.real(np.sum([c[i] * Φ[:, i] for i in range(min(cutoff_idx, M))], axis=0))

# Subtract the T-wave from the original ECG to isolate the QRS complex
qrs = ecg - t_wave

# Plot the results
x_time = np.arange(N)
plt.figure(figsize=(16, 5))
plt.plot(x_time, ecg, label="Original ECG Signal", color='black', alpha=0.6)
plt.plot(x_time, t_wave, '--r', label="T-wave")       # Low-frequency content
plt.plot(x_time, qrs, ':g', label="QRS Complex")      # High-frequency content
plt.title("Truncated ℓ₁ Fourier-series expansion")
plt.xlabel("Sample")
plt.ylabel("Amplitude (mV)")
plt.xticks(np.arange(0, N+1, 500))
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()