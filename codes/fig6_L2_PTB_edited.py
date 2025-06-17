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

# Set the sampling frequency and calculate the number of samples
fs = 1000
N = len(ecg)

# Compute the Fast Fourier Transform of the ECG signal
fft_ecg = np.fft.fft(ecg)

# Compute corresponding frequency values for each FFT component
freqs = np.fft.fftfreq(N, d=1/fs)

# Set a cutoff frequency to separate low and high frequency components
cutoff = 8  # Hz

# Create copies of the FFT for isolating components
t_fft = np.copy(fft_ecg)     # For T-wave (low-frequency)
qrs_fft = np.copy(fft_ecg)   # For QRS complex (high-frequency)

# Zero out high-frequency components in t_fft (keep only low frequencies)
t_fft[np.abs(freqs) > cutoff] = 0

# Zero out low-frequency components in qrs_fft (keep only high frequencies)
qrs_fft[np.abs(freqs) <= cutoff] = 0

# Inverse FFT to convert filtered signals back to time domain
t_wave = np.fft.ifft(t_fft).real     # Low-frequency T-wave
qrs = np.fft.ifft(qrs_fft).real      # High-frequency QRS complex

# Plot the original ECG along with its separated components
x = np.arange(N)
plt.figure(figsize=(16, 5))
plt.plot(x, ecg, label="Original ECG", color='black', alpha=0.6)
plt.plot(x, t_wave, '--r', label="T-wave (Low Freq)")
plt.plot(x, qrs, ':g', label="QRS Complex (High Freq)")
plt.title("Truncated ℓ₂ Fourier-series expansion")
plt.xlabel("Samples")
plt.ylabel("Amplitude (mV)")
plt.xticks(np.arange(0, 4001, 500))
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
