from pathlib import Path
import wfdb
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

# Set base directory and construct path to the record
base_dir = Path(__file__).resolve().parent
data_dir = base_dir.parent / "ptb-diagnostic-ecg-database-1.0.0"
patient_id = "patient003"
record_name = "s0017lre"
record_path = data_dir / patient_id / record_name

# Load ECG record
record = wfdb.rdrecord(str(record_path), sampfrom=0, sampto=4000)
ecg = record.p_signal[:, 8]
fs = 1000
N = len(ecg)
n = np.arange(N)

# Butterworth low-pass filter design
order = 3
cutoff = 8
b, a = butter(order, cutoff / (fs / 2), btype='low')

# Apply zero-phase filtering
t_wave = filtfilt(b, a, ecg)
qrs = ecg - t_wave

# Plot ECG, T-wave, and QRS complex
plt.figure(figsize=(14, 4))
plt.plot(n, ecg, label='ECG', color='dodgerblue')
plt.plot(n, t_wave, label='T-wave ', color='red')
plt.plot(n, qrs, label='QRS complex', color='goldenrod')
plt.title(r" Zero-phase Butterworth filter ")
plt.xlabel("Samples")
plt.ylabel("Amplitude (mV)")
plt.xlim(0, N)
plt.xticks(np.arange(0, N+1, 500))
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
