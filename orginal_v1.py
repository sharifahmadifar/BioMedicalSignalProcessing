from pathlib import Path
import wfdb
import numpy as np
import matplotlib.pyplot as plt

# Set base directory and construct path to the record
base_dir = Path(__file__).resolve().parent
data_dir = base_dir.parent / "ptb-diagnostic-ecg-database-1.0.0"
patient_id = "patient003"
record_name = "s0017lre"
record_path = data_dir / patient_id / record_name

# Load ECG record
record = wfdb.rdrecord(str(record_path), sampfrom=0, sampto=4000)
ecg = record.p_signal[:, 8]  # 9th lead (index 8)

# Remove DC offset by centering the signal around zero
ecg = ecg - np.mean(ecg)
# Set desired R-peak amplitude (for normalization)
desired_R_peak = 2.5
# Compute scaling factor to match the desired peak
scale = desired_R_peak / np.max(ecg)
# Apply scaling to normalize ECG amplitude
ecg = ecg * scale

# Plot
N = len(ecg)
x = np.arange(N)

plt.figure(figsize=(12, 3))
plt.plot(x, ecg, color='#0072BD', linewidth=1.5, label='ECG')
plt.xlabel("Samples", fontsize=13)
plt.ylabel("Amplitude (mV)", fontsize=13)
plt.title("Real ECG record s0017lre (PTB Diagnostic ECG Database)", fontsize=15)
plt.xlim(0, N)
plt.ylim(-2, 3)
plt.xticks(np.arange(0, N + 1, 500))
plt.yticks(np.arange(-2, 3.5, 1))
plt.grid(True, which='both', linestyle=':', linewidth=0.6, alpha=0.6)
plt.legend(loc='upper right', fontsize=11, frameon=True)
plt.tight_layout()
plt.show()
