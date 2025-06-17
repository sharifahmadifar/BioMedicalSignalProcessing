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

# Limit the number of samples to load
num_samples = 4000

try:
    # Read the ECG record by using Wfdb library
    record = wfdb.rdrecord(str(record_path), sampto=num_samples)

    # Extract the raw ECG signal
    signal = record.p_signal

    # Choose the 9th lead if available, otherwise fallback to the 2nd
    lead_index = 8 if signal.shape[1] > 8 else 1
    ecg_signal = signal[:, lead_index]

    # Plot the selected ECG lead
    plt.figure(figsize=(14, 4))
    plt.plot(ecg_signal, color='navy')
    plt.title(f"Real ECG record {record_name} from the PhysioNet PTB Diagnostic ECG Database (ptbdb)")
    plt.xlabel("Samples")
    plt.ylabel("Amplitude (mV)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

except FileNotFoundError:
    # Handle the case where the specified ECG file doesn't exist
    print(f"Error: The record file was not found at the specified path: {record_path}")
except Exception as e:
    # Catch and print any other unexpected errors
    print(f"An error occurred: {e}")
