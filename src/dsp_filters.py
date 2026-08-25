# this file cleans the data from noise (breathing, muscle movements, etc.) by implementing
# a butterworth bandpass filter
# will essentially cutoff outlier data according to an upper and lower threshold

from scipy import signal
from .data_loader import load_ecg_record
import matplotlib.pyplot as plt
import numpy as np


def bandpass_filter(sampling_freq, f_nyq, original_signal):
    # attempt at butterworth bandpass
    upper_bound = 40 / f_nyq  # normalized upper bound
    lower_bound = 0.5 / f_nyq  # normalized lower bound
    b_band, a_band = signal.butter(N=2, btype='bandpass', Wn=[lower_bound, upper_bound])

    # 60Hz notch filter
    b, a = signal.iirnotch(w0=60, Q=30, fs=sampling_freq)

    # zero phase filtering
    filtered_signal = signal.filtfilt(b, a, original_signal)
    filtered_signal = signal.filtfilt(b_band, a_band, filtered_signal)

    # validate in frequency domain
    f_raw, Pxx_raw = signal.welch(original_signal, fs=sampling_freq, nperseg=1024)
    f_filtered, Pxx_filtered = signal.welch(filtered_signal, fs=sampling_freq, nperseg=1024)

    # At the end of bandpass_filter():
    return filtered_signal, f_raw, Pxx_raw, f_filtered, Pxx_filtered


def plot_ecg_comparison(time_axis, original_signal, filtered_signal, f_raw, Pxx_raw, f_filt, Pxx_filt):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=False)

    # panel 1: time domain (Raw vs Filtered Overlay)
    ax1.plot(
        time_axis,
        original_signal,
        label="Raw Signal",
        color="crimson",
        alpha=0.5,
        linewidth=1,
    )
    ax1.plot(
        time_axis,
        filtered_signal,
        label="Filtered (0.5 - 40 Hz + 60Hz Notch)",
        color="blue",
        linewidth=1.2,
    )
    ax1.set_title("Time Domain: Raw vs. Filtered ECG Signal", fontsize=12)
    ax1.set_xlabel("Time (seconds)")
    ax1.set_ylabel("Voltage (mV)")
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(loc="upper right")

    # panel 2: frequency domain (Power Spectral Density)
    ax2.plot(
        f_raw, Pxx_raw, label="Raw Spectrum", color="crimson", alpha=0.5, linewidth=1
    )
    ax2.plot(
        f_filt,
        Pxx_filt,
        label="Filtered Spectrum",
        color="blue",
        linewidth=1.2,
    )
    ax2.set_title("Frequency Domain: Power Spectral Density (PSD)", fontsize=12)
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_ylabel("Power Density ($mV^2 / Hz$)")

    # zoom in on 0 - 100 Hz where cardiac frequencies and 60 Hz hum exist
    ax2.set_xlim(0, 100)
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend(loc="upper right")

    plt.tight_layout()
    plt.show()

    
if __name__ == "__main__":
    original_signal, sampling_freq = load_ecg_record("100", sample_limit=3600)
    print(f"signal shape: {original_signal.shape}")
    print(f"sampling rate: {sampling_freq} Hz")
    print(f"first 5 voltage samples (mV) {original_signal[:5]}")
    filtered_signal, f_raw, Pxx_raw, f_filt, Pxx_filt = bandpass_filter(
        sampling_freq=sampling_freq, 
        f_nyq=sampling_freq/2, 
        original_signal=original_signal
    )

    # plot the 2 signals
    time_axis = np.arange(len(original_signal)) / sampling_freq
    plot_ecg_comparison(time_axis, original_signal, filtered_signal, f_raw, Pxx_raw, f_filt, Pxx_filt )
