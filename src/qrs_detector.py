# using the filtered data, this narrows down the bandpass filter to 5-15Hz
# overall, makes the QRS segments a lot more 'visible' and extra parts are reduced to near zero
from scipy import signal
import numpy as np
from data_loader import load_ecg_record
from dsp_filters import bandpass_filter
import matplotlib.pyplot as plt


def pan_tompkins_preprocessing(input_signal, sampling_freq):
    lower_bound = 5/(sampling_freq/2) # normalized frequencies
    upper_bound = 15/(sampling_freq/2) 

    # create another bandpass filter
    b_band, a_band = signal.butter(N=2, btype='bandpass', Wn=[lower_bound, upper_bound])

    # 60Hz notch filter
    b, a = signal.iirnotch(w0=60, Q=30, fs=sampling_freq)

    # zero phase filtering
    filtered_signal = signal.filtfilt(b, a, input_signal)
    bandpassed_signal = signal.filtfilt(b_band, a_band, filtered_signal)

    # implement derivative filter
    # np.diff finds the difference between adjacent samples
    derivative_array = np.gradient(bandpassed_signal)

    # square derivatives
    squared_array = np.square(derivative_array)

    # moving window integrator
    # helps to average out and smooth out samples
    n = int(np.round(0.15 * sampling_freq)) # window sample size
    u = [1/n for i in range(n)] # uniform averaging kernal

    # discrete convolution
    result = np.convolve(squared_array, u, mode='same')

    return result, bandpassed_signal, derivative_array, squared_array


if __name__ == "__main__":
    # load 10 seconds of record 100
    raw_signal, fs = load_ecg_record("100", sample_limit=3600)

    clean_ecg, f_raw, Pxx_raw, f_filt, Pxx_filt = bandpass_filter(
        sampling_freq=fs, f_nyq=fs/2, original_signal=raw_signal
    )

    # Pan-Tompkins preprocessing
    integrated, bandpass, derivative, squared = pan_tompkins_preprocessing(
        clean_ecg, fs
    )

    time_axis = np.arange(len(raw_signal)) / fs

    # 5-panel visualization plot
    fig, axes = plt.subplots(5, 1, figsize=(12, 10), sharex=True)

    axes[0].plot(time_axis, clean_ecg, color="black", linewidth=1)
    axes[0].set_title("1. Cleaned ECG Signal")
    axes[0].set_ylabel("mV")

    axes[1].plot(time_axis, bandpass, color="blue", linewidth=1)
    axes[1].set_title("2. Step A: Bandpass Filtered (5 - 15 Hz)")
    axes[1].set_ylabel("mV")

    axes[2].plot(time_axis, derivative, color="green", linewidth=1)
    axes[2].set_title("3. Step B: Derivative (dy/dt)")

    axes[3].plot(time_axis, squared, color="purple", linewidth=1)
    axes[3].set_title("4. Step C: Squared Derivative ($y^2$)")

    axes[4].plot(time_axis, integrated, color="crimson", linewidth=1.5)
    axes[4].set_title(
        "5. Step D: Moving Window Integration (150ms Energy Envelope)"
    )
    axes[4].set_xlabel("Time (seconds)")

    for ax in axes:
        ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.show()





    