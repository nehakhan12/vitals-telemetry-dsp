# using the filtered data, this narrows down the bandpass filter to 5-15Hz
# overall, makes the QRS segments a lot more 'visible' and extra parts are reduced to near zero
from scipy import signal
import numpy as np
from .data_loader import load_ecg_record
from .dsp_filters import bandpass_filter
import matplotlib.pyplot as plt
import wfdb


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


def adaptive_peak_detection(integrated, sampling_freq, cleaned_signal):
    arr = np.array(integrated)

    # use numpy to find local maxima in the array 
    greater_than_left = arr[1:-1] > arr[:-2]
    greater_than_right = arr[1:-1] > arr[2:]

    local_max_times = np.where(greater_than_left & greater_than_right)[0] + 1
    local_max_heights = arr[local_max_times]

    # initialize values for spki (signal level) and npki (noise level)
    spki = np.max(local_max_heights[:5]) * 0.35  # sets to 35% of the max peak height of the first few seconds
    npki = np.mean(local_max_heights) * 0.1  # sets noise level to be 10% of the avg peak height

    check = 0.2 * sampling_freq
    prev = -check


    accepted_timestamps = []
    for i in range(len(local_max_heights)):
        if local_max_times[i] - prev > check:
            threshold = npki + (0.25 * (spki - npki))

            if local_max_heights[i] > threshold: # add to list of accepted heartbeat timestamps
                accepted_timestamps.append(local_max_times[i])
                spki = 0.125 * local_max_heights[i] + (0.875 * spki)
                prev = local_max_times[i]

            else: # peak gets treated as background noise
                npki = 0.125 * local_max_heights[i] + (0.875*npki)
    
    # using the accepted timestamps, find the max voltage value within the window for actual r peaks
    r_peaks = [] # to store the indeces of the max r_peaks
    for i in accepted_timestamps:
        if i > 25 and i < len(cleaned_signal) - 30:
            r = np.argmax(cleaned_signal[i - 25: i + 30])
            r = i - 25 + r
            r_peaks.append(r)

    return r_peaks


def check_expert_annotations(r_peaks, record_name='100'):
    annotation = wfdb.rdann(record_name, 'atr', pn_dir='mitdb', sampto=3600)
    non_beat_symbols = ['+', '~', '|', 'x', '"', '[', ']']

    # keep only true heartbeat timestamps
    expert_timestamps = [
        sample for sample, symbol in zip(annotation.sample, annotation.symbol)
        if symbol not in non_beat_symbols
    ]
    
    true_positives = 0
    tolerance = 5
    for exp in expert_timestamps:
        if any(abs(r - exp) <= tolerance for r in r_peaks):
            true_positives += 1

    sensitivity = (true_positives / len(expert_timestamps)) * 100
    
    return sensitivity




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

    # 3. run peak detection & evaluation before plotting
    r_peaks = adaptive_peak_detection(integrated, fs, clean_ecg)
    sensitivity = check_expert_annotations(r_peaks)

    time_axis = np.arange(len(raw_signal)) / fs

    # Fig 1: 5-panel visualization plot
    fig1, axes = plt.subplots(5, 1, figsize=(12, 10), sharex=True)

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

    fig1.tight_layout()

    # Fig 2: Detected R-Peaks Plot 
    fig2 = plt.figure(figsize=(12, 5))
    plt.plot(
        time_axis, clean_ecg, color="black", label="Cleaned ECG", alpha=0.8
    )

    # overlay detected R-peaks as red dots
    r_peaks_arr = np.array(r_peaks)
    plt.plot(
        time_axis[r_peaks_arr],
        clean_ecg[r_peaks_arr],
        "ro",
        markersize=7,
        label="Detected R-Peaks",
    )

    plt.title(
        f"Pan-Tompkins R-Peak Detection (Sensitivity: {sensitivity:.1f}%)",
        fontsize=12,
    )
    plt.xlabel("Time (seconds)")
    plt.ylabel("Voltage (mV)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="upper right")
    fig2.tight_layout()
    plt.show()