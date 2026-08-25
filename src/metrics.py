import numpy as np
from data_loader import load_ecg_record
from dsp_filters import bandpass_filter
from qrs_detector import pan_tompkins_preprocessing, adaptive_peak_detection
import matplotlib.pyplot as plt


def calculate_metrics(fs, r_peaks):
    rr_intervals = []
    bpm = []
    for i in range(1, len(r_peaks)):
        time_diff = (r_peaks[i] - r_peaks[i - 1]) * (1000 / fs)
        rr_intervals.append(time_diff)
        bpm.append(60000 / time_diff)

    peak_times_sec = np.array(r_peaks[1:]) / fs
    rr_intervals_arr = np.array(rr_intervals)
    bpm_arr = np.array(bpm)

    # filter ectopic beats / jump artifacts (>20% deviation from median)
    median_rr = np.median(rr_intervals_arr)
    valid_mask = np.abs(rr_intervals_arr - median_rr) / median_rr <= 0.20

    rr_intervals_arr = rr_intervals_arr[valid_mask]
    peak_times_sec = peak_times_sec[valid_mask]
    bpm_arr = bpm_arr[valid_mask]

    sdnn = []
    rmssd = []
    window_times = []
    total_duration = int(peak_times_sec[-1])
    
    # 30 second sliding window
    for start_time in range(0, total_duration - 29, 1):
        end_time = start_time + 30
        mask = (peak_times_sec >= start_time) & (peak_times_sec < end_time)
        window_intervals = rr_intervals_arr[mask]

        #check if there is more than one heartbeat first then calculate
        if len(window_intervals) > 1:
            sdnn.append(np.std(window_intervals))
            diffs = np.diff(window_intervals)
            rmssd.append(np.sqrt(np.mean(diffs**2)))
            window_times.append(start_time + 15)

    return (
        rr_intervals_arr,
        bpm_arr,
        sdnn,
        rmssd,
        window_times,
        peak_times_sec,
    )


if __name__ == "__main__":
    raw_signal, fs = load_ecg_record("100", sample_limit=108000)

    clean_ecg, f_raw, Pxx_raw, f_filt, Pxx_filt = bandpass_filter(
        sampling_freq=fs, f_nyq=fs/2, original_signal=raw_signal
    )

    # Pan-Tompkins preprocessing
    integrated, bandpass, derivative, squared = pan_tompkins_preprocessing(
        clean_ecg, fs
    )

    # 3. run peak detection & evaluation before plotting
    r_peaks = adaptive_peak_detection(integrated, fs, clean_ecg)

    rr_intervals, bpm, sdnn, rmssd, window_times, peak_times_sec = calculate_metrics(fs, r_peaks)


    # Console Output Summary 
    print("\n" + "="*45)
    print("      VITAL SIGNS & HRV METRICS      ")
    print("="*45)
    print(f"Total Beats Detected : {len(r_peaks)}")
    print(f"Mean Heart Rate      : {np.mean(bpm):.1f} BPM")
    print(f"Mean RR-Interval     : {np.mean(rr_intervals):.1f} ms")
    print(f"Overall SDNN (30s)   : {np.mean(sdnn):.2f} ms")
    print(f"Overall RMSSD (30s)  : {np.mean(rmssd):.2f} ms")
    print("="*45 + "\n")

    # Line Plot: Real-Time BPM & HRV Metrics
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Top Plot: Instantaneous Heart Rate (BPM)
    axes[0].plot(peak_times_sec, bpm, color="crimson", linewidth=1.2, marker="o", markersize=3, label="Instantaneous BPM")
    axes[0].axhline(np.mean(bpm), color="black", linestyle="--", alpha=0.7, label=f"Mean BPM ({np.mean(bpm):.1f})")
    axes[0].set_title("Instantaneous Heart Rate (BPM) Across 5-Minute Trace", fontsize=12)
    axes[0].set_ylabel("Heart Rate (BPM)")
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend(loc="upper right")

    # Bottom Plot: Time-Domain HRV Metrics (SDNN & RMSSD over 30s Rolling Windows)
    axes[1].plot(window_times, sdnn, color="navy", linewidth=2, label="SDNN (Total Variability)")
    axes[1].plot(window_times, rmssd, color="teal", linewidth=2, label="RMSSD (Short-Term Variability)")
    axes[1].set_title("Sliding 30-Second Window HRV Metrics (SDNN & RMSSD)", fontsize=12)
    axes[1].set_xlabel("Time (seconds)")
    axes[1].set_ylabel("Variability (ms)")
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].legend(loc="upper right")

    plt.tight_layout()
    plt.show()
