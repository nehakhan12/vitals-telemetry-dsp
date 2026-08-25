# this file triggers alarms based on automated thresholds to find heart anomalies

import numpy as np
from .data_loader import load_ecg_record
from .dsp_filters import bandpass_filter
from .qrs_detector import pan_tompkins_preprocessing, adaptive_peak_detection
from .metrics import calculate_metrics
import matplotlib.pyplot as plt


def clinical_status_flags(start_time, bpm, rr_interval, median_rr):
    # check for any anamolies according to thresholds
    tachy_strikes = 0
    brady_strikes = 0

    if len(bpm) > 0:
        if np.mean(bpm) > 100:
            tachy_strikes = 1
        elif np.mean(bpm) < 60:
            brady_strikes = 1
    if len(rr_interval) > 0:
        for r in rr_interval:
            if (median_rr - r) / median_rr > 0.20:
                print(f"PVC event flagged at {start_time}")
    
    return tachy_strikes, brady_strikes


def process_telemetry_stream(bpm, rr_intervals, peak_times_sec, median_rr):
    tachy_strikes = 0
    brady_strikes = 0

    for start_time in range(0, int(peak_times_sec[-1]), 2): # loop through 2 seconds of data at a time
        end_time = start_time + 2

        valid_mask = (peak_times_sec >= start_time) & (peak_times_sec < end_time)
        window_bpm = bpm[valid_mask]
        window_rr_intervals = rr_intervals[valid_mask]

        avg_bpm = np.mean(window_bpm) if len(window_bpm) > 0 else 0
        print(f"[Time: {start_time:03d}s - {end_time:03d}s] Avg BPM: {avg_bpm:.1f}")

        t, b = clinical_status_flags(start_time, window_bpm, window_rr_intervals, median_rr)

        # check for tachycardia and bradycardia
        # needs to occur consecutively 3 times for it to send an alert
        if t == 1:
            tachy_strikes += 1
        elif t == 0 and len(window_bpm) > 0:
            tachy_strikes = 0

        if b == 1:
            brady_strikes += 1
        elif b == 0 and len(window_bpm) > 0:
            brady_strikes = 0
        
        if tachy_strikes == 3:
            print(f"Tachycardia detected at {start_time} seconds")
        elif brady_strikes == 3:
            print(f"Bradycardia detected at {start_time} seconds")


def run_alarm_pipeline(record_name="100", sample_limit=108000):
    raw_signal, fs = load_ecg_record(record_name, sample_limit=sample_limit)

    clean_ecg, f_raw, Pxx_raw, f_filt, Pxx_filt = bandpass_filter(
        sampling_freq=fs, f_nyq=fs/2, original_signal=raw_signal
    )

    # Pan-Tompkins preprocessing
    integrated, bandpass, derivative, squared = pan_tompkins_preprocessing(
        clean_ecg, fs
    )

    # run peak detection & evaluation
    r_peaks = adaptive_peak_detection(integrated, fs, clean_ecg)

    rr_intervals, bpm, sdnn, rmssd, window_times, peak_times_sec = calculate_metrics(fs, r_peaks)

    median_rr = np.median(rr_intervals)

    process_telemetry_stream(bpm, rr_intervals, peak_times_sec, median_rr)


if __name__ == "__main__":
    run_alarm_pipeline()