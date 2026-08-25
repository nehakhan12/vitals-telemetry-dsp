# Real-Time ECG Telemetry & Arrhythmia Dashboard
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from src.data_loader import load_ecg_record
from src.dsp_filters import bandpass_filter
from src.qrs_detector import pan_tompkins_preprocessing, adaptive_peak_detection
from src.alarm_engine import clinical_status_flags


def prepare_pipeline_data(record_name="100", sample_limit=108000):
    # runs signal processing pipeline and extracts raw metrics for telemetry
    raw_signal, fs = load_ecg_record(record_name, sample_limit=sample_limit)

    clean_ecg, _, _, _, _ = bandpass_filter(
        sampling_freq=fs, f_nyq=fs / 2, original_signal=raw_signal
    )

    integrated, bandpass, derivative, squared = pan_tompkins_preprocessing(
        clean_ecg, fs
    )

    r_peaks = adaptive_peak_detection(integrated, fs, clean_ecg)

    # calculate raw intervals and BPM for streaming (unfiltered)
    raw_rr_intervals = np.diff(r_peaks) * (1000 / fs)
    raw_bpm = 60000 / raw_rr_intervals
    peak_times_sec = np.array(r_peaks[1:]) / fs
    median_rr = np.median(raw_rr_intervals)

    return clean_ecg, fs, raw_rr_intervals, raw_bpm, peak_times_sec, median_rr


def run_animated_dashboard():
    # Run preprocessing via pipeline helper 
    clean_ecg, fs, raw_rr_intervals, raw_bpm, peak_times_sec, median_rr = (
        prepare_pipeline_data(record_name="100", sample_limit=108000)
    )

    total_time_sec = len(clean_ecg) / fs
    time_axis = np.arange(len(clean_ecg)) / fs

    # Set up Matplotlib Layout
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(11, 6), gridspec_kw={"height_ratios": [2.5, 1]}
    )
    fig.canvas.manager.set_window_title("MIT-BIH Record 100 - Live ECG Telemetry Monitor")

    window_size_sec = 5.0

    line_ecg, = ax1.plot([], [], lw=1.5, color="#1f77b4", label="ECG Signal")
    peak_scatter = ax1.scatter([], [], color="red", s=40, zorder=5, label="R-Peak")
    line_bpm, = ax2.plot([], [], lw=1.8, color="#2ca02c", label="Instantaneous BPM")

    ax1.set_ylabel("Voltage (mV)")
    ax1.set_title("Live ECG Signal & Peak Detection (Record 100)")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.set_ylim(np.min(clean_ecg) - 0.2, np.max(clean_ecg) + 0.3)
    ax1.legend(loc="upper left")

    status_text = ax1.text(
        0.98,
        0.90,
        "STATUS: NORMAL",
        transform=ax1.transAxes,
        fontsize=12,
        fontweight="bold",
        ha="right",
        va="top",
        bbox=dict(
            boxstyle="round,pad=0.5",
            facecolor="green",
            alpha=0.2,
            edgecolor="green",
        ),
    )

    ax2.set_xlabel("Time (seconds)")
    ax2.set_ylabel("BPM")
    ax2.set_title("Heart Rate Trend")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.set_ylim(40, 150)
    ax2.axhline(100, color="red", linestyle=":", alpha=0.7, label="Tachycardia Threshold (>100)")
    ax2.axhline(60, color="blue", linestyle=":", alpha=0.7, label="Bradycardia Threshold (<60)")
    ax2.legend(loc="lower left", fontsize=8)

    state = {
        "tachy_strikes": 0,
        "brady_strikes": 0,
        "active_status": "STATUS: NORMAL",
        "status_color": "green",
    }

    # Frame update callback
    def update(frame):
        current_time = frame * 0.04
        t_start = current_time
        t_end = current_time + window_size_sec

        if t_end > total_time_sec:
            return line_ecg, peak_scatter, line_bpm, status_text

        sig_mask = (time_axis >= t_start) & (time_axis <= t_end)
        visible_time = time_axis[sig_mask]
        visible_sig = clean_ecg[sig_mask]

        peak_mask = (peak_times_sec >= t_start) & (peak_times_sec <= t_end)
        visible_peaks_time = peak_times_sec[peak_mask]

        visible_peaks_volts = []
        for p_t in visible_peaks_time:
            idx = int(p_t * fs)
            if idx < len(clean_ecg):
                visible_peaks_volts.append(clean_ecg[idx])

        bpm_history_mask = peak_times_sec <= t_end
        history_time = peak_times_sec[bpm_history_mask]
        history_bpm = raw_bpm[bpm_history_mask]

        line_ecg.set_data(visible_time, visible_sig)
        peak_scatter.set_offsets(
            np.c_[visible_peaks_time, visible_peaks_volts]
            if len(visible_peaks_time) > 0
            else np.empty((0, 2))
        )
        line_bpm.set_data(history_time, history_bpm)

        ax1.set_xlim(t_start, t_end)
        ax2.set_xlim(max(0, t_end - 20), max(20, t_end))

        recent_mask = (peak_times_sec >= max(0, t_end - 2)) & (peak_times_sec <= t_end)
        recent_bpm = raw_bpm[recent_mask]
        recent_rr = raw_rr_intervals[recent_mask]

        t_flag, b_flag = clinical_status_flags(
            t_end, recent_bpm, recent_rr, median_rr
        )

        if t_flag == 1:
            state["tachy_strikes"] += 1
        elif t_flag == 0 and len(recent_bpm) > 0:
            state["tachy_strikes"] = 0

        if b_flag == 1:
            state["brady_strikes"] += 1
        elif b_flag == 0 and len(recent_bpm) > 0:
            state["brady_strikes"] = 0

        if state["tachy_strikes"] >= 3:
            state["active_status"] = "ALERT: TACHYCARDIA"
            state["status_color"] = "red"
        elif state["brady_strikes"] >= 3:
            state["active_status"] = "ALERT: BRADYCARDIA"
            state["status_color"] = "blue"
        else:
            state["active_status"] = "STATUS: NORMAL"
            state["status_color"] = "green"

        status_text.set_text(state["active_status"])
        status_text.set_bbox(
            dict(
                boxstyle="round,pad=0.5",
                facecolor=state["status_color"],
                alpha=0.3,
                edgecolor=state["status_color"],
            )
        )

        return line_ecg, peak_scatter, line_bpm, status_text

    # Run animation
    num_frames = int((total_time_sec - window_size_sec) / 0.04)

    # Set interval to 40ms (matches 25 FPS real-time speed)
    ani = animation.FuncAnimation(
        fig, 
        update, 
        frames=num_frames, 
        interval=40,   # 40ms delay between frames = 25 FPS real-time speed
        blit=False, 
        repeat=False
    )
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_animated_dashboard()