# ECG Telemetry & Real-Time Arrhythmia Monitoring System

## Problem & Motivation
Cardiovascular diseases remain the leading cause of global mortality, with cardiac arrhythmias often acting as early indicators of life-threatening events. While continuous Electrocardiogram (ECG) monitoring is critical for early detection, raw medical sensor streams are full of noise, such as muscle movement, respiration baseline wander, and 60 Hz powerline interference. On the other hand, manual inspection of high-volume telemetry data by clinicians is time-consuming and prone to human error.

This project addresses these challenges by building an automated software pipeline to ingest raw, noisy time-series ECG data, filtering out environmental artifacts in real time, and extracting the QRS complexes. By transforming raw voltage signals into metrics like instantaneous heart rate (BPM) and heart rate variability (HRV), this program provides automated clinical alerts for heart anomalies like Tachycardia and Bradycardia, and includes rapid telemetry visualization.

## Key Features

* Digital Signal Processing (DSP) Pipeline:
  * Bandpass Butterworth Filtering (5-15Hz) to isolate QRS energy and remove muscle noise or baseline wander
  * 60 Hz Notch Filtering for interference removal
  * Continuous derivative filtering, signal squaring, and moving-window integration (150ms)
* Adaptive QRS & R-Peak Detection:
  * Real-time dual-thresholding tracking signal (SPKI) and noise (NPKI) levels.
  * Refinement windowing to map integrated energy envelope peaks back to true max voltage coordinates on the cleaned ECG trace
* Vital Signs & HRV Metrics:
  * Calculation of instantaneous BPM and RR-intervals
  * Ectopic beat and jump-artifact filtering
  * Time-domain HRV analysis (SDNN and RMSSD) using 30-second sliding windows
* Animated Telemetry Dashboard:
  * Live Matplotlib dashboard showcasing scrolling 5-second ECG traces, aligned R-peak scatter markers, and real-time heart rate trends
  * Integrated clinical alarm engine detecting continuous Tachycardia (>100 BPM) and Bradycardia (<60 BPM) flags
* Ground-Truth Validation:
  * Benchmark verification against PhysioNet expert annotations ('atr'), returning the sensitivity metrics per record


## Project Architecture

```text
├── data/                      # Local cache for MIT-BIH record files
├── src/
│   ├── data_loader.py         # WFDB database loader & signal ingestion
│   ├── dsp_filters.py         # Bandpass & notch filtering functions
│   ├── qrs_detector.py        # Pan-Tompkins pipeline & peak detection
│   ├── metrics.py            # RR-interval, BPM, SDNN & RMSSD processing
│   └── alarm_engine.py       # Clinical thresholding and status logic
├── main.py                    # Animated telemetry dashboard entry point
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation
```

## Installation steps
- Clone the repository
- Create and activate python virtual environment
  ```text
  # macOS/Linux
  python3 -m venv venv
  source venv/bin/activate

  # Windows
  python -m venv venv
  venv\Scripts\activate
  ```
- Install requirements: `pip install -r requirements.txt`

## Usage Guide
- To start real-time telemetry display on MIT-BIH Record 100: `python main.py`

Running individual modules:
  - Test DSP Filtering & Peak Detection: `python src/qrs_detector.py`
    - Generates the 5-step Pan-Tompkins breakdown figure alongside detected R-peaks and annotation sensitivity metrics
  - Test HRV Metrics & Vital Signs Summary: `python src/metrics.py`
    - Prints total beats detected, mean BPM, SDNN, and RMSSD statistics to the console

## Live Telemetry Sample
https://github.com/user-attachments/assets/bd47232f-cfe1-446a-b92f-8caf0e645d05

  
