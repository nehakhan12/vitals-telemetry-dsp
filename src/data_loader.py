import numpy as np 
import scipy
import wfdb
import os


# get the absolute path of the current directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")


# loader function
def load_ecg_record(record_name="100", sample_limit=360):
    record_path = os.path.join(DATA_DIR, record_name)
    header_file = f"{record_path}.hea"

    if not os.path.exists(header_file):
        #file does not exist, triggers download
        print("Downloading data into", DATA_DIR)
        wfdb.dl_database(
            db_dir="mitdb", 
            dl_dir=DATA_DIR, 
            records=[record_name], 
            overwrite=False,
        )

    record = wfdb.rdrecord(record_path, sampto=sample_limit)
    # attributes = [attr for attr in dir(record) if not attr.startswith("_")]

    # get Lead II voltage signal from channel 0 and sampling frequency
    lead_ii_signal = record.p_signal[:, 0]
    sampling_freq = record.fs

    return lead_ii_signal, sampling_freq

if __name__=="__main__":
    signal, freq = load_ecg_record("100", sample_limit=3600)
    print(f"signal shape: {signal.shape}")
    print(f"sampling rate: {freq} Hz")
    print(f"first 5 voltage samples (mV) {signal[:5]}")