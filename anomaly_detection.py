def detect_anomaly(df):
    status = 'OK'
    ears = df["Ear"].unique()
    for ear in ears:
        ear_data = df[df["Ear"] == ear]
        thresholds = ear_data['Threshold (dB HL)'].reset_index(drop=True)
        frequencies = ear_data['Frequency (Hz)'].reset_index(drop=True)

        for frequency in frequencies:
            if frequency < 125 or frequency > 8000:
                status = 'Error'
                return status

        for threshold in thresholds:
            if threshold > 120 or threshold < -10:
                status = 'Error'
                return status

        for i in range(len(thresholds)-2):
            if abs(thresholds[i] - thresholds[i+1]) > 40:
                status = 'Suspect'

    return status
