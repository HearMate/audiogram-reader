import cv2
import pandas as pd
import numpy as np


def normalize(df, image):
    """
    Given image and pixel coordinates of data points maps them to 125-8kHz and -10-120dBHL
    """
    ears = df["Ear"].unique()
    for ear in ears:
        ear_data = df[df["Ear"] == ear]

        normalized_freq = normalize_hz(ear_data)
        normalized_thresh = normalize_db(ear_data, image)

        result = pd.DataFrame({"Frequency (Hz)": normalized_freq, "Threshold (dB HL)": normalized_thresh, "Ear": ear})
        result = result.sort_values(by="Frequency (Hz)")
        df.loc[df["Ear"] == ear, ["Frequency (Hz)", "Threshold (dB HL)", "Ear"]] = result.values

    return df


def normalize_hz(ear_data):
    hz_mapping = {0: 125, 1: 125, 2: 250, 3: 250, 4: 500, 5: 750, 6: 1000, 7: 1500, 8: 2000, 9: 3000, 10: 4000,
                  11: 6000, 12: 8000}

    min_freq = ear_data["Frequency (Hz)"].min()
    max_freq = ear_data["Frequency (Hz)"].max()

    normalized_freq = ((ear_data["Frequency (Hz)"] - min_freq) * (12 / (max_freq - min_freq))).round()
    normalized_freq = [hz_mapping[int(x)] for x in normalized_freq]
    return normalized_freq


def normalize_db(ear_data, image):
    db_mapping = [x for x in range(-10, 125, 5)]

    _, _, top, bottom = find_bounding_box(image, ear_data)

    normalized_thresh = np.round((ear_data["Threshold (dB HL)"] - top) * ((len(db_mapping) - 1) / (bottom - top)))
    normalized_thresh = [db_mapping[int(x)] for x in normalized_thresh]
    return normalized_thresh


def find_bounding_box(img, ear_data):
    """
    Finds the top and bottom border of the plotting space in one graph
    """
    img = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    plot_rect = None
    min_viable_area = float('inf')

    min_x = ear_data["Frequency (Hz)"].min()
    max_x = ear_data["Frequency (Hz)"].max()
    min_y = ear_data["Threshold (dB HL)"].min()
    max_y = ear_data["Threshold (dB HL)"].max()

    # find minimal space containing all datapoints
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        if area < min_viable_area and x < min_x and x + w > max_x and y < min_y and y + h > max_y:
            min_viable_area = area
            plot_rect = (x, y, w, h)

    max_viable_area = 0
    x, y, w, h = plot_rect
    margin = 5
    left_border_min = x - margin
    left_border_max = x + margin
    right_border_min = x + w - margin
    right_border_max = x + w + margin

    # find maximal space with previously found width
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        if area > max_viable_area and left_border_max > x > left_border_min and right_border_min < (x + w) < right_border_max:
            max_viable_area = area
            y += 5
            h -= 5
            plot_rect = (x, y, w, h)
            # cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # cv2.imshow("Detected Plot Area", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    x, y, w, h = plot_rect
    return x, x+w, y, y+h
