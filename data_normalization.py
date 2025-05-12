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

        plot_rect = find_bounding_box(image, ear_data)

        normalized_freq = normalize_hz(ear_data, plot_rect)
        normalized_thresh = normalize_db(ear_data, plot_rect)

        result = pd.DataFrame({"Frequency (Hz)": normalized_freq, "Threshold (dB HL)": normalized_thresh, "Ear": ear})
        result = result.sort_values(by="Frequency (Hz)")
        df.loc[df["Ear"] == ear, ["Frequency (Hz)", "Threshold (dB HL)", "Ear"]] = result.values

    return df


def normalize_hz(ear_data, plot_rect):
    """
    Maps ear_data frequency from pixel values to range 125 to 8000 Hz given audiogram coordinates
    """
    hz_mapping = {0: 125, 1: 125, 2: 250, 3: 250, 4: 500, 5: 750, 6: 1000, 7: 1500, 8: 2000, 9: 3000, 10: 4000,
                  11: 6000, 12: 8000}

    min_freq = ear_data["Frequency (Hz)"].min()
    max_freq = ear_data["Frequency (Hz)"].max()

    x, _, w, _ = plot_rect

    margin = min(abs(x-min_freq), abs(x+w-max_freq))

    left = x + margin
    right = x + w - margin
    normalized_freq = ((ear_data["Frequency (Hz)"] - left) * (12 / (right - left))).round()
    normalized_freq = [hz_mapping[int(x)] for x in normalized_freq]
    return normalized_freq


def normalize_db(ear_data, plot_rect):
    """
    Maps ear_data db from pixel values to range -10 to 125 dbHL given audiogram coordinates
    """
    db_mapping = [x for x in range(-10, 125, 5)]

    _, y, _, h = plot_rect
    # correction needed due to imperfect cv2 contour selection
    y += 5
    h -= 10

    top = y
    bottom = y + h
    normalized_thresh = np.round((ear_data["Threshold (dB HL)"] - top) * ((len(db_mapping) - 1) / (bottom - top)))
    normalized_thresh = [db_mapping[int(x)] for x in normalized_thresh]
    return normalized_thresh


def find_bounding_box(img, ear_data):
    """
    Finds the plotting area of audiogram
    """
    img = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    min_x = ear_data["Frequency (Hz)"].min()
    max_x = ear_data["Frequency (Hz)"].max()
    min_y = ear_data["Threshold (dB HL)"].min()
    max_y = ear_data["Threshold (dB HL)"].max()

    plot_rect = (min_x, min_y, max_x-min_x, max_y - min_y)

    # find contour containing all points
    plot_rect = find_min_width_valid_contour(contours, plot_rect)

    # find maximal contour with the same width - contour with correct height
    plot_rect = find_max_height_valid_contour(contours, plot_rect)

    # find maximal contour with the same height - contour with correct width
    plot_rect = find_max_width_valid_contour(contours, plot_rect, img)

    return plot_rect


def find_min_width_valid_contour(contours, plot_rect):
    """
    Finds the smallest in width contour containing points in plot_rect
    """
    min_viable_area = float('inf')
    min_x, min_y, w, h = plot_rect
    max_x = min_x + w
    max_y = min_y + h
    plot_rect = None
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        if area < min_viable_area and x < min_x and x + w > max_x and y < min_y and y + h > max_y:
            min_viable_area = area
            plot_rect = (x, y, w, h)

    return plot_rect


def find_max_height_valid_contour(contours, plot_rect):
    """
    Finds the tallest contour with width given in plot_rect
    """
    max_viable_area = 0
    x, y, w, h = plot_rect
    margin = 5

    left_border_min = x - margin
    left_border_max = x + margin
    right_border_min = x + w - margin
    right_border_max = x + w + margin

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        if area > max_viable_area and left_border_max > x > left_border_min and right_border_min < (
                x + w) < right_border_max:
            max_viable_area = area
            plot_rect = (x, y, w, h)

    return plot_rect


def find_max_width_valid_contour(contours, plot_rect, img):
    """
    Finds largest in width contour with height given in plot_rect
    """
    max_viable_area = 0
    x_original, y_original, w_original, h_original = plot_rect
    margin = 5
    top_border_min = y_original - margin
    top_border_max = y_original + margin
    bottom_border_min = y_original + h_original - margin
    bottom_border_max = y_original + h_original + margin

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h

        top_aligned = top_border_min < y < top_border_max
        bottom_aligned = bottom_border_min < (y + h) < bottom_border_max
        horizontally_overlapping = x < x_original + w_original and x + w > x_original

        if area > max_viable_area and top_aligned and bottom_aligned and horizontally_overlapping:
            max_viable_area = area
            plot_rect = (x, y, w, h)

    return plot_rect
