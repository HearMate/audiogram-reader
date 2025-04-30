import cv2
import numpy as np
import pandas as pd
from data_normalization import *

def map_x_to_freq(x: int, width: int) -> int:
    """
    Maps X coordinates from the image to frequency values (Hz).
    Based on evenly spaced positions corresponding to known frequencies.
    """
    return x
    freq_ticks = [125, 250, 500, 1000, 2000, 4000, 8000]
    x_positions = np.linspace(0, width, len(freq_ticks))
    return int(np.interp(x, x_positions, freq_ticks))


def map_y_to_db(y: int, height: int) -> int:
    """
    Maps Y coordinates from the image to hearing threshold levels in dB HL.
    Assumes -10 dB at the top and 120 dB at the bottom of the image.
    """
    return y
    return int(np.interp(y, [0, height], [-10, 120]))


def find_points(mask: np.ndarray, image: np.ndarray, ear_label: str, color: tuple) -> list:
    """
    Detects audiogram points using contour detection, maps their coordinates to dB and frequency.

    Returns a list of (frequency, dB, ear_label) values.
    """
    points = []
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    height, width = image.shape[:2]

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        if 3 < w < 40 and 3 < h < 40 and area > 10:
            freq = map_x_to_freq(x, width)
            db = map_y_to_db(y, height)
            points.append([freq, db, ear_label])
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)

    return points


def run(image_path: str, output_csv: str) -> None:
    """
    Parses a 'type_1' audiogram image, detects threshold points, and saves to a combined CSV.

    Parameters:
        image_path (str): Path to input image.
        output_csv (str): Path to save CSV file.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_red1, upper_red1 = np.array([0, 50, 50]), np.array([10, 255, 255])
    lower_red2, upper_red2 = np.array([170, 50, 50]), np.array([180, 255, 255])
    lower_blue, upper_blue = np.array([70, 30, 50]), np.array([140, 255, 255])

    mask_red = cv2.inRange(hsv, lower_red1, upper_red1) + cv2.inRange(hsv, lower_red2, upper_red2)
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

    right_ear_data = find_points(mask_red, image, "Right Ear", (0, 255, 0))
    left_ear_data = find_points(mask_blue, image, "Left Ear", (255, 0, 0))

    all_data = right_ear_data + left_ear_data

    df = pd.DataFrame(all_data, columns=["Frequency (Hz)", "Threshold (dB HL)", "Ear"])
    df.to_csv(output_csv, index=False)

    # print(f"[INFO] Saved combined results to: {output_csv}")
    # print(df)
    normalize(df, image)

    cv2.imshow("Detected Points", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
