import cv2
import pandas as pd
import numpy as np
from data_normalization import *
from logger import setup_logger

logger = setup_logger(__name__)


def drop_bone_data(df):
    """
    Removes lower threshold duplicates (we assume bone result is better than air)
    """
    duplicates = df[df.duplicated(subset=['Frequency (Hz)', 'Ear'], keep=False)]
    air_data = duplicates.loc[duplicates.groupby(['Frequency (Hz)', 'Ear'])['Threshold (dB HL)'].idxmax()]
    bone_data = duplicates.drop(air_data.index)
    return df.drop(bone_data.index)


def find_points(mask: np.ndarray, image: np.ndarray, ear_label: str, color: tuple) -> list:
    """
    Detects audiogram points using contour detection.

    Returns a list of pixel coordinates.
    """
    points = []
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if 3 < w and 3 < h:
            center_x = x + w / 2
            center_y = y + h / 2
            points.append([center_x, center_y, ear_label])
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
    lower_blue, upper_blue = np.array([110, 50, 100]), np.array([130, 255, 255])

    mask_red = cv2.inRange(hsv, lower_red1, upper_red1) + cv2.inRange(hsv, lower_red2, upper_red2)
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

    right_ear_data = find_points(mask_red, image, "Right Ear", (0, 255, 0))
    left_ear_data = find_points(mask_blue, image, "Left Ear", (255, 0, 0))

    all_data = right_ear_data + left_ear_data

    df = pd.DataFrame(all_data, columns=["Frequency (Hz)", "Threshold (dB HL)", "Ear"])
    df = normalize(df, image)
    df = drop_bone_data(df)

    logger.info("\n%s", df)
    logger.info("Total detected points: %d", df.shape[0])
    df.to_csv(output_csv, index=False)

    # cv2.imshow("Detected Points", image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()