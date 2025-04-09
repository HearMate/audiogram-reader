import cv2
import numpy as np
import pandas as pd
from werkzeug.datastructures import FileStorage


def map_x_to_freq(x: int, width: int) -> int:
    """
    Maps X coordinates from the image to frequency values (Hz).
    The mapping is based on evenly spaced positions on the X-axis corresponding to known frequencies.

    x: int - X coordinate in the image
    width: int - Width of the image
    return: int - Interpolated frequency value
    """
    freq_ticks = [125, 250, 500, 1000, 2000, 4000, 8000]
    x_positions = np.linspace(0, width, len(freq_ticks))
    return int(np.interp(x, x_positions, freq_ticks))


def map_y_to_db(y: int, height: int) -> int:
    """
    Maps Y coordinates from the image to hearing levels (dB).
    The mapping assumes -10 dB at the top of the image and 120 dB at the bottom.

    y: int - Y coordinate in the image
    height: int - Height of the image
    return: int - Interpolated dB level
    """
    db_min, db_max = -10, 120
    return int(np.interp(y, [0, height], [db_min, db_max]))


def find_points(mask: np.ndarray, image: np.ndarray, ear_label: str, color: tuple) -> list:
    """
    Detects points representing hearing thresholds on the audiogram.
    Uses contour detection to find points and maps their coordinates to frequency and dB values.

    mask: np.ndarray - Binary mask of detected color areas
    image: np.ndarray - Original image for visualization
    ear_label: str - Label indicating whether it's left or right ear
    color: tuple - RGB color used for visualization
    return: list - List of detected points with frequency, dB, and ear label
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


def process_audiogram(image_path: str, output_csv_right: str, output_csv_left: str) -> None:
    """
    Processes an audiogram image, detects hearing threshold points, and saves extracted data to CSV files.

    image_path: str - Path to the audiogram image
    output_csv_right: str - Path to the output CSV file for the right ear
    output_csv_left: str - Path to the output CSV file for the left ear
    return: None
    """
    image = cv2.imread(image_path)  # Load image
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)  # Convert image to HSV color space

    lower_red1, upper_red1 = np.array([0, 50, 50]), np.array([10, 255, 255])
    lower_red2, upper_red2 = np.array([170, 50, 50]), np.array([180, 255, 255])
    lower_blue, upper_blue = np.array([70, 30, 50]), np.array([140, 255, 255])

    mask_red = cv2.inRange(hsv, lower_red1, upper_red1) + cv2.inRange(hsv, lower_red2, upper_red2)
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

    right_ear_data = find_points(mask_red, image, "Right Ear", (0, 255, 0))
    left_ear_data = find_points(mask_blue, image, "Left Ear", (255, 0, 0))

    pd.DataFrame(right_ear_data, columns=['Frequency_Hz', 'Hearing_Level_dB', 'Ear']).to_csv(output_csv_right,
                                                                                             index=False)
    pd.DataFrame(left_ear_data, columns=['Frequency_Hz', 'Hearing_Level_dB', 'Ear']).to_csv(output_csv_left,
                                                                                            index=False)

    cv2.imshow("Detected Points", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def process_audiogram_image(file: FileStorage):
    """
    Processes an audiogram image, detects hearing threshold points, and saves extracted data to CSV files.

    file: FileStorage - POST file from request
    return: DataFrame, DataFrame - data from both ears
    """
    file_bytes = file.read()

    # Convert to NumPy array and decode with OpenCV
    npimg = np.frombuffer(file_bytes, np.uint8)
    image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)  # Convert image to HSV color space

    lower_red1, upper_red1 = np.array([0, 50, 50]), np.array([10, 255, 255])
    lower_red2, upper_red2 = np.array([170, 50, 50]), np.array([180, 255, 255])
    lower_blue, upper_blue = np.array([70, 30, 50]), np.array([140, 255, 255])

    mask_red = cv2.inRange(hsv, lower_red1, upper_red1) + cv2.inRange(hsv, lower_red2, upper_red2)
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

    right_ear_data = find_points(mask_red, image, "Right Ear", (0, 255, 0))
    left_ear_data = find_points(mask_blue, image, "Left Ear", (255, 0, 0))

    right_ear_data = pd.DataFrame(right_ear_data, columns=['Frequency_Hz', 'Hearing_Level_dB', 'Ear'])
    left_ear_data = pd.DataFrame(left_ear_data, columns=['Frequency_Hz', 'Hearing_Level_dB', 'Ear'])

    return right_ear_data, left_ear_data
