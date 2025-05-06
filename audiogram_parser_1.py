import cv2
import pandas as pd


    """
    """


def find_points(mask: np.ndarray, image: np.ndarray, ear_label: str, color: tuple) -> list:
    """

    """
    points = []
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
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

    mask_red = cv2.inRange(hsv, lower_red1, upper_red1) + cv2.inRange(hsv, lower_red2, upper_red2)
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

    right_ear_data = find_points(mask_red, image, "Right Ear", (0, 255, 0))
    left_ear_data = find_points(mask_blue, image, "Left Ear", (255, 0, 0))

    all_data = right_ear_data + left_ear_data

    df = pd.DataFrame(all_data, columns=["Frequency (Hz)", "Threshold (dB HL)", "Ear"])

    print(df)

