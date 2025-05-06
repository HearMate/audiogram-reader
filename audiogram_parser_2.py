import cv2
import numpy as np
import pandas as pd
from logger import setup_logger


logger = setup_logger(__name__)

FREQUENCIES = ['125', '250', '500', '1k', '2k', '4k', '8k']
DB_MIN, DB_MAX = -10, 120
Y_VALID_RANGE = (30, 130)


def map_y_to_db(y: int, height: int) -> int:
    """
    Maps vertical Y-pixel coordinates to hearing threshold values in dB HL.
    Top of the image = -10 dB, bottom = 120 dB.
    """
    return int(np.interp(y, [0, height], [DB_MIN, DB_MAX]))


def enhance_saturation(image: np.ndarray, factor: float = 2.5) -> np.ndarray:
    """
    Boosts saturation to make colors (red and blue markers) more prominent.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype("float32")
    h, s, v = cv2.split(hsv)
    s = np.clip(s * factor, 0, 255)
    enhanced = cv2.merge([h, s, v]).astype("uint8")
    return cv2.cvtColor(enhanced, cv2.COLOR_HSV2BGR)


def detect_red_circles(image: np.ndarray) -> list:
    """
    Detects red circular markers representing right ear thresholds.
    """
    hsv = cv2.cvtColor(enhance_saturation(image), cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, 70, 50), (10, 255, 255)) | \
        cv2.inRange(hsv, (160, 70, 50), (180, 255, 255))
    blurred = cv2.GaussianBlur(mask, (9, 9), 2)
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=25,
                               param1=100, param2=10, minRadius=4, maxRadius=15)
    points = []
    if circles is not None:
        for (x, y, _) in np.round(circles[0, :]).astype("int"):
            if Y_VALID_RANGE[0] < y < Y_VALID_RANGE[1]:
                points.append((x, y))
    return sorted(points, key=lambda p: p[0])[:7]


def detect_blue_crosses(image: np.ndarray) -> list:
    """
    Detects blue cross markers representing left ear thresholds.
    """
    hsv = cv2.cvtColor(enhance_saturation(image), cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (100, 150, 0), (140, 255, 255))
    corners = cv2.goodFeaturesToTrack(mask, maxCorners=50, qualityLevel=0.01, minDistance=20)
    points = []
    if corners is not None:
        for i in np.intp(corners):
            x, y = i.ravel()
            if 50 < y < 120:
                points.append((x, y))
    return sorted(points, key=lambda p: p[0])[:7]


class PointEditor:
    """
    Interactive point editor using OpenCV windows.
    Allows user to manually correct points with mouse clicks.
    """

    def __init__(self, window_name: str, image: np.ndarray, color: str, points: list):
        self.window_name = window_name
        self.base_image = image.copy()
        self.display_image = image.copy()
        self.color = color
        self.points = points[:]

    def _draw_points(self) -> None:
        self.display_image = self.base_image.copy()
        for (x, y) in self.points:
            if self.color == 'red':
                cv2.circle(self.display_image, (x, y), 7, (0, 255, 0), 2)
            else:
                cv2.drawMarker(self.display_image, (x, y), (0, 255, 255),
                               markerType=cv2.MARKER_TILTED_CROSS, markerSize=15, thickness=2)
        cv2.imshow(self.window_name, self.display_image)

    def _on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.points.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN and self.points:
            closest = min(self.points, key=lambda p: (p[0] - x) ** 2 + (p[1] - y) ** 2)
            self.points.remove(closest)
        self.points = sorted(self.points, key=lambda p: p[0])[:7]
        self._draw_points()

    def edit(self) -> list:
        """
        Launches OpenCV window for manual point editing.
        ESC or closing the window ends editing.
        """
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self._on_mouse)
        self._draw_points()
        while True:
            if cv2.waitKey(1) == 27 or cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) < 1:
                break
        if cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) >= 1:
            cv2.destroyWindow(self.window_name)
        while len(self.points) < 7:
            self.points.append((None, None))
        return self.points


def run(image_path: str, output_csv: str) -> None:
    """
    Entry point for type_2 audiogram parsing with detection and manual correction.
    Saves a single CSV with both ears.

    Args:
        image_path (str): Path to the audiogram image
        output_csv (str): Path to save the output CSV
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"{image_path} not found.")

    height, width, _ = image.shape
    right_img = image[:, :width // 2]
    left_img = image[:, width // 2:]

    right_points = detect_red_circles(right_img)
    left_points = detect_blue_crosses(left_img)

    logger.info("Left-click = add, Right-click = remove. ESC or close window to finish.")

    right_points = PointEditor("Right Ear – Red Circles", right_img, 'red', right_points).edit()
    left_points = PointEditor("Left Ear – Blue Crosses", left_img, 'blue', left_points).edit()

    data = []
    for (x, y) in right_points:
        if y is not None:
            data.append(["Right Ear", map_y_to_db(y, height)])
    for (x, y) in left_points:
        if y is not None:
            data.append(["Left Ear", map_y_to_db(y, height)])

    df = pd.DataFrame(data, columns=["Ear", "Threshold (dB HL)"])
    df.insert(0, "Frequency (Hz)", FREQUENCIES * 2 if len(df) > 7 else FREQUENCIES)

    df.to_csv(output_csv, index=False)
    logger.info(f"[INFO] Saved results to: {output_csv}")
    logger.info(df)
