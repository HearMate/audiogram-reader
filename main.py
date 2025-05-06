import sys
import logging
import logging
from audiogram_classify import classify_audiogram_type
import audiogram_parser_1
import audiogram_parser_2

logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(message)s'
)

def main(image_path: str, output_csv: str):
    """
    Main execution entry point. Classifies the audiogram type and dispatches
    to the appropriate parser depending on the result.

    Parameters:
        image_path (str): Path to the audiogram image.
        output_csv (str): Path to the output CSV file.
    """
    try:
        logging.info(f"Processing file: {image_path}")
        audiogram_type = classify_audiogram_type(image_path)
        logging.info(f"[INFO] Detected audiogram type: {audiogram_type}")

        if audiogram_type == "type_1":
            logging.info("[INFO] Executing parser for type_1 audiogram")
            audiogram_parser_1.run(image_path, output_csv)

        elif audiogram_type == "type_2":
            logging.info("[INFO] Executing parser for type_2 audiogram")
            audiogram_parser_2.run(image_path, output_csv)

        else:
            logging.error("[ERROR] Unable to classify audiogram. No parser executed.")

    except Exception as e:
        logging.exception(f"[ERROR] Exception occurred: {e}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        logging.error("Usage: python main.py <audiogram_image_path> <output_csv_path>")
    else:
        main(sys.argv[1], sys.argv[2])
