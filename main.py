import sys
from audiogram_classify import classify_audiogram_type
import  audiogram.audiogram_parser_1 as audiogram_parser_1
import  audiogram.audiogram_parser_2 as audiogram_parser_2
from logger import setup_logger

logger = setup_logger(__name__)

def main(image_path: str, output_csv: str):
    """
    Main execution entry point. Classifies the audiogram type and dispatches
    to the appropriate parser depending on the result.

    Parameters:
        image_path (str): Path to the audiogram image.
        output_csv (str): Path to the output CSV file.
    """
    try:
        logger.info(f"Processing file: {image_path}")
        audiogram_type = classify_audiogram_type(image_path)
        logger.info(f"[INFO] Detected audiogram type: {audiogram_type}")

        if audiogram_type == "type_1":
            logger.info("[INFO] Executing parser for type_1 audiogram")
            audiogram_parser_1.run(image_path, output_csv)

        elif audiogram_type == "type_2":
            logger.info("[INFO] Executing parser for type_2 audiogram")
            audiogram_parser_2.run(image_path, output_csv)

        else:
            logger.error("[ERROR] Unable to classify audiogram. No parser executed.")

    except Exception as e:
        logger.exception(f"[ERROR] Exception occurred: {e}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        logger.error("Usage: python main.py <audiogram_image_path> <output_csv_path>")
    else:
        main(sys.argv[1], sys.argv[2])
