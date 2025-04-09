import sys
from audiogram_classify import classify_audiogram_type
import audiogram_parser_1
import audiogram_parser_2

def main(image_path: str, output_csv: str):
    """
    Main execution entry point. Classifies the audiogram type and dispatches
    to the appropriate parser depending on the result.

    Parameters:
        image_path (str): Path to the audiogram image.
        output_csv (str): Path to the output CSV file.
    """
    try:
        print(f"[INFO] Processing file: {image_path}")
        audiogram_type = classify_audiogram_type(image_path)
        print(f"[INFO] Detected audiogram type: {audiogram_type}")

        if audiogram_type == "type_1":
            print("[INFO] Executing parser for type_1 audiogram")
            audiogram_parser_1.run(image_path, output_csv)

        elif audiogram_type == "type_2":
            print("[INFO] Executing parser for type_2 audiogram")
            audiogram_parser_2.run(image_path, output_csv)

        else:
            print("[ERROR] Unable to classify audiogram. No parser executed.")

    except Exception as e:
        print(f"[ERROR] Exception occurred: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python main.py <audiogram_image_path> <output_csv_path>")
    else:
        main(sys.argv[1], sys.argv[2])
