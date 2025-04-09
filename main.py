from audiogram_parser import *
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/', methods=['GET'])
def home():
    return "Flask server is running!"


@app.route('/run-script', methods=['POST'])
def run_script():
    data = request.json
    image_path = data.get('image_path')
    output_csv_right = data.get('output_csv_right')
    output_csv_left = data.get('output_csv_left')

    if not all([image_path, output_csv_right, output_csv_left]):
        return jsonify({'error': 'Missing parameters'}), 400

    process_audiogram(image_path, output_csv_right, output_csv_left)
    return "ok"


@app.route('/upload-image', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return 'no image', 400

    file = request.files['image']
    right_ear_df, left_ear_df = process_audiogram_image(file)

    right_ear_json = right_ear_df.to_dict(orient='records')
    left_ear_json = left_ear_df.to_dict(orient='records')

    return jsonify({
        'right_ear': right_ear_json,
        'left_ear': left_ear_json
    })


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)
