import json
from flask import Flask, request
from flask_cors import CORS

from audiogram_parser_1 import *
from anomaly_detection import *

app = Flask(__name__)
CORS(app)


@app.route('/', methods=['GET'])
def home():
    return "Flask server is running!"


@app.route('/upload-image', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return 'no image', 400

    file = request.files['image']
    df = process_audiogram_image(file)
    anomaly_status = detect_anomaly(df)
    result = df.groupby('Ear').apply(
        lambda g: dict(zip(g['Frequency (Hz)'], g['Threshold (dB HL)']))
    ).to_dict()
    result.update({'Status': anomaly_status, 'Version': '2'})
    json_str = json.dumps(result, indent=2)
    return json_str


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)
