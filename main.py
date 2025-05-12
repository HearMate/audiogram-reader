import json
from audiogram_parser_1 import *
from flask import Flask, request
from flask_cors import CORS

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
    result = df.groupby('Ear').apply(
        lambda g: dict(zip(g['Frequency (Hz)'], g['Threshold (dB HL)']))
    ).to_dict()
    result.update({'Version': '1'})
    json_str = json.dumps(result, indent=2)
    print(json_str)
    return json_str


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)
