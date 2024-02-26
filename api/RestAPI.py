import io
import os
import torch
# import torch.backends.dir
from TTS.api import TTS
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
import time


# initial AI
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
print(TTS().list_models())
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes in the Flask app

{'output_name': "1",
 'input': "2",
 'language': "3",
 "voice": "4", }

# get Time Stamp as a string


def get_timestamp():
    return time.strftime("%Y%m%d-%H%M%S")


@app.route('/api', methods=['POST'])
def api():
    data = request.json
    print(data)
    wav = tts.tts_to_file(text=request.json['input'], speaker_wav=r"D:\__ai\AI-Audiobook-Generator\api\voices\voices_Training_WAV\en2.wav",
                          language="en", file_path=f"./output/{get_timestamp()}.wav")

    # Read the file as binary data
    with open(wav, "rb") as file:
        wav_binary = file.read()

    # Convert the binary data to an in-memory file-like object
    wav_bytes_io = io.BytesIO(wav_binary)

    # delete existing file
    os.remove(wav)

    # Send the binary data as a file response
    return send_file(wav_bytes_io, mimetype="audio/wav", as_attachment=True, download_name="output.wav")


# @app.route('/api', methods=['OPTIONS'])
# def handle_options():
#     # Add the necessary CORS headers
#     response = jsonify({'status': 'OK'})
#     response.headers.add('Access-Control-Allow-Origin',
#                          'http://127.0.0.1:5500')
#     response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
#     response.headers.add('Access-Control-Allow-Methods', 'POST')
#     return response


if __name__ == '__main__':
    app.run()
# def create_app():
#     return app
