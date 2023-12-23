import torch
from TTS.api import TTS
from flask import Flask, jsonify, request

# initial AI
device = "cuda" if torch.cuda.is_available() else "cpu"
print(TTS().list_models())
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

app = Flask(__name__)

{'output_name': "1",
 'text': "2",
 'language': "3",
 "speaker_wav": "4", }


@app.route('/api', methods=['POST'])
def api():
    data = request.json
    print(data)
    tts.tts_to_file(text=request.json['text'], speaker_wav=r"D:\__ai\TextToSpeech\voices\voices_Training_WAV\2_01.wav",
                    language="en", file_path=f"output/{request.json['output_name']}.wav")
    return jsonify(data)


if __name__ == '__main__':
    app.run(debug=True)
