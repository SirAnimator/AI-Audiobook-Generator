import torch
from TTS.api import TTS

# Get device
device = "cuda" if torch.cuda.is_available() else "cpu"

# List available 🐸TTS models
print(TTS().list_models())

# Init TTS
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

# Run TTS
# ❗ Since this model is multi-lingual voice cloning model, we must set the target speaker_wav and language
# Text to speech list of amplitude values as output
# wav = tts.tts(text="Hallo Ich bin dein Virtueller Assistent und Helfe dir dabei deine Aufgaben zu erledigen!",
#               speaker_wav=r"D:\__ai\TextToSpeech\voices\voices_Training_WAV\2_01.wav", language="de")
# Text to speech to a file
tts.tts_to_file(text="Hello I am your Virtual Assistant and help you with your tasks", speaker_wav=r"D:\__ai\TextToSpeech\voices\voices_Training_WAV\2_01.wav",
                language="en", file_path="output.wav")
