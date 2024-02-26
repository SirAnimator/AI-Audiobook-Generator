import pip
import io
import os
import torch
from TTS.api import TTS
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
import time
import wave


class APIAudioConverter:

    def __init__(self, speech="en.mp3", language="en", speaker_wav_path=f"{os.getcwd()}/drive/MyDrive/Colab_Notebooks/voices/", file_output_path=f"{os.getcwd()}/drive/MyDrive/Colab_Notebooks/output/"):
        self.speech = speech
        self.language = language
        self.speaker_wav_path = speaker_wav_path+f"{self.speech}"
        self.file_output_path = file_output_path
        self.tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(
            "cuda" if torch.cuda.is_available() else "cpu")

    def convert_text_to_audio(self, input):
        splittet_output = self.split_sentence(input)
        audio_path_list = []
        for split_sentence in splittet_output:
            audio_path_list.append(self.convert_to_audio(split_sentence))
        merge_audio = self.merge_audio_files(audio_path_list)

    def merge_audio_files(self, input_wave_files):
        # Open the first wave file to get parameters
        with wave.open(input_wave_files[0], 'rb') as first_wave:
            params = first_wave.getparams()

        # Create the output wave file
        with wave.open(f"{self.file_output_path}merged.wav", 'wb') as output_wave:
            output_wave.setparams(params)

            # Loop through the input wave files and append their audio data to the output wave file
            for input_file in input_wave_files:
                with wave.open(input_file, 'rb') as input_wave:
                    frames = input_wave.readframes(input_wave.getnframes())
                    output_wave.writeframes(frames)
                os.remove(input_file)

    def get_timestamp(self):
        return time.strftime("%Y%m%d-%H%M%S")

    def split_sentence(self, input_text):
        """
        Split the input text into individual sentences.

        Args:
        input_text (str): The input text to be split into sentences.

        Returns:
        list: A list of individual sentences.
        """
        # Split the input text using period as the delimiter
        split_text = input_text.split(".")
        # Add the period back to each split sentence except for the last one
        split_text = [s + "." for s in split_text[:-1]]
        return split_text

    def convert_to_audio(self, input):
        file_path = self.file_output_path+f"{self.get_timestamp()}.wav"
        self.tts.tts_to_file(text=input, speaker_wav=self.speaker_wav_path,
                             language=self.language, file_path=file_path)
        return file_path


if __name__ == '__main__':

    with open(f"{os.getcwd()}/drive/MyDrive/Colab_Notebooks/text.txt", "r") as f:
        text = f.read()


    input = "One day, my students sat me down and ordered me to write this book. They wanted people to be able to use our work to make their lives better. It was something I’d wanted to do for a long time, but it became my number one priority."
    APIAudioConverter(speaker_wav_path="./voices/voices_Training_WAV/",
                      file_output_path="./output/").convert_text_to_audio(input)
