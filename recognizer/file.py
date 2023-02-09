import numpy as np
from recognizer.phone import Phone

class File():

    def __init__(self, path, name) -> None:
        self.path = path
        self.name = name  # filename without extension
        self.wav = np.array([])
        self.samplerate = 16000
        self.phn = []

    def __str__(self) -> str:
        return f"path = {self.path}, name = {self.name}, wav = {self.wav}, \
            phn = {self.phn}"

    def get_phones(self):
        phones = []
        for line in self.phn:
            start, end, transcription = line.split()
            start, end = int(start), int(end)
            wav_data = self.wav[start:end]
            phone = Phone(self.samplerate, wav_data, transcription)
            phones.append(phone)
        return phones