from python_speech_features import mfcc
from tslearn.metrics import dtw


class Phone():

    def __init__(self, samplerate, data, transcription) -> None:
        self.samplerate = samplerate
        self.data = data
        self.mfcc_seq = None
        self.transcription = transcription

    def __str__(self) -> str:
        return f"samplerate = {self.samplerate}, data = {self.data}, \
            transcription = {self.transcription}, mfcc_seq = {self.mfcc_seq}"

    def get_mfcc_seq(self):
        self.mfcc_seq = mfcc(self.data, self.samplerate)
        return self.mfcc_seq

    def distance_to(self, other: "Phone"):
        return dtw(self.mfcc_seq, other.mfcc_seq)

