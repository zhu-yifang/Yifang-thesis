import os
import re
import numpy as np
from python_speech_features import mfcc
from tslearn.metrics import dtw
from scipy.io import wavfile
from collections import Counter

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


# root should be given as the absolute path
# return files
def get_all_matched_files(root: str) -> list[File]:
    phn_re = re.compile(r".+\.PHN")
    matched_files = []
    for dirpath, _, filenames in os.walk(root):
        for filename in filenames:
            if phn_re.match(filename):
                filename = filename[:-4]
                file = File(dirpath, filename)
                matched_files.append(file)
    return matched_files


# read .wav and .PHN
def read_files(files):
    for file in files:
        filepath = os.path.join(file.path, file.name)
        # read .PHN
        with open(filepath + ".PHN") as f:
            file.phn = f.readlines()
        # read .wav
        file.samplerate, file.wav = wavfile.read(filepath + ".WAV.wav")
    return files


if __name__ == "__main__":
    # get the paths of all .wav and .PHN files in the training set
    train_set_path = "/Users/zhuyifang/Downloads/archive/data/TRAIN"
    wav_re = re.compile(r".+WAV\.wav")
    train_set_files = get_all_matched_files(train_set_path)
    read_files(train_set_files)
    train_set_phones = []
    for file in train_set_files:
        train_set_phones += file.get_phones()
    for phone in train_set_phones:
        phone.get_mfcc_seq()
    
    # test
    test_set_path = "/Users/zhuyifang/Downloads/archive/data/TEST"
    test_set_files = get_all_matched_files(test_set_path)
    read_files(test_set_files)
    test_set_phones = []
    for file in test_set_files:
        test_set_phones += file.get_phones()
    correct_num = 0
    print(len(test_set_phones))
    for phone in test_set_phones:
        phone.get_mfcc_seq()
        # using KNN to find the nearest neighbor, k = 5
        distances = []
        for train_phone in train_set_phones:
            distances.append((phone.distance_to(train_phone), train_phone.transcription))
        distances.sort(key=lambda x: x[0])
        counter = Counter()
        for distance, transcription in distances:
            counter[transcription] += 1
            if counter[transcription] == 5:
                #print(f"Predicted transcription: {transcription}, actual transcription: {phone.transcription}")
                if transcription == phone.transcription:
                    correct_num += 1
                break
    print(f"Accuracy: {correct_num / len(test_set_phones)}")

