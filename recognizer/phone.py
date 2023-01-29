import os
import re
import numpy as np
from scipy.io import wavfile


class Phone():

    def __init__(self, data, transcription) -> None:
        self.data = data
        self.mfcc_seq = None
        self.transcription = transcription

    def __str__(self) -> str:
        return f"{self.transcription}"

    def get_mfcc_seq(self):
        pass


class File():

    def __init__(self, path, name) -> None:
        self.path = path
        self.name = name  # filename without extension
        self.wav = np.array([])
        self.phn = []

    def __str__(self) -> str:
        return f"path = {self.path}, name = {self.name}, wav = {self.wav}, phn = {self.phn}"

    def get_phones(self):
        phones = []
        for line in self.phn:
            start, end, transcription = line.split()
            start, end = int(start), int(end)
            wav_data = self.wav[start:end]
            phone = Phone(wav_data, transcription)
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
        _, data = wavfile.read(filepath + ".WAV.wav")
        file.wav = data
    return files


if __name__ == "__main__":
    # get the paths of all .wav and .PHN files in the training set
    train_set_path = "/Users/zhuyifang/Downloads/archive/data/TRAIN"
    wav_re = re.compile(r".+WAV\.wav")
    files = get_all_matched_files(train_set_path)
    read_files(files)
    phones = []
    for file in files:
        phones += file.get_phones()
    print(phones[0].data)