import os
import re
import numpy as np
from scipy.io import wavfile


class Phone():

    def __init__(self, data, transcription) -> None:
        self.data = None
        self.mfcc_seq = None
        self.transcription = None

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


# root should be given as the absolute path
# return files
def get_all_matched_files(root: str):
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


# get the paths of all .wav and .PHN files in the training set
train_set_path = "/Users/zhuyifang/Downloads/archive/data/TRAIN"
wav_re = re.compile(r".+WAV\.wav")
files = get_all_matched_files(train_set_path)
read_files(files)
for line in files[0].phn:
    print(line.split())