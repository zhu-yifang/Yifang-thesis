import os
import random
import re
import numpy as np
import sys
from pathlib import Path
from python_speech_features import mfcc
from tslearn.metrics import dtw
from scipy.io import wavfile
from collections import Counter
import heapq

TIMIT = Path("/Users/zhuyifang/Downloads/archive")
#TIMIT = Path("/home/bart/work/reed-theses/zhu-thesis/timit")

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
    # read all the files in the training set and make them into Phone objects
    train_set_path = TIMIT / "data/TRAIN"
    wav_re = re.compile(r".+WAV\.wav")
    train_set_files = get_all_matched_files(train_set_path)
    read_files(train_set_files)
    train_set_phones = []
    for file in train_set_files:
        train_set_phones += file.get_phones()
    for phone in train_set_phones:
        phone.get_mfcc_seq()

    print(f"train set parse finished: {len(train_set_phones)} phones")

    # read all the files in the testing set and make them into Phone objects
    test_set_path = TIMIT / "data/TEST"
    test_set_files = get_all_matched_files(test_set_path)
    read_files(test_set_files)
    test_set_phones = []
    for file in test_set_files:
        test_set_phones += file.get_phones()

    print(f"test set parse finished: {len(test_set_phones)} phones")

    correct_num = 0

    # iterate all the phones in the test set
    nphones = int(sys.argv[1])
    test_phones = random.sample(test_set_phones, nphones)
    test_set_phones in test_set_phones
    for test_set_phone in test_phones:
        test_set_phone.get_mfcc_seq()
        # using KNN to find the nearest neighbor
        k = 10
        # using a heap to keep track of the samllest k element
        # the items in the heap are tuples like (negative distance to the test_set_phone, train_set_phone transcription)
        heap = []
        heapq.heapify(heap)
        for train_set_phone in train_set_phones:
            distance = test_set_phone.distance_to(train_set_phone)
            if len(heap) < k:
                heapq.heappush(heap,
                               (-distance, train_set_phone.transcription))
            else:
                if -heap[0][0] > distance:
                    heapq.heapreplace(
                        heap, (-distance, train_set_phone.transcription))

        # using Counter to get the most common phone in the heap
        counter = Counter()
        for i in range(k):
            _, transcription = heapq.heappop(heap)
            counter[transcription] += 1

        # predicted_phone is the most common phone in the heap
        predicted_phone = counter.most_common(1)[0][0]

        # if the prediction is correct
        if predicted_phone == test_set_phone.transcription:
            correct_num += 1
            print("correct")
        # if the prediction is wrong
        else:
            print(
                f'predicted phone is: {predicted_phone}, actual phone is: {phone.transcription}'
            )
    # print the accuracy
    print(f"Accuracy: {correct_num / nphones}")
