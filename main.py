from pathlib import Path
from recognizer.file import File
from recognizer.phone import Phone
import re
import os
import pickle
import random
from collections import Counter
import heapq
from scipy.io import wavfile
from sklearn.metrics import confusion_matrix

TIMIT = Path("/Users/zhuyifang/Downloads/archive")
#TIMIT = Path("/home/bart/work/reed-theses/zhu-thesis/timit")

IGNORED_PHONES = {"h#", "#h", "sil", "pau", "epi"}

GROUP_1 = {'axr', 'er'}
GROUP_2 = {'m', 'em'}
GROUP_3 = {'n', 'en', 'nx'}
GROUP_4 = {'ng', 'eng'}


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
def read_files(files: list[File]) -> list[File]:
    for file in files:
        filepath = os.path.join(file.path, file.name)
        # read .PHN
        with open(filepath + ".PHN") as f:
            file.phn = f.readlines()
        # read .wav
        file.samplerate, file.wav = wavfile.read(filepath + ".WAV.wav")
    return files


# read all the files in the training set and make them into Phone objects
def get_phones_from_TIMIT(TIMIT_path: Path, set_name: str) -> list[Phone]:
    set_path = TIMIT_path / f"data/{set_name}"
    set_files = get_all_matched_files(set_path)
    read_files(set_files)
    set_phones = []
    for file in set_files:
        set_phones += file.get_phones()
    for phone in set_phones:
        phone.get_mfcc_seq()
    print(f"set parse finished: {len(set_phones)} phones")
    return set_phones


# save the phones into a file
def save_phones_to_pkl(phones: list[Phone], filename: str):
    with open(filename, "wb") as f:
        pickle.dump(phones, f)


# read phones from a file
def read_phones_from_pkl(filename: str) -> list[Phone]:
    with open(filename, "rb") as f:
        phones = pickle.load(f)
    return phones


def get_phones() -> tuple[list[Phone], list[Phone]]:
    # if test_set_phones.pkl and train_set_phones.pkl are not created
    # run the following code to create them
    if not Path("test_set_phones.pkl").exists() or not Path(
            "train_set_phones.pkl").exists():

        # read all the files in the training set and make them into Phone objects
        train_set_phones = get_phones_from_TIMIT(TIMIT, "TRAIN")

        # save the train_set_phones to a file
        save_phones_to_pkl(train_set_phones, "train_set_phones.pkl")

        # read all the files in the testing set and make them into Phone objects
        test_set_phones = get_phones_from_TIMIT(TIMIT, "TEST")

        # save the test_set_phones to a file
        save_phones_to_pkl(test_set_phones, "test_set_phones.pkl")

    else:
        # read the train_set_phones from a file
        train_set_phones = read_phones_from_pkl("train_set_phones.pkl")

        # read the test_set_phones from a file
        test_set_phones = read_phones_from_pkl("test_set_phones.pkl")
    return train_set_phones, test_set_phones


def drop_ignored_phones(phones: list[Phone]) -> list[Phone]:
    return list(
        filter(lambda phone: phone.transcription not in IGNORED_PHONES,
               phones))


def group_phones(phones: list[Phone]) -> dict[str, set[Phone]]:
    res = {
        'ix': {},
        'iy': {},
        's': {},
        'r': {},
        'n/en/nx': {},
        'l': {},
        'tcl': {},
        'kcl': {},
        'ih': {},
        'dcl': {},
        'k': {},
        't': {},
        'm/em': {},
        'eh': {},
        'ae': {},
        'axr/er': {},
        'ax': {},
        'z': {},
        'd': {},
        'q': {},
        'w': {},
        'ao': {},
        'aa': {},
        'dh': {},
        'pcl': {},
        'p': {},
        'dx': {},
        'f': {},
        'b': {},
        'ah': {},
        'ay': {},
        'gcl': {},
        'ey': {},
        'sh': {},
        'ow': {},
        'bcl': {},
        'g': {},
        'v': {},
        'y': {},
        'ux': {},
        'ng/eng': {},
        'jh': {},
        'hv': {},
        'hh': {},
        'el': {},
        'th': {},
        'oy': {},
        'ch': {},
        'uh': {},
        'aw': {},
        'uw': {},
        'ax-h': {},
        'zh': {}
    }

    for phone in phones:
        # fold the 4 groups
        if phone.transcription in GROUP_1:
            phone.transcription = 'axr/er'
            res['axr/er'].add(phone)
        elif phone.transcription in GROUP_2:
            phone.transcription = 'm/em'
            res['m/em'].add(phone)
        elif phone.transcription in GROUP_3:
            phone.transcription = 'n/en/nx'
            res['n/en/nx'].add(phone)
        elif phone.transcription in GROUP_4:
            phone.transcription = 'ng/eng'
            res['ng/eng'].add(phone)
        else:
            res[phone.transcription].add(phone)
    return res


def predict_phone(train_set_phones, test_phone):

    # using KNN to find the nearest neighbor
    k = 10
    # using a heap to keep track of the samllest k element
    # the items in the heap are tuples like (negative distance to the test_set_phone, train_set_phone transcription)
    heap = []
    heapq.heapify(heap)
    for train_set_phone in train_set_phones:
        distance = test_phone.distance_to(train_set_phone)
        if len(heap) < k:
            heapq.heappush(heap, (-distance, train_set_phone.transcription))
        else:
            if -heap[0][0] > distance:
                heapq.heapreplace(heap,
                                  (-distance, train_set_phone.transcription))

    # using Counter to get the most common phone in the heap
    counter = Counter()
    for _ in range(k):
        _, transcription = heapq.heappop(heap)
        counter[transcription] += 1

    # predicted_phone is the most common phone in the heap
    predicted_phone = counter.most_common(1)[0][0]
    return predicted_phone


def test_accuracy(train_set_phones: list[Phone], test_set_phones: list[Phone]):
    # test accuracy
    correct_num = 0

    # iterate all the phones in the test set
    nphones = 100
    test_phones = random.sample(test_set_phones, nphones)

    predicted_phone = predict_phone(train_set_phones, test_phones)
    for test_phone in test_phones:
        # if the prediction is correct
        if predicted_phone == test_phone.transcription:
            correct_num += 1
            print("correct")

        # if the prediction and test phone are in the same group
        else:
            for i in range(len(GROUPS)):
                if predicted_phone in GROUPS[
                        i] and test_phone.transcription in GROUPS[i]:
                    correct_num += 1
                    print("correct")
                    break
        print(
            f'predicted phone is: {predicted_phone}, actual phone is: {test_phone.transcription}'
        )
    # print the accuracy
    print(f"Accuracy: {correct_num / nphones}")


if __name__ == "__main__":
    train_set_phones, test_set_phones = get_phones()
    train_set_phones = drop_ignored_phones(train_set_phones)
    test_set_phones = drop_ignored_phones(test_set_phones)
    test_phones = group_phones(test_set_phones)

    # # get the number of each phone in the training set and test set
    # train_set_counter = Counter()
    # for phone in train_set_phones:
    #     train_set_counter[phone.transcription] += 1
    # print(f"The stats of the training set: {train_set_counter}")
    # test_set_counter = Counter()
    # for phone in test_set_phones:
    #     test_set_counter[phone.transcription] += 1
    # print(f"The stats of the testing set: {test_set_counter}")

    # test accuracy
    test_accuracy(train_set_phones, test_set_phones)

    # confusion matrix test
    labels = [
        'ix', 'iy', 's', 'r', 'n/en/nx', 'l', 'tcl', 'kcl', 'ih', 'dcl', 'k',
        't', 'm/em', 'eh', 'ae', 'axr/er', 'ax', 'z', 'd', 'q', 'w', 'ao',
        'aa', 'dh', 'pcl', 'p', 'dx', 'f', 'b', 'ah', 'ay', 'gcl', 'ey', 'sh',
        'ow', 'bcl', 'g', 'v', 'y', 'ux', 'ng/eng', 'jh', 'hv', 'hh', 'el',
        'th', 'oy', 'ch', 'uh', 'aw', 'uw', 'ax-h', 'zh'
    ]