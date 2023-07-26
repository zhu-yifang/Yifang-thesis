import os
import pandas as pd
from pathlib import Path
import numpy as np
import librosa
from python_speech_features import mfcc
import sounddevice as sd
import tensorflow as tf
from typing import Union

# Defines the global variables here
# Path to the dataset
TIMIT = Path(os.environ["TIMIT"])
TRAIN_METADATA_PATH = TIMIT / "train_data.csv"
TEST_METADATA_PATH = TIMIT / "test_data.csv"
CORE_TEST_SET_SPEAKER_IDS = [
    "MDAB0",
    "MWBT0",
    "FELC0",
    "MTAS1",
    "MWEW0",
    "FPAS0",
    "MJMP0",
    "MLNT0",
    "FPKT0",
    "MLLL0",
    "MTLS0",
    "FJLM0",
    "MBPM0",
    "MKLT0",
    "FNLP0",
    "MCMJ0",
    "MJDH0",
    "FMGD0",
    "MGRT0",
    "MNJM0",
    "FDHC0",
    "MJLN0",
    "MPAM0",
    "FMLD0",
]
PHONEME_TO_INDEX = {
    'ix': 0,
    's': 1,
    'n': 2,
    'iy': 3,
    'tcl': 4,
    'r': 5,
    'kcl': 6,
    'l': 7,
    'ih': 8,
    'dcl': 9,
    'k': 10,
    't': 11,
    'ae': 12,
    'm': 13,
    'eh': 14,
    'z': 15,
    'ax': 16,
    'q': 17,
    'd': 18,
    'axr': 19,
    'w': 20,
    'aa': 21,
    'ao': 22,
    'dh': 23,
    'dx': 24,
    'pcl': 25,
    'p': 26,
    'ay': 27,
    'ah': 28,
    'ey': 29,
    'sh': 30,
    'gcl': 31,
    'f': 32,
    'b': 33,
    'ow': 34,
    'er': 35,
    'g': 36,
    'v': 37,
    'bcl': 38,
    'ux': 39,
    'y': 40,
    'ng': 41,
    'jh': 42,
    'hv': 43,
    'nx': 44,
    'hh': 45,
    'el': 46,
    'ch': 47,
    'th': 48,
    'aw': 49,
    'en': 50,
    'oy': 51,
    'uw': 52,
    'uh': 53,
    'ax-h': 54,
    'zh': 55,
    'em': 56,
    'eng': 57
}


def read_metadata(metadata_path: Path) -> pd.DataFrame:
    """
    Read the csv as a `pd.DataFrame`

    :param metadata_path: the value is either `TRAIN_METADATA_PATH` or `TEST_METADATA_PATH`
    :type metadata_path: Path
    :return: a DataFrame
    :rtype: pd.DataFrame
    """
    return pd.read_csv(metadata_path)


def get_core_test_set(df: pd.DataFrame) -> pd.DataFrame:
    """
    Get the core test set from the test set.

    :param df: a DataFrame from `read_metadata`
    :type df: pd.DataFrame
    :return: a DataFrame
    :rtype: pd.DataFrame
    """
    return df[df["speaker_id"].isin(CORE_TEST_SET_SPEAKER_IDS)]


def get_paths_no_ext(df: pd.DataFrame) -> pd.Series:
    """
    Get the paths to all files in the training set or test set without extensions.

    :param df: a DataFrame, either from `read_metadata` or `get_core_test_set`
    :type df: pd.DataFrame
    :return: a Series of file extensions
    :rtype: pd.Series
    """
    paths_no_ext = df["path_from_data_dir"].str.split(".").str[0]
    paths_no_ext.drop_duplicates(inplace=True)
    # Reset the index of `train_path_no_ext`
    paths_no_ext.reset_index(drop=True, inplace=True)
    return paths_no_ext


def read_wav_file(path: str) -> np.ndarray:
    wav_path = TIMIT / "data" / Path(path + ".WAV.wav")
    wav_array, _ = librosa.load(wav_path, sr=None)
    return wav_array


def read_phn_file(path: str) -> list[tuple[str, str, str]]:
    phn_path = TIMIT / "data" / Path(path + ".phn")
    with open(phn_path, "r") as f:
        return [line.split() for line in f]


def get_samples_from_file(
        path: str) -> list[dict[str, Union[int, str, np.ndarray]]]:
    wav_array = read_wav_file(path)
    phn_data = read_phn_file(path)

    samples = []
    for start, end, transcription in phn_data:
        start, end = int(start), int(end)
        sample = {
            "class": PHONEME_TO_INDEX[transcription],
            "phonetic_transcription": transcription,
            "wav_array": wav_array[start:end]
        }
        samples.append(sample)
    return samples


def get_samples(paths_no_ext: pd.Series) -> pd.DataFrame:
    all_samples = []
    for path in paths_no_ext:
        samples_from_file = get_samples_from_file(path)
        all_samples.extend(samples_from_file)
    return pd.DataFrame(all_samples)


# To make all the features with the same length, we need to use `librosa` to stretch or shrink the `wav_array` to the same length. Because the mean length of `wav_array` is 1279.5, we will stretch the `wav_array` to the length of 1024. And add a new column `wav_array_stretched` to `train_data_new` to store the stretched `wav_array`
def stretch_wav_array(wav_array: np.ndarray) -> np.ndarray:
    """
    Stretch the wav array to the length of 1024.

    :param wav_array: a wav array
    :type wav_array: np.ndarray
    :return: a stretched wav array
    :rtype: np.ndarray
    """
    # Set n_fft to be no larger than the length of the signal
    n_fft = min(len(wav_array), 512)
    return librosa.effects.time_stretch(wav_array,
                                        rate=(len(wav_array) / 1024),
                                        n_fft=n_fft)


def get_mfcc_vect(wav_array: np.ndarray) -> np.ndarray:
    """
    Get the MFCC vector from the wav array.

    :param wav_array: a wav array with length of 1024
    :type wav_array: np.ndarray
    :return: a MFCC vector with length 195
    :rtype: np.ndarray
    """
    assert len(wav_array) == 1024, "The length of the wav array must be 1024"
    mfcc_order0 = mfcc(wav_array, samplerate=16000).reshape(-1)
    mfcc_vect = np.concatenate(
        mfcc(wav_array, samplerate=16000).reshape(-1),
        librosa.feature.delta(mfcc_order0).reshape(-1),
        librosa.feature.delta(mfcc_order0, order=2).reshape(-1),
    )
    assert len(mfcc_vect) == 195, "The length of the MFCC vector must be 195"
    return mfcc_vect


# Calculate the MFCC vectors for all the samples in the training set and test set
def add_mfcc_vects(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add the MFCC vectors to the DataFrame.

    :param df: a DataFrame, either from `read_metadata` or `get_core_test_set`
    :type df: pd.DataFrame
    :return: a DataFrame
    :rtype: pd.DataFrame
    """
    df["mfcc_vect"] = df["wav_array"].apply(stretch_wav_array).apply(
        get_mfcc_vect)
    return df


def get_X_y(df: pd.DataFrame) -> (np.ndarray, np.ndarray):
    """
    Get the X and y from the DataFrame.

    :param df: a `DataFrame` contains the `mfcc_vect` column
    :type df: pd.DataFrame
    :return: a tuple of X and y
    :rtype: (np.ndarray, np.ndarray)
    """
    X = np.array(df["mfcc_vect"].tolist())
    y = np.array(df["class"].tolist())
    return X, y


def normalize_X(X_train: np.ndarray,
                X_test: np.ndarray) -> (np.ndarray, np.ndarray):
    """
    Normalize the X matrices with z-score normalization

    :param X_train: the X matrix of the training set
    :type X_train: np.ndarray
    :param X_test: the X matrix of the test set
    :type X_test: np.ndarray
    :return: a tuple of normalized X matrices
    :rtype: (np.ndarray, np.ndarray)
    """
    X_train_mean = np.mean(X_train, axis=0)
    X_train_std = np.std(X_train, axis=0)
    X_train_normalized = (X_train - X_train_mean) / X_train_std
    X_test_normalized = (X_test - X_train_mean) / X_train_std
    return X_train_normalized, X_test_normalized
