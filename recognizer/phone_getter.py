import os
import re
import scipy.io


# root should be given as the absolute path
def get_all_matched_paths(root: str, regex: re.Pattern):
    wav_filenames = []
    for dirpath, _, filenames in os.walk(root):
        for filename in filenames:
            print(filename)
            if regex.match(filename):
                wav_filenames.append(os.path.join(dirpath, filename))
    return wav_filenames


# get the paths of all wav files in the train set
cwd = "/Users/zhuyifang/Downloads/archive/data/TRAIN"
wav_re = re.compile(r".+WAV\.wav")
print(get_all_matched_paths(cwd, wav_re))

# read all the wav files