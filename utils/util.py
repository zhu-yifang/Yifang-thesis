import os
import pandas as pd
from pathlib import Path
import numpy as np
import librosa
from python_speech_features import mfcc
import sounddevice as sd

# Defines the global variables here
# Path to the dataset
TIMIT = Path(os.environ["TIMIT"])


