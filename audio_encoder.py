from fileinput import filename
import librosa
import numpy as np

class AudioFeatureExtraction():

    def __init__(self, feature_count):
        self.feature_count = feature_count
        self.audio_path = "data/audio/"
        self.duration = 6

    def feature_extraction(self, filename):
        aud, sr = librosa.load(self.audio_path + filename, mono = True, duration = self.duration)
        return np.array((aud, sr, self.feature_count)[0])
    
    def feature_extraction_test(self, filename, filepath):
        aud, sr = librosa.load(filepath + filename, mono = True, duration = self.duration)
        return np.array((aud, sr, self.feature_count)[0])
