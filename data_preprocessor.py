import os
from helpers.audio_encoder import AudioFeatureExtraction
from helpers.face_extraction_model import FaceExtraction
from helpers.video_decoder import VideoDecoder


class DataPreprocessor():

    def __init__(self):
        self.video_path = "data/videos/"
        self.images_from_video_path = "data/images_from_video/"
        self.audio_path = "data/audio/"
        self.videoDecoder = VideoDecoder()
        self.audioFeatureExtractor = AudioFeatureExtraction(30)
        self.dimension = (100, 100)

    def extract_audio(self):
        for file_name in os.listdir(self.video_path):
            self.video_decoder.extract_audio(file_name)
    
    def extract_frames_from_video(self, no_of_frames):
        for file_name in os.listdir(self.video_path):
            self.videoDecoder.extract_frames(file_name, no_of_frames)

    def extract_faces_from_frames(self):
        for file_name in os.listdir(self.images_from_video_path):
            faceExtraction = FaceExtraction(self.dimension)
            faceExtraction.extract_face_from_frames(file_name)

    def extract_features_from_audio(self):
        for file_name in os.listdir(self.audio_path):
            print(self.audioFeatureExtractor.feature_extraction(file_name))
            
    
if __name__ == "__main__":
    dataPreprocessor = DataPreprocessor()
    # dataPreprocessor.extract_audio()
    # dataPreprocessor.extract_frames_from_video(5)
    # dataPreprocessor.extract_faces_from_frames()
    # dataPreprocessor.extract_features_from_audio()

    