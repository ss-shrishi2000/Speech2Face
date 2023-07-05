from helpers.audio_encoder import AudioFeatureExtraction
from helpers.video_decoder import VideoDecoder

if __name__ == "__main__":
    videoDecoder = VideoDecoder()
    videoDecoder.extract_audio("boss.mov")
    audioFeatureExtraction = AudioFeatureExtraction(50)
    print(audioFeatureExtraction.feature_extraction("data/audio/boss.wav").shape)

