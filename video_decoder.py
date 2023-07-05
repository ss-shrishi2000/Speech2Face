import moviepy.editor
from pathlib import Path

class VideoDecoder():

    def __init__(self):
        self.audio_path = "data/audio/"
        self.video_path = "data/videos/"
        self.images_from_video_path = "data/images_from_video/"

    def extract_audio(self, file_name):
        video = moviepy.editor.VideoFileClip(self.video_path + file_name)
        video.audio.write_audiofile(self.audio_path + Path(file_name).resolve().stem + ".wav")

    def extract_frames(self, file_name, no_of_frames):
        video = moviepy.editor.VideoFileClip(self.video_path + file_name)
        duration = int(video.duration)
        count = 1
        for frame_time in range(1, duration, duration // no_of_frames):
            video.save_frame(self.images_from_video_path +  Path(file_name).resolve().stem + str(count) + ".png", frame_time)
            count += 1
        print(">>> done extracting frame for fileName ", file_name)
        
