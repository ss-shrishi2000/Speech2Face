import pandas as pd
from pytube import YouTube

def download_data():
    audio_data = pd.read_csv("avspeech_train.csv")
    for audio_row in audio_data.values[:50]:
        try:
            yt = YouTube("https://www.youtube.com/watch?v=" + audio_row[0], use_oauth=True, allow_oauth_cache=True)
            yt.streams.first().download('videos', audio_row[0] + ".mp4")
            print(">>>> Video downloaded is: ", audio_row[0] + ".mp4")
        except Exception as e:
            print("error while creating video for id ", audio_row[0])
            print(e)
        

if __name__ == "__main__":
    download_data()