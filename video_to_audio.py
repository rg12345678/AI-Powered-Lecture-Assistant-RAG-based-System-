import os
import subprocess

files = os.listdir("videos")

for file in files:
    title = file[ :-4]
    subprocess.run(["ffmpeg", "-i", f"videos/{file}", f"audios/{title}.mp3"])   