import os
import subprocess
from glob import glob

gifs_dir = "/media/patricija/Pat/gif_data/gifs"
frames_dir = "/media/patricija/Pat/gif_data/frames"
os.makedirs(frames_dir, exist_ok=True)

gifs = glob(os.path.join(gifs_dir, "*.gif"))

for gif_path in gifs:
    gif_id = os.path.splitext(os.path.basename(gif_path))[0]
    out_dir = os.path.join(frames_dir, gif_id)
    os.makedirs(out_dir, exist_ok=True)
    
    # extract 16 evenly spaced frames
    cmd = [
        "ffmpeg", "-y", "-i", gif_path,
        "-vf", "fps=16/1",  # OR use "select='not(mod(n,4))'" for longer gifs
        os.path.join(out_dir, "frame_%02d.jpg")
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
