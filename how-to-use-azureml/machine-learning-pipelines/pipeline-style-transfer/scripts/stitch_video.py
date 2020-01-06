import argparse
import os
import subprocess

parser = argparse.ArgumentParser(description="Process input video")
parser.add_argument('--images_dir', required=True)
parser.add_argument('--input_audio', required=True)
parser.add_argument('--output_dir', required=True)

args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

subprocess.run("ffmpeg -framerate 30 -i {}/%05d_video.jpg -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p "
               "-y {}/video_without_audio.mp4"
               .format(args.images_dir, args.output_dir),
               shell=True, check=True)

subprocess.run("ffmpeg -i {}/video_without_audio.mp4 -i {}/video.aac -map 0:0 -map 1:0 -vcodec "
               "copy -acodec copy -y {}/video_with_audio.mp4"
               .format(args.output_dir, args.input_audio, args.output_dir),
               shell=True, check=True)
