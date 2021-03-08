"""
Video preprocessing module
"""
import os
import subprocess
import time
from datetime import datetime
from typing import Optional

import cv2
from tqdm import tqdm

from .aml import get_logger
from .common import reuse
from .timestamp import format_timestamp_to_seconds

log = get_logger(__name__)


@reuse
def extract_audio_from_video(
    input_video_path: str,
    output_audio_path: str,
    sr: Optional[int] = None,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    use_gpu: bool = False,
    overwrite: bool = False,
):
    """Extracts audio track from video file and saves it as wav file.
    It will also change sampling rate based on `sr` param provided.
    For video->audio extraction ffmpeg library is used.

    If a start_time and end_time parameter are passed, the extracted audio will only
    be in the segment between those times

    Parameters
    ----------
    input_video_path : str
        Video file path.
    output_audio_path : str
        Output wav file path.
    sr : int, optional
        Target sampling rate.
        If `None`, then the default sampling rate of the input video will be used
    start_time : str, optional
        Start time to extract audio from video in HH:MM:SS.sss format
        If `None` then the whole audio will be extracted
    end_time : str, optional
        End time to extract audio from video in HH:MM:SS.sss format.
        If `None` then the whole audio will be extracted
    use_gpu : bool, optional
        Use ffmpeg enabled with GPU.
        By default False.
    overwrite : bool, optional
        Overwrite audio files at output_audio_path.
        By default False.
    """
    # Create dir if not exists
    dir_name = os.path.dirname(output_audio_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)

    target_sr_flag = f"-ar {sr}" if sr is not None else ""

    overwrite_flag = "-y" if overwrite else ""
    cuda_flag = "-hwaccel cuda" if use_gpu else ""

    if start_time is not None and end_time is not None:
        end_time = format_timestamp_to_seconds(end_time) - format_timestamp_to_seconds(
            start_time
        )

        # Timestamp flag will preceed the input -i see https://trac.ffmpeg.org/wiki/Seeking for details
        # pylint: disable=line-too-long
        command = f"ffmpeg {cuda_flag} -ss {start_time} -i {input_video_path} -t {end_time} -vn -acodec pcm_s16le {target_sr_flag} -ac 1 -y {output_audio_path}"
    else:
        # prepare command for ffmpeg
        command = f"ffmpeg {overwrite_flag} {cuda_flag} -i {input_video_path} {target_sr_flag} {output_audio_path}"

    log.info("Executing ffmpeg command: %s", command)
    subprocess.call(command, shell=True)


def make_ffmpeg_command(fps: int, video: str, output_dir: str, output_filename: str):
    """Generate a command for extracting images frame by frame from a video using
    FFmpeg. Uses provided extraction frame rate (fps).

    Parameters
    ----------
    fps : int
        the frame rate extration (frames per second).
    video: str
        full path to the video which want to extract frames
    output_dir : str
        full path to the folder where frames will be saved
    output_filename :str
        Base image name be saved.

    Returns
    -------
    srt
        a string that corresponds to a command for extracting images frame by frame
        from a video using FFmpeg
    """
    command = (
        f"ffmpeg -i {video} -vf fps=1/{fps} {output_dir}/{output_filename}_%0d.jpeg"
    )
    return command


def extract_frames_from_video(
    input_path: str, output_path: str, fps=1, video_extension=".mp4"
):
    """Extract frames (images) from videos in provided input path with defined
    extraction rate (frames per second). Extracted frames will be saved on a
    folder titled as video name created in provided output path. Frames will be
    saved as jpeg images tittled <video-name>_%0d.jpeg, where d means the number of
    seconds until that frame.

    NOTE: This method assumes FFmpeg is installed and will work with mp4 files only.

    Parameters
    ----------
    input_path : str
        full path to folder containing the video(s)
    output_path : str
        full path to a folder where frames will be saved
    fps : int, optional
        the frame rate extration. Defaults to 1 (one frame per second).
    video_extension : str, optional
        the video extension. Dafaults to ".mp4"

    Raises
    ------
    Exception
        Rise if the extension of the file does not correspond to provided video extension.
    """
    videos = [file for file in os.listdir(input_path) if file.endswith(video_extension)]
    if not videos:
        raise Exception(
            "The extension of video files should be {video_extension}.",
            video_extension=video_extension,
        )

    start_time = time.time()
    # extract frames for each mp4 video file in input folder
    for video_file in tqdm(videos):
        filename = os.path.splitext(video_file)[0]
        output_dir = os.path.join(output_path, filename)
        # create a folder titled as video where images will be saved
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        input_video = os.path.join(input_path, video_file)
        command = make_ffmpeg_command(
            fps=fps, video=input_video, output_dir=output_dir, output_filename=filename
        )
        # execute command for frame extraction
        os.system(command)

    exec_time = datetime.utcfromtimestamp(time.time() - start_time).strftime("%H:%M:%S")
    log.info(f"Execution time of extract_frames_from_video was {exec_time}")


def get_video_length(filepath: str):
    """
    Description
    Parameters
    ----------
    filepath : str
        Path to video file

    Returns
    -------
    duration : float
        Duration of the video in seconds
    frame_count : int
        Number of frames in the video
    fps : float
        Frames per second (not always a clean number)
    """
    video = cv2.VideoCapture(filepath)
    cv_version = int(cv2.__version__.split(".")[0])
    if cv_version > 2:
        fps = video.get(cv2.CAP_PROP_FPS)
    else:
        fps = video.get(cv2.CV_CAP_PROP_FPS)
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    try:
        duration = frame_count / fps
    except ZeroDivisionError:
        duration = 0
        print("ZeroDivisionError when calculating duration")
        print(f"duration = {frame_count}/{fps}")
    video.release()

    return duration, frame_count, fps
