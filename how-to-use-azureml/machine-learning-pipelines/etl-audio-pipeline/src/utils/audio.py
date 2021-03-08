"""
Audio preprocessing module
"""
import os
import shutil
import subprocess
import tempfile
import uuid
from os.path import join
from typing import Optional

import librosa
import numpy as np
from pydub import AudioSegment
from scipy.io import wavfile

from .aml import get_logger
from .common import reuse

log = get_logger(__name__)


def convert_audio_segment_to_array(audio: AudioSegment):
    """Converts `pydub.AudioSegment` to `numpy.ndarray`.

    Parameters
    ----------
    audio : AudioSegment
        pydub.AudioSegment class object

    Returns
    -------
    numpy.ndarray
        Audio data as array of shape (n_samples,) for mono and (n_samples, channels) for stereo
    """
    # convert AudioSegment to numpy.ndarray
    data = np.asarray(audio.get_array_of_samples())
    # reshape from (num_samples*num_channels,) into (num_samples, num_channels)
    data = data.reshape((-1, audio.channels))
    # if num_channels==1 -> d.shape == (num_samples, 1)
    # np.squeeze will remove that last unnecessary dimension
    # final shape will be (num_samples,)
    data = np.squeeze(data)
    return data


def convert_array_to_audio_segment(data: np.ndarray, sample_rate: int):
    """Converts audio in `numpy.ndarray` to `pydub.AudioSegment` object.

    Parameters
    ----------
    data : np.ndarray
        Audio data as array of shape (n_samples,) for mono and (n_samples, channels) for stereo
    sample_rate : int
        Sampling rate.

    Returns
    -------
    AudioSegment
        pydub.AudioSegment object
    """
    # d = d.astype(np.float32)
    audio = AudioSegment(
        data.tobytes(),
        frame_rate=sample_rate,
        sample_width=data.dtype.itemsize,
        channels=1 if data.ndim == 1 else data.shape[1],
    )
    return audio


def read_wav_to_audio_segment(
    file_path: str, target_sr: int = None, enforce_mono: bool = True
):
    """Reads `wav` file as `pydub.AudioSegment`.
    If `target_sr` is provided - audio will be resampled.
    Converts audio to mono by default.

    Parameters
    ----------
    file_path : str
        Path to wav file
    target_sr : int, optional
        Resamples audio if provided, by default None
    enforce_mono : bool, optional
        Converts audio to mono, by default True

    Returns
    -------
    AudioSegmnet
        pydub.AudioSegmnet object
    """
    # read wav file from path provided
    audio = AudioSegment.from_wav(file_path)
    # change sampling rate if necessary
    if target_sr is not None:
        audio = resample_audio_segment(audio, target_sr)
    # convert stereo to mono
    if enforce_mono:
        audio = audio.set_channels(1)
    return audio


def read_wav_to_array(file_path: str, target_sr: int = None, enforce_mono: bool = True):
    """Reads `wav` file as `numpy.ndarray`.
    If `target_sr` is provided - audio will be resampled.
    Converts audio to mono by default.

    Parameters
    ----------
    file_path : str
        Path to wav file
    target_sr : int, optional
        Resamples audio if provided, by default None
    enforce_mono : bool, optional
        Converts audio to mono, by default True

    Returns
    -------
    numpy.ndarray
        Audio data as array of shape (n_samples,) for mono and (n_samples, channels) for stereo
    """
    audio = read_wav_to_audio_segment(file_path, target_sr, enforce_mono)
    data = convert_audio_segment_to_array(audio)
    # sr, d = wavfile.read(file_path)
    # if enforce_mono:
    #     d = np.mean(d, axis=-1)
    # if target_sr is not None:
    #     d = resample_array(d, sr, target_sr)
    return data


def resample_audio_segment(audio: AudioSegment, target_sr: int):
    """Changes AudioSegment frame_rate parameter based on target_sr arg.

    Parameters
    ----------
    audio : AudioSegment
        pydub.AudioSegment object
    target_sr : int
        Target sampling/frame rate

    Returns
    -------
    AudioSegment
        Resampled pydub.AudioSegment object
    """
    audio = audio.set_frame_rate(target_sr)
    return audio


def resample_array(data: np.ndarray, org_sr: int, target_sr: int):
    """Changes numpy.ndarray samples number based on org_sr and target_sr args.

    Parameters
    ----------
    data : numpy.ndarray
        Audio data as array of shape (n_samples,) for mono and (n_samples, channels) for stereo
    org_sr : int
        Base sampling/frame rate
    target_sr : int
        Target sampling/frame rate

    Returns
    -------
    numpy.ndarray
        Audio data as array of shape (n_samples,) for mono and (n_samples, channels) for stereo
    """
    # convert ndarray to AudioSegment
    audio = convert_array_to_audio_segment(data, org_sr)
    # set target frame rate
    audio = audio.set_frame_rate(target_sr)
    # convert back from AudioSegment to ndarray
    data = convert_audio_segment_to_array(audio)
    return data


def write_audio_segment_to_wav(audio: AudioSegment, wav_path: str):
    """Writes AudioSegment to wav file.

    Parameters
    ----------
    audio : AudioSegment
        Audio segment
    wav_path : str
        File path to same the wav file
    """
    dir_name = os.path.dirname(wav_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    audio.export(wav_path, "wav")


def write_array_to_wav(data: np.ndarray, wav_path: str, sample_rate: int):
    """Writes numpy.ndarray to wav file.

    Parameters
    ----------
    data : np.ndarray
        Data
    wav_path : str
        File path to same the wav file
    sample_rate : int
        Sample rate
    """
    dir_name = os.path.dirname(wav_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    wavfile.write(wav_path, sample_rate, data)


def crop_from_audio_segment(
    audio: AudioSegment, start: float, end: float, unit: str = "sec"
):
    """Extracts a single crop from AudioSegment based on start, end params.
    Start and end values should be provided in seconds (unit=='sec') or in milliseconds (unit=='ms').
    Unit arg should be one of: [`ms`, `sec`].
    Returns AudioSegment.

    Parameters
    ----------
    audio : AudioSegment
        Audio segment
    start : float
        Start value for the crop.
        Should be provided in seconds (unit=='sec') or in milliseconds (unit=='ms').
    end : float
        End value for the crop.
        Should be provided in seconds (unit=='sec') or in milliseconds (unit=='ms').
    unit : str, optional
        Should be one of: [`ms`, `sec`], by default 'sec'

    Returns
    -------
    AudioSegment
        Cropped out AudioSegment
    """
    assert unit in [
        "ms",
        "sec",
    ], "`unit` value needs to be one of ['ms', 'sec']"  # NOQA E501
    if unit == "sec":
        # AudioSegment cropping works in ms by default so we need to convert sec to ms
        start *= 1000.0
        end *= 1000.0

    audio_crop = audio[start:end]
    return audio_crop


def crop_from_array(
    data: np.ndarray, start: float, end: float, sample_rate: int, unit: str = "frames"
):
    """Extracts a single crop from AudioSegment based on start, end params.
    Start and end values should be provided in frame number
        (unit='frames'), seconds (unit=='sec') or in milliseconds (unit=='ms').
    Unit arg should be one of: [`frames`, `ms`, `sec`].
    Returns AudioSegment.

    Parameters
    ----------
    data : np.ndarray
        Data
    start : float
        Start value for the crop.
        Should be provided in frame number (unit='frames'), seconds (unit=='sec') or in milliseconds (unit=='ms').
    end : float
        End value for the crop.
        Should be provided in frame number (unit='frames'), seconds (unit=='sec') or in milliseconds (unit=='ms').
    sample_rate : int
        Sampling rate.
    unit : str, optional
        Should be one of: [`frames`, `ms`, `sec`], by default 'frames'

    Returns
    -------
    numpy.ndarray
        Cropped out audio data
    """
    assert unit in [
        "ms",
        "sec",
        "frames",
    ], "`unit` value needs to be one of ['ms', 'sec', 'frames']"  # NOQA E501
    if unit == "ms":
        # 1000 ms = sr (sampling rate)
        # 1 ms = sr / 1000
        start *= sample_rate / 1000
        end *= sample_rate / 1000
    elif unit == "sec":
        # 1 sec = sr (sampling rate)
        start *= sample_rate
        end *= sample_rate
    return data[int(start) : int(end)]


def mixup_audio_arrays(
    audio1_arr: np.ndarray,
    audio2_arr: np.ndarray,
    sample_rate: int,
    audio1_volume: int = None,
    position: int = 0,
) -> np.ndarray:
    """Mixes two audio tracks represented as numpy arrays into one

    Parameters
    ----------
    audio1_arr : np.ndarray
        First audio track to mix
    audio2_arr : np.ndarray
        Second audio track to mix
    sample_rate : int
        Sampling rate
    audio1_volume : int, optional
        Changes the first segment's volume by the specified amount during the
        duration of time that seg is overlaid on top of it. When negative,
        this has the effect of 'ducking' the audio under the overlay., by default None
    position : int, optional
        The position to start overlaying the second segment in to the first one, by default 0

    Returns
    -------
    np.ndarray
        Mixed audio track represented as numpy array
    """
    # convert numpy arrays to AudioSegments
    audio1_seg = convert_array_to_audio_segment(audio1_arr, sample_rate)
    audio2_seg = convert_array_to_audio_segment(audio2_arr, sample_rate)
    # Mixup audio tracks
    output_seg = audio1_seg.overlay(
        audio2_seg, position=position, gain_during_overlay=audio1_volume
    )
    # Convert AudioSegment back to numpy array
    output_arr = convert_audio_segment_to_array(output_seg)
    return output_arr


@reuse
def run_ffmpeg_dynaudnorm(
    input_filepath: str, output_filepath: str, overwrite: bool = True
) -> None:
    """Runs dynamic range compression using ffmpeg

    Parameters
    ----------
    input_filepath : str
        Filepath of the wav file to apply dynamic range compression
    output_filepath : str
        Name of output file after running dynamic range compression
    overwrite : bool
        True if this function should overwrite output_filepath if it already exists
    """
    with tempfile.TemporaryDirectory() as tempdir:
        if input_filepath == output_filepath:
            output_ffmpeg_filepath = join(tempdir, f"{str(uuid.uuid4())}.wav")
        else:
            output_ffmpeg_filepath = output_filepath

        overwrite_flag = "-y" if overwrite else ""
        command = f'ffmpeg {overwrite_flag} -i "{input_filepath}" -af dynaudnorm "{output_ffmpeg_filepath}"'

        # If this subprocess call fails then empty blobs (0 bytes) will be created and subsequent steps will fail
        subprocess.call(command, shell=True)

        if input_filepath == output_filepath:
            shutil.move(output_ffmpeg_filepath, output_filepath)


@reuse
def reformat_audio_for_custom_speech(
    input_filepath: str, output_filepath: str, overwrite: bool = True
):
    """Reformats audio for custom speech

    The guidelines for data format for custom speech can be found here:
    https://docs.microsoft.com/en-us/azure/cognitive-services/speech-service/how-to-custom-speech-test-and-train

    Currently we reformat to the following
    - Single audio channel (mono)
    - Sample rate of 16,000 Hz
    - PCM signed 16-bit little-endian

    Parameters
    ----------
    input_filepath : str
        Input filepath to the audio file to reformat
    output_filepath : str
        Output filepath that the reformatted audio file should be wrriten to
    overwrite : bool
        True if this function should overwrite output_filepath if it already exists

    Raises
    ------
    ValueError
        If input_filepath is equal to output_filepath this exception is raised as
        ffmpeg cannot write files in-place
    """
    if input_filepath == output_filepath:
        raise ValueError(
            "Parameters input_filepath and output_filepath must not be equal"
        )

    overwrite_flag = "-y" if overwrite else ""
    command = f"ffmpeg {overwrite_flag} -i {input_filepath} -acodec pcm_s16le -ac 1 -ar 16000 {output_filepath}"

    subprocess.call(command, shell=True)


@reuse
def segment_audio(
    input_audio_filepath: str,
    output_audio_filepath: str,
    sample_rate: int,
    duration: int,
    start_time: float,
    end_time: float,
    overwrite: bool = False,
    audio_clip: Optional[np.ndarray] = None,
):
    """Segment an audio clip to a length of duration given a start and end time in seconds.

    Parameters
    ----------
    input_audio_filepath : str
        Input filepath to the audio file to segment. If audio_clip is set this parameter will be ignored
    output_audio_filepath : str
        Output filepath to the audio file that is segmented
    sample_rate : int
        Sampling rate
    duration : int
        Duration of an audio segment. If the start_time and end_time are less than duration the
        segment will be zero padded to the right
    start_time : float
        Start time to extract the audio segment from the audio clip in seconds
    end_time : float
        End time to extract the audio segment from the audio clip in seconds
    overwrite : bool
        True if output_audio_filepath should be overwritten if it already exists
    audio_clip : np.ndarray, optional
        Audio clip to extract the audio segment from. If set, input_audio_filepath will be ignored
        and audio_clip will be used instead
    """
    # pylint: disable=unused-argument
    # overwrite is used by the decorator reuse.
    # It is kept in this function signature to reflect that it is a valid parameter to pass in.
    if audio_clip is None:
        audio_clip, _ = librosa.load(input_audio_filepath, sample_rate)

    start_time = int(start_time * sample_rate)
    end_time = int(end_time * sample_rate)
    audio_clip = audio_clip[start_time:end_time]

    # Zero Pad the audio until it matches the length duration
    if len(audio_clip) < duration * sample_rate:
        audio_clip = np.pad(
            audio_clip,
            (0, int(duration * sample_rate - len(audio_clip))),
            "constant",
        )

    wavfile.write(output_audio_filepath, sample_rate, audio_clip)


def get_audio_length(filepath: str) -> float:
    """Retrieve length of audio file in seconds

    Parameters
    ----------
    filepath : str
        Path to audio file to get length of

    Returns
    -------
    audio_length_in_sec : float
        Duration of the audio file in seconds
    """
    # Load file
    audio_signal, rate = librosa.load(filepath)

    # Get full length
    audio_length_in_sec = float(len(audio_signal) / rate)

    return audio_length_in_sec
