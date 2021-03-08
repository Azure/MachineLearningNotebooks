"""
Test extract audio from video
"""
import os
import tempfile
import unittest.mock

import librosa
import pytest

from ..video import (
    extract_audio_from_video,
    extract_frames_from_video,
    get_video_length,
    make_ffmpeg_command,
)

TEST_WAV1 = "data/test_data/test1.wav"
TEST_VIDEO1 = "data/test_data/test_video1.mp4"
TEST_DIR_NO_VIDEO = "data/test_data/no_video"
TEST_DIR_VIDEO = "data/test_data/video"
SAMPLE_RATE = 22050


def test_extract_audio_from_video():
    """
    Test extract audio from video
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        test_wav_path = os.path.join(temp_dir, "test_extract_audio_from_video.wav")
        extract_audio_from_video(TEST_VIDEO1, test_wav_path)
        # Since we relay on ffmpeg to do all the conversion we assume that it is already well tested.
        # So the only thing we should validate is if audio file exists
        assert os.path.exists(test_wav_path)


@pytest.mark.parametrize(
    "sr, start_time, end_time, full_video_path, expected_audio_length",
    [
        (SAMPLE_RATE, "00:00:00", "00:00:00.500", TEST_VIDEO1, SAMPLE_RATE // 2),
        (SAMPLE_RATE, "00:00:00", "00:00:01", TEST_VIDEO1, SAMPLE_RATE),
        (SAMPLE_RATE, "00:00:00", "00:00:01.500", TEST_VIDEO1, 3 * SAMPLE_RATE // 2),
    ],
)
def test_extract_audio_from_video_start_and_end_times(
    sr: int,
    start_time: str,
    end_time: str,
    full_video_path: str,
    expected_audio_length: int,
):
    """
    Test extract audio from video

    Parameters
    ----------
    sr : int
        Parameter to test
    start_time : str
        Parameter to test
    end_time : str
        Paramter to test
    full_video_path : str
        Paramter for validation
    expected_audio_length : int
        Parameter for validation
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        test_wav_path = os.path.join(temp_dir, "test_extract_audio_from_video.wav")
        extract_audio_from_video(
            full_video_path,
            test_wav_path,
            sr=sr,
            start_time=start_time,
            end_time=end_time,
        )
        audio_clip, sample_rate = librosa.load(test_wav_path)

        assert sample_rate == sr
        assert len(audio_clip) == expected_audio_length


@unittest.mock.patch("os.system")
def test_extract_frames_from_video(os_system):
    """
    Test extract frames from video

    Parameters
    ----------
    os_system : str
        mocked os.system
    """
    # testing file extansion
    pytest.raises(
        Exception, extract_frames_from_video, TEST_DIR_NO_VIDEO, TEST_DIR_NO_VIDEO
    )
    # testing if ffmpeg call is made by os.system
    command = make_ffmpeg_command(
        video=os.path.join(TEST_DIR_VIDEO, "test_video.mp4"),
        output_dir=os.path.join(TEST_DIR_VIDEO, "test_video"),
        output_filename="test_video",
        fps=1,
    )
    extract_frames_from_video(
        input_path=TEST_DIR_VIDEO, output_path=TEST_DIR_VIDEO, fps=1
    )
    os_system.assert_called_once_with(command)


def test_make_ffmpeg_command():
    """
    Test make ffmepg command
    """
    excepted = (
        "ffmpeg -i test_video.mp4 -vf fps=1/5 provided_output_dir/test_video_%0d.jpeg"
    )
    assert excepted == make_ffmpeg_command(
        video="test_video.mp4",
        output_dir="provided_output_dir",
        output_filename="test_video",
        fps=5,
    )


@pytest.mark.parametrize(
    "filepath, expected_duration, expected_fps",
    [
        (TEST_VIDEO1, 2.0, 30),
    ],
)
def test_get_video_length(filepath: str, expected_duration: float, expected_fps: int):
    """
    Test get video length command

    Parameters
    ----------
    filepath : int
        Path to video file
    expected_duration : float
        Expected duration of the video in seconds
    expected_fps : int
        Expected frame rate
    """
    # Arrange

    # Act
    test_duration, _, test_fps = get_video_length(filepath)

    # Assert
    assert test_duration == expected_duration
    assert test_fps == expected_fps
