"""
test_audio.py
"""
import filecmp
import os
import tempfile

import numpy as np
import pytest
from _pytest.fixtures import FixtureFunctionMarker
from pydub import AudioSegment
from scipy.io import wavfile

from ..audio import (
    convert_array_to_audio_segment,
    convert_audio_segment_to_array,
    crop_from_array,
    crop_from_audio_segment,
    get_audio_length,
    mixup_audio_arrays,
    read_wav_to_array,
    read_wav_to_audio_segment,
    resample_array,
    resample_audio_segment,
    run_ffmpeg_dynaudnorm,
    write_array_to_wav,
    write_audio_segment_to_wav,
)

TEST_WAV1 = "data/test_data/test1.wav"
TEST_WAV1_SR = 24000
TEST_WAV1_LEN_SEC = 2

READ_TEST_DATA = [
    # (wav_path, clip_len_sec, enforce_mono, test_channels, target_sr, test_sr)
    (TEST_WAV1, TEST_WAV1_LEN_SEC, True, 1, None, TEST_WAV1_SR),
    (TEST_WAV1, TEST_WAV1_LEN_SEC, False, 2, None, TEST_WAV1_SR),
    (TEST_WAV1, TEST_WAV1_LEN_SEC, False, 2, 6000, 6000),
    (TEST_WAV1, TEST_WAV1_LEN_SEC, True, 1, 12000, 12000),
]

READ_WAV_TO_AUDIO_SEGMENT_TEST_DATA = [
    # (wav_path, enforce_mono, test_channels, target_sr, test_sr)
    (TEST_WAV1, True, 1, None, TEST_WAV1_SR),
    (TEST_WAV1, False, 2, None, TEST_WAV1_SR),
    (TEST_WAV1, False, 2, 6000, 6000),
    (TEST_WAV1, True, 1, 12000, 12000),
]


@pytest.fixture(name="wav1_audio_segment")
def test_wav1_audio_segment():
    """Retrieves AudioSegment data from file

    Returns
    -------
    AudioSegment
        Audio sample
    """
    return AudioSegment.from_wav(TEST_WAV1)


@pytest.fixture(name="wav1_array")
def test_wav1_array():
    """Retrieves wavfile data from file

    Returns
    -------
    Any
        Return wavfile data
    """
    return wavfile.read(TEST_WAV1)[1]


@pytest.mark.parametrize(
    "wav_path, enforce_mono, test_channels, target_sr, test_sr",
    READ_WAV_TO_AUDIO_SEGMENT_TEST_DATA,
)
def test_read_wav_to_audio_segment(
    wav_path, enforce_mono, test_channels, target_sr, test_sr
):
    """[summary]

    Parameters
    ----------
    wav_path : str
        [description]
    enforce_mono : bool
        [description]
    test_channels : int
        [description]
    target_sr : Union[int, None]
        [description]
    test_sr : int
        [description]
    """
    audio_segment = read_wav_to_audio_segment(
        wav_path, target_sr=target_sr, enforce_mono=enforce_mono
    )
    assert isinstance(audio_segment, AudioSegment)
    assert audio_segment.channels == test_channels
    assert audio_segment.frame_rate == test_sr


@pytest.mark.parametrize(
    "wav_path, clip_len_sec, enforce_mono, test_channels, target_sr, test_sr",
    READ_TEST_DATA,
)
def test_read_wav_to_array(
    wav_path, clip_len_sec, enforce_mono, test_channels, target_sr, test_sr
):
    """[summary]

    Parameters
    ----------
    wav_path : str
        [description]
    clip_len_sec : int
        [description]
    enforce_mono : bool
        [description]
    test_channels : int
        [description]
    target_sr : Union[int, None]
        [description]
    test_sr : int
        [description]
    """
    data = read_wav_to_array(wav_path, target_sr=target_sr, enforce_mono=enforce_mono)
    assert isinstance(data, np.ndarray)
    assert data.ndim == test_channels
    assert data.shape[0] / clip_len_sec == test_sr


def test_audio_segment_to_array(
    wav1_audio_segment: FixtureFunctionMarker, wav1_array: FixtureFunctionMarker
):
    """[summary]

    Parameters
    ----------
    wav1_audio_segment : FixtureFunctionMarker
        [description]
    wav1_array : FixtureFunctionMarker
        [description]
    """
    data = convert_audio_segment_to_array(wav1_audio_segment)
    assert np.array_equal(data, wav1_array)


def test_array_to_audio_segment(
    wav1_audio_segment: FixtureFunctionMarker, wav1_array: FixtureFunctionMarker
):
    """[summary]

    Parameters
    ----------
    wav1_audio_segment : FixtureFunctionMarker
        [description]
    wav1_array : FixtureFunctionMarker
        [description]
    """
    audio = convert_array_to_audio_segment(wav1_array, TEST_WAV1_SR)
    assert wav1_audio_segment.channels == audio.channels
    assert wav1_audio_segment.frame_rate == audio.frame_rate
    assert wav1_audio_segment.frame_width == audio.frame_width
    assert wav1_audio_segment.get_array_of_samples() == audio.get_array_of_samples()


def test_resample_audio_segment(wav1_audio_segment: FixtureFunctionMarker):
    """[summary]

    Parameters
    ----------
    wav1_audio_segment : FixtureFunctionMarker
        [description]
    """
    audio = resample_audio_segment(wav1_audio_segment, int(TEST_WAV1_SR / 2))
    assert audio.frame_rate == TEST_WAV1_SR / 2


def test_resample_array(wav1_array: FixtureFunctionMarker):
    """[summary]

    Parameters
    ----------
    wav1_array : FixtureFunctionMarker
        [description]
    """
    data = resample_array(
        wav1_array, org_sr=TEST_WAV1_SR, target_sr=int(TEST_WAV1_SR / 2)
    )
    assert data.shape[0] == wav1_array.shape[0] / 2


def test_write_audio_segment_to_wav(wav1_audio_segment: FixtureFunctionMarker):
    """[summary]

    Parameters
    ----------
    wav1_audio_segment : FixtureFunctionMarker
        [description]
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        test_wav_path = os.path.join(temp_dir, "test_write_audio_segment_to_wav.wav")
        write_audio_segment_to_wav(wav1_audio_segment, wav_path=test_wav_path)
        assert os.path.exists(test_wav_path)
        assert filecmp.cmp(TEST_WAV1, test_wav_path, shallow=False)


def test_write_array_to_wav(wav1_array: FixtureFunctionMarker):
    """[summary]

    Parameters
    ----------
    wav1_array : FixtureFunctionMarker
        [description]
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        test_wav_path = os.path.join(temp_dir, "test_write_array_to_wav.wav")
        write_array_to_wav(wav1_array, wav_path=test_wav_path, sample_rate=TEST_WAV1_SR)
        assert os.path.exists(test_wav_path)
        assert filecmp.cmp(TEST_WAV1, test_wav_path, shallow=False)


def test_crop_from_audio_segment(wav1_audio_segment: FixtureFunctionMarker):
    """[summary]

    Parameters
    ----------
    wav1_audio_segment : FixtureFunctionMarker
        [description]
    """
    crop_audio = crop_from_audio_segment(wav1_audio_segment, start=1, end=2, unit="sec")
    assert len(crop_audio.get_array_of_samples()) == TEST_WAV1_SR * crop_audio.channels

    crop_audio = crop_from_audio_segment(
        wav1_audio_segment, start=1000, end=2000, unit="ms"
    )
    assert len(crop_audio.get_array_of_samples()) == TEST_WAV1_SR * crop_audio.channels


def test_crop_from_array(wav1_array: FixtureFunctionMarker):
    """[summary]

    Parameters
    ----------
    wav1_array : FixtureFunctionMarker
        [description]
    """
    crop_arr = crop_from_array(
        wav1_array, start=1, end=2, sample_rate=TEST_WAV1_SR, unit="sec"
    )
    assert np.array_equal(crop_arr, wav1_array[TEST_WAV1_SR:])

    crop_arr = crop_from_array(
        wav1_array,
        start=TEST_WAV1_SR,
        end=2 * TEST_WAV1_SR,
        sample_rate=TEST_WAV1_SR,
        unit="frames",
    )
    assert np.array_equal(crop_arr, wav1_array[TEST_WAV1_SR:])

    crop_arr = crop_from_array(
        wav1_array, start=1000, end=2000, sample_rate=TEST_WAV1_SR, unit="ms"
    )
    assert np.array_equal(crop_arr, wav1_array[TEST_WAV1_SR:])


def test_mixup_audio_arrays(wav1_array: FixtureFunctionMarker):
    """Tests `mixup_audio_arrays_function`

    Parameters
    ----------
    wav1_array : FixtureFunctionMarker
        First array to mixup
    """
    result = mixup_audio_arrays(wav1_array, wav1_array, TEST_WAV1_SR, 10)
    assert isinstance(result, np.ndarray)
    assert len(result) == len(wav1_array)
    assert not np.array_equal(result, wav1_array)


def test_run_ffmpeg_dynaudnorm():
    """Tests `run_ffmpeg_dynaudnorm`"""

    with tempfile.TemporaryDirectory() as temp_dir:
        test_wav_path = os.path.join(temp_dir, "compressed_audio.wav")
        run_ffmpeg_dynaudnorm(TEST_WAV1, test_wav_path)
        assert os.path.exists(test_wav_path)
        assert not filecmp.cmp(TEST_WAV1, test_wav_path, shallow=False)


@pytest.mark.parametrize(
    "wav_path, expected_duration",
    [
        (TEST_WAV1, TEST_WAV1_LEN_SEC),
    ],
)
def test_get_audio_length(wav_path: str, expected_duration: float):
    """Tests `get_audio_length`

    Parameters
    ----------
    wav_path : str
        Path to audio file
    expected_duration : float
        Expected duration of the audio in seconds
    """
    # Arrange

    # Act
    test_length = get_audio_length(wav_path)

    # Assert
    assert test_length == expected_duration
