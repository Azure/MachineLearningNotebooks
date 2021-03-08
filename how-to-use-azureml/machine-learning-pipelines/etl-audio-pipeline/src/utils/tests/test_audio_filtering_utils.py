"""
audio_filtering_utils.py tests
"""
import unittest.mock

import librosa
import numpy as np
import pytest
from scipy.fftpack import fft

from ..audio_filtering_utils import (
    audio_updating,
    butter_filter,
    clean_audio_with_filt_denoise,
    compute_snr,
    denoise_whole_audio,
    filter_whole_audio,
    find_noise,
    save_wav,
    set_bandpass_filter_freq,
)

t = np.linspace(0, 4.0, 4000)  # 1 second
noise = np.random.normal(0, 0.01, 2000)
sig = 2 * np.sin(2 * np.pi * 5 * t[0:2000])
NOISE_SIG = np.asarray([noise, sig]).reshape(-1)
RATE = 1000  # Fake sampling rate lower than default 44.1 kHz


def fft_spect(signal: np.ndarray):
    """
    FFT transformation

    Parameters
    -------
    signal : np.ndarray
        [signal to transform]
    Returns
    -------
    singal_fft_magn
        magnitude of FFT signal in dB
    """
    n_fft = 1024
    signal_fft = fft(signal, n_fft)
    signal_fft = signal_fft[: n_fft // 2 + 1]
    signal_fft_magn, _ = librosa.magphase(signal_fft)
    return signal_fft_magn


@unittest.mock.patch("scipy.io.wavfile.write")
def test_save_wav(saved_wav):
    """
    test save_wav function

    Parameters
    -------
    saved_wav : np.ndarray
        [mocked objected]

    """
    # Arrange
    sample_rate = 22050
    wav = np.random.rand(2 * sample_rate)
    # unittest.mock.patch.object(os, 'path', return_value=True)

    # Act
    save_wav(wav, path="./data", output_name="test", sample_rate=sample_rate)

    # Assert
    assert saved_wav.called_once


@pytest.mark.parametrize(
    "f_c, filt_type, expected_signal",
    [
        (125, "low", np.sin(2 * np.pi * 5 * t)),
        (125, "high", np.sin(2 * np.pi * 250 * t)),
    ],
)
def test_butter_filter(f_c, filt_type, expected_signal):
    """
    test butter_filter function

    Parameters
    -------
    f_c : int
        [fc]
    filt_type : str
        [filter type]
    expected_signal : np.ndarray
        [expected output]

    """
    # Arrange
    signal = np.sin(2 * np.pi * 5 * t) + np.sin(2 * np.pi * 250 * t)
    tolerance = 10  # tolerance after filtering

    # Act
    output = butter_filter(signal, f_c, filt_type)
    # after filtering just one peak frequency should be presented in the signal, "very closed" to the expected output
    output_m = fft_spect(output)
    expected_signal_m = fft_spect(expected_signal)

    # Assert
    assert np.abs(np.argmax(output_m) - np.argmax(expected_signal_m)) < tolerance


@pytest.mark.parametrize(
    "sign1, sign2, expected_snr",
    [
        (np.array([2, 4, 6, 8]), np.array([1, 2, 3, 4]), 4.77),
        (np.array([2, 4, 6, 8]), np.array([1, 1, 1, 1]), 30),
    ],
)
def test_compute_srn(sign1, sign2, expected_snr):
    """
    test compute_snr function

    Parameters
    -------
    sign1 : np.ndarray
        [first signal]
    sign2 : np.ndarray
        [second signal]
    expected_snr : float
        [SNR]

    """
    # Act
    snr = compute_snr(sign1, sign2)

    assert np.round(snr, 2) == expected_snr


@pytest.mark.parametrize(
    "signal, expected_signal, clip_flag",
    [(NOISE_SIG, NOISE_SIG[:1000], True), (sig, np.zeros(1000), False)],
)
def test_find_noise(signal, expected_signal, clip_flag):
    """
    test find_noise function

    Parameters
    -------
    signal : np.ndarray
        [input signal]
    expected_signal : np.ndarray
        [expected output]
    clip_flag : book
        [flag for found/not found noise clip]

    """
    # Arrange
    noise_len = 1

    # Act
    noise_clip, found_clip = find_noise(signal, sample_rate=1000, noise_len=noise_len)

    # Assert
    assert found_clip == clip_flag
    np.testing.assert_array_almost_equal(noise_clip, expected_signal, decimal=1)


@pytest.mark.parametrize(
    "audio, filtered_audio, snr_in, expected_signal, clip_flag, snr_out",
    [
        (sig, NOISE_SIG, 16.0, NOISE_SIG, False, 40.0),  # snr_out > snr_in
        (sig, NOISE_SIG, 42.0, sig, True, 42.0),  # snr_out < snr_in
        (sig, np.zeros(1000), 30, sig, True, 30),
    ],
)  # cannot find noise_clip
def test_audio_updating(
    audio, filtered_audio, snr_in, expected_signal, clip_flag, snr_out
):
    """
    test audio_updating function

    Parameters
    -------
    audio : np.ndarray
        [original signal]
    filtered_audio : np.ndarray
        [filtered audio]
    snr_in : float
        [SNR in]
    expected_signal : np.ndarray
        [expected output]
    clip_flag : book
        [flag for break or not the loop]
    snr_out : float
        [SNR out]

    """
    # Arrange
    sample_rate = 1000
    min_improve = 0.5

    # Act
    signal, break_loop, snr = audio_updating(
        filtered_audio, sample_rate, audio, snr_in, min_improve
    )

    # Assert
    assert break_loop == clip_flag
    assert np.round(snr) == snr_out
    np.testing.assert_array_almost_equal(signal, expected_signal, decimal=1)


@pytest.mark.parametrize(
    "input_sig, expected_fc_low, expected_fc_high",
    [("clapperboard", 50, 5), ("voice", 450, 80)],
)
def test_set_bandpass_filter_freq(input_sig, expected_fc_low, expected_fc_high):
    """
    test set_bandpass_filter_freq

    Parameters
    -------
    input_sig : str
        [researched signal]
    expected_fc_low : int
        [expected fc_low]
    expected_fc_high : int
        [expected fc_high]

    """
    # Act
    fc_low, fc_high = set_bandpass_filter_freq(input_sig)

    # Assert
    assert fc_low == expected_fc_low
    assert fc_high == expected_fc_high


@pytest.mark.parametrize("input_sig, rate", [(NOISE_SIG, RATE)])
def test_filter_whole_audio(input_sig, rate):
    """
    test filter_whole_audio

    Parameters
    -------
    input_sig : np.ndarray
        [The audio signal]
    rate : int
        [Sampling rate in Hz]
    """
    # Arrange
    researched_signal = "voice"
    n = 4

    # Act
    sig_filt, rate_filt = filter_whole_audio(input_sig, rate, researched_signal, n)

    # Assert
    assert rate == rate_filt
    assert len(input_sig) == len(sig_filt)
    assert (type(sig_filt[0]) in [float, np.float64]) is True  # Check value type
    assert (True in np.isnan(sig_filt)) is False  # No NaN values
    assert (len(np.unique(sig_filt)) > 2) is True  # Values are not all zeros


@pytest.mark.parametrize("input_sig, rate", [(NOISE_SIG, RATE)])
def test_denoise_whole_audio(input_sig, rate):
    """
    test denoise_whole_audio

    Parameters
    -------
    input_sig : np.ndarray
        [The audio signal]
    rate : int
        [Sampling rate in Hz]
    """
    # Act
    sig_filt, rate_filt = denoise_whole_audio(input_sig, rate)

    # Assert
    assert rate == rate_filt
    assert len(input_sig) == len(sig_filt)
    assert (
        type(sig_filt[0]) in [float, np.float32, np.float64]
    ) is True  # Check value type
    assert (True in np.isnan(sig_filt)) is False  # No NaN values
    assert (len(np.unique(sig_filt)) > 2) is True  # Values are not all zeros


@pytest.mark.parametrize("input_sig, rate", [(NOISE_SIG, RATE)])
def test_clean_audio_with_filt_denoise(input_sig, rate):
    """
    test clean_audio_with_filt_denoise

    Parameters
    -------
    input_sig : np.ndarray
        [The audio signal]
    rate : int
        [Sampling rate in Hz]
    """
    # Arrange
    researched_signal = "voice"
    discard_snr_11 = False
    iter_max = 3
    min_improve = 0.1

    # Act
    sig_filt, rate_filt = clean_audio_with_filt_denoise(
        input_sig, rate, researched_signal, discard_snr_11, iter_max, min_improve
    )

    # Assert
    assert rate == rate_filt
    assert len(input_sig) == len(sig_filt)
    assert (type(sig_filt[0]) in [float, np.float64]) is True  # Check value type
    assert (True in np.isnan(sig_filt)) is False  # No NaN values
    assert (len(np.unique(sig_filt)) > 2) is True  # Values are not all zeros
