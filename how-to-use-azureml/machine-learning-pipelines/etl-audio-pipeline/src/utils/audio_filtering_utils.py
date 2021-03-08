"""
Helper functions for denoising and filtering audio tracks
"""

import os
from typing import Tuple

import librosa
import matplotlib.pyplot as plt
import noisereduce as nr
import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, filtfilt

from .common import reuse

# pylint: disable=invalid-name


def find_noise(
    audio: np.ndarray, sample_rate: int, noise_len: int = 1
) -> Tuple[np.ndarray, bool]:
    """
    Automatic noise detection

    Args:
        audio: audio track
        sample_rate: audio sample rate
        noise_len: length of noise clip (in seconds). default to 1
    ---

    Return:
        noise_clip, found_clip: clip with noise and bool_flag. if the noise clip is not found, a np.zeros array is
        returned
    """

    # transform amplitude to dB for noise detection
    audio_db = librosa.amplitude_to_db(audio, ref=np.max)
    found_clip = False
    noise_clip = np.zeros(noise_len * sample_rate)
    for i in range(len(audio_db)):
        clip = audio_db[i : i + noise_len * sample_rate]
        if not found_clip:
            # -35dB is noise
            if np.max(clip) <= -35:
                noise_clip = audio[i : i + noise_len * sample_rate]
                found_clip = True

    # check noise_clip dimension:
    if len(noise_clip) == 1:  # if last audio sample has amplitude < -35dB:
        noise_clip = np.zeros(noise_len * sample_rate)
        found_clip = False

    return noise_clip, found_clip


def compute_snr(audio: np.ndarray, noise_clip: np.ndarray) -> float:
    """
    Signal-to-noise ratio (SNR) for the recording s(t), where s(t) = x(t) + n(t).
    A recording s(t) is made of the 'real signal' x(t) and the noise n(t).

    For random singal SNR = 10*log(var_x/var_n), where var=variance.

    Args:
        audio: recorder track
        noise_clip: noise clip
    ---

    Return:
        snr (dB)

    """
    var_noise = np.var(noise_clip)  # noise variance
    if var_noise == 0:
        snr = 30
    else:
        # x(t) variance. s(t) = x(t) + n(t), where n(t): noise, s(t): recording
        var_sign = np.var(audio) - np.var(noise_clip)
        snr = 10 * np.log10(var_sign / var_noise)

    return snr


def butter_filter(
    x: np.ndarray, f_c: int, filt_type: str, n: int = 8, x_band: int = 500
) -> np.ndarray:
    """
    Application of Butterworth filter on the signal.

    Args:
        x: original signal
        f_c: cut-off frequency of the filter (Hz)
        filt_type: type of filter. 'low' for low-pass filter, 'high' for high-pass filter.
        x_band: max frequency of the signal to detect. For clapperboard, default to 500 (Hz)
        n: filter order
    ---
    Return:
        filtered_x
    """
    f_s = 2 * x_band  # based on Nyquist theorem
    w = f_c / (f_s / 2)  # Normalize the frequency
    b_filt, a_filt = butter(n, w, filt_type)
    # set output=sos for numerical computational problems
    # padlen to presever orignal signal length
    filtered_x = filtfilt(
        b_filt, a_filt, x, padlen=max(3 * len(a_filt), 3 * len(b_filt), 150)
    )

    return filtered_x


def compare_plots(
    filtered_signal: np.ndarray, signal: np.ndarray, sample_rate: str, filename: str
):
    """
    Plots comparison: before and after the filtering

    Args:
        filtered_signal: signal after the filtering
        signal: original signal
        sample_rate: sample rate of the signals
        filename: filename of the original signal

    """
    fig, a_x = plt.subplots(2, figsize=(12, 5))
    fig.subplots_adjust(hspace=0.5)

    timesteps_out = np.arange(len(filtered_signal)) / sample_rate  # in seconds
    a_x[0].plot(timesteps_out, signal[: len(timesteps_out)])
    a_x[0].set_xlabel("Time (s)")
    a_x[0].set_ylabel("Amplitude")
    a_x[0].set_title(f"Raw Audio: {filename}")

    a_x[1].plot(timesteps_out, filtered_signal)
    a_x[1].set_xlabel("Time (s)")
    a_x[1].set_ylabel("Amplitude")
    a_x[1].set_title("Filtered Audio")


def save_wav(wav: np.ndarray, path: str, output_name: str, sample_rate: int):
    """
    Save wav file in the specified path

    Args:
        wav: audio file to be saved
        path: path to store results
        output_name: output name
        sample_rate: sample rate

    """
    wav *= 32767 / max(0.01, np.max(np.abs(wav)))
    wavfile.write(
        os.path.join(path, f"{output_name}.wav"), sample_rate, wav.astype(np.int16)
    )


def audio_updating(
    filtered_audio: np.ndarray,
    sample_rate: int,
    audio: np.ndarray,
    snr_in: float,
    min_improve: float,
) -> Tuple[np.ndarray, bool, float]:
    """
    Audio updating for SNR loop. If after filtering SNR is higher than before filtering, audio = filterd_audio.
    Otherwise audio is not updated. Moreover, a check on how much SNR is place: only if SNR improvement is greater or
    equal to min_improve, the SNR is not interrupted.

    Args:
        filtered_audio: filtered audio
        sample_rate: sample rate
        audio: original audio file
        snr_in: SNR before filtering
        min_improve: min SNR improvement required to continue the loop
    ---
    Return:
        audio_updated, break_loop, snr
    """
    # initialization
    break_loop = True
    audio_updated = audio
    snr = snr_in

    noise_clip, found_clip = find_noise(filtered_audio, sample_rate)  # noise extraction
    if not found_clip:
        return audio_updated, break_loop, snr

    # compute SNR
    snr_out = compute_snr(filtered_audio, noise_clip)
    # update audio
    if (snr_out - snr_in) >= min_improve:
        audio_updated = filtered_audio
        snr = snr_out
        break_loop = False

    return audio_updated, break_loop, snr


def set_bandpass_filter_freq(researched_signal: str) -> Tuple[int, int]:
    """
    Set cut-off frequencies of BPF based on researched signal

    Args:
        researched_signal: str
            clapperboard or voice
    ---
    Return:
        fc_low, fc_high
    """
    if researched_signal == "clapperboard":
        fc_low = 50  # Hz. Cut-off freq for LPF
        fc_high = 5  # Hz. Cut-off freq for HPF
    else:  # for human speech
        fc_low = 450  # Hz. Cut-off freq for LPF
        fc_high = 80  # Hz. Cut-off freq for HPF

    return fc_low, fc_high


def filter_whole_audio(
    sig: np.ndarray, rate: int, researched_signal: str = "voice", n: int = 4
) -> Tuple[np.ndarray, int]:
    """Run filtering across the whole audio file.

    Args:
        sig (np.ndarray): The audio signal
        rate (int): Sampling rate (Hz)
        researched_signal (str): Either 'voice' or 'clapperboard'
        n (int): Butterworth filter order

    Returns:
        sig_filt (np.ndarray): The filtered audio signal
        rate (int): Sampling rate (Hz)
    """
    # Sample frequency must be greater or equal to 2*B, where B is the max frequency of the signal.
    # (Nyquist Theorem)
    nyq = rate / 2

    # Get cut-off frequencies depending on whether 'voice' or 'clapperboard'
    fc_low, fc_high = set_bandpass_filter_freq(researched_signal)

    # LPF
    w = fc_low / nyq  # normalize cut-off freq
    b, a = butter(n, w, "low")
    sig_LPF = filtfilt(b, a, sig)  # apply the filter

    # HPF
    w = fc_high / nyq  # normalize cut-off freq
    b, a = butter(n, w, "high")
    sig_filt = filtfilt(b, a, sig_LPF)  # apply the filter

    return sig_filt, rate


def denoise_whole_audio(sig: np.ndarray, rate: int) -> Tuple[np.ndarray, int]:
    """Run denoising across all of the audio

    Args:
        sig (np.ndarray): The audio signal
        rate (int): Sampling rate (Hz)

    Returns:
        sig_filt (np.ndarray): The filtered audio signal
        rate (int): Sampling rate (Hz)
    """

    # Check if audio is too long (will lead to MemoryError)
    # Note: Try/Except for MemoryError does not seem to work
    is_over_30_min = len(sig) / rate / 60 > 30

    if is_over_30_min is True:
        # If signal is so big that it can't be processed all at once
        split_a = int(len(sig) / 4)
        split_b = int(len(sig) / 4) * 2
        split_c = int(len(sig) / 4) * 3

        def denoise_partial_signal(
            sig_to_split: np.ndarray, split_point_1: int, split_point_2: int, rate: int
        ) -> np.ndarray:
            """Temporary function to denoise part of a signal

            Args:
                sig_to_split (np.ndarray): The audio signal to split
                split_point_1 (int): Index point to begin split
                split_point_2 (int): Index point to end split
                rate (int): Sampling rate of the signal

            Returns:
                sig_temp (np.ndarray): The denoised partial signal
            """

            # Cast values to float32 if currently at float64
            if isinstance(sig_to_split[0], np.float64):
                sig_to_split = sig_to_split.astype(np.float32)

            split_sig = sig_to_split[split_point_1:split_point_2]
            noise_clip, _ = find_noise(split_sig, rate)

            sig_temp = nr.reduce_noise(
                audio_clip=split_sig, noise_clip=noise_clip, verbose=False
            )
            sig_temp = np.asarray(sig_temp).astype(np.float32)
            sig_temp /= np.max(np.abs(sig_temp), axis=0)  # rescale audio
            return sig_temp

        # Parts
        sig_filt1 = denoise_partial_signal(sig, 0, split_a, rate)
        sig_filt2 = denoise_partial_signal(sig, split_a, split_b, rate)
        sig_filt3 = denoise_partial_signal(sig, split_b, split_c, rate)
        sig_filt4 = denoise_partial_signal(sig, split_c, len(sig), rate)

        # Combine
        sig_filt = np.concatenate([sig_filt1, sig_filt2, sig_filt3, sig_filt4])
        print("Signal processed as parts, then concatenated post-processing")
    else:
        # Detect noise clip
        noise_clip, _ = find_noise(sig, rate)

        sig_filt = nr.reduce_noise(audio_clip=sig, noise_clip=noise_clip, verbose=False)
        sig_filt = np.asarray(sig_filt).astype(np.float32)
        sig_filt /= np.max(np.abs(sig_filt), axis=0)  # rescale audio

    return sig_filt, rate


@reuse
def denoise_audio(
    input_audio_filepath: str,
    output_audio_filepath: str,
    overwrite: bool = False,
):
    """Wrapper function around denoise_whole_audio to match signature required for reuse

    Parameters
    ----------
    input_audio_filepath : str
        Input filepath to the audio file to denoise
    output_audio_filepath : str
        Output filepath to the audio file that is denoised
    overwrite : bool
        True if output_audio_filepath should be overwritten
    """
    # pylint: disable=unused-argument
    # overwrite is used by the decorator reuse.
    # It is kept in this function signature to reflect that it is a valid parameter to pass in.
    sig, sample_rate = librosa.load(input_audio_filepath, sr=None)
    audio_clip, sample_rate = denoise_whole_audio(sig, sample_rate)
    wavfile.write(output_audio_filepath, sample_rate, audio_clip)


def clean_audio_with_filt_denoise(
    sig: np.ndarray,
    rate: int,
    researched_signal: str,
    discard_snr_11: bool,
    iter_max: int,
    min_improve: float,
) -> Tuple[np.ndarray, int]:
    """
    Audio preprocessing system to reduce noise

    Args:
        sig (np.ndarray): The audio signal
        rate (int): Sampling rate (Hz)
        researched_signal (str): Either 'voice' or 'clapperboard'
        discard_snr_11 (bool): Discard or not signal with SNR < 10
        iter_max (int): Max number of loop iterations
        min_improve (float): Min SNR improvement

    Returns:
        sig_filt (np.ndarray): The filtered audio signal
        rate (int): Sampling rate (Hz)
    """

    # set cut-off frequencies based on researched signal
    fc_low, fc_high = set_bandpass_filter_freq(researched_signal)

    # set SNR loop parameters
    iter_n = 0  # number of iteration of SNR loop

    # 1. rescale audio: if after rescaling max(audio) = 0 meant that the signal is empty, discard
    if np.abs(np.max(sig)) != 0:
        sig /= np.max(np.abs(sig), axis=0)  # rescale audio

        # 2. detect a noise clip
        noise_clip, found_noise_clip = find_noise(sig, rate)

        # 3. Enter in the loop
        if found_noise_clip:
            # compute SNR
            snr = compute_snr(sig, noise_clip)
            break_loop = False

            # iteration count
            while iter_n < iter_max:
                iter_n += 1

                # save output
                if snr >= 30 or break_loop:
                    # if SNR>= 30dB, no need to filter the audio track
                    iter_n = iter_max  # move to next file
                    # save results
                    sig_filt = sig

                # denoising
                elif 15 <= snr < 30 or (not discard_snr_11 and snr < 11):
                    sig_filt, _ = denoise_whole_audio(sig, rate)
                    # filtering evaluation and audio updating
                    sig_filt, break_loop, snr = audio_updating(
                        sig_filt, rate, sig, snr, min_improve
                    )

                # band-pass filter
                elif 11 <= snr < 15:

                    # low-pass filtering
                    low_audio = butter_filter(sig, fc_low, filt_type="low")
                    # high-pass filtering
                    sig_filt = butter_filter(low_audio, fc_high, filt_type="high")
                    sig_filt /= np.max(np.abs(sig_filt), axis=0)  # rescale audio
                    # filtering evaluation and audio updating
                    sig_filt, break_loop, snr = audio_updating(
                        sig_filt, rate, sig, snr, min_improve
                    )

    return sig_filt, rate
