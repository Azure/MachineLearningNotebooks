"""
Extract, Transform, Load (ETL) Pipeline for Audio Data

Available steps are
- Extraction of audio from video files
- Dynamic Range Compression to amplify main signal
- Denoising to reduce background noise
"""
import argparse
from os.path import join
from pathlib import Path
from typing import List

from src.utils.aml import get_logger, remove_mini_batch_directory_from_path
from src.utils.audio import run_ffmpeg_dynaudnorm
from src.utils.audio_filtering_utils import denoise_audio
from src.utils.video import extract_audio_from_video

log = get_logger(__name__)


def init():
    """Init for Azure Machine Learning ParallelRunStep

    Raises
    ------
    ValueError
        If transform_order is incorrectly specified
    """
    # pylint: disable=global-variable-undefined
    global base_dir
    global output_dir
    global overwrite
    global sample_rate
    global transform_order

    TRANSFORM_ORDER = ["compress", "denoise"]
    VALID_TRANSFORM_OPTIONS = {"compress", "denoise"}

    parser = argparse.ArgumentParser(description="ETL Pipeline for Audio Data")
    parser.add_argument(
        "--base-dir",
        required=True,
        help="""Base Directory that files will be read from.
        Should be the Azure Blob Container to use""",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="""Directory to write ETL audio files from.
        Will be combined with base_dir for the full path""",
    )
    parser.add_argument(
        "--overwrite",
        type=bool,
        default=True,
        help="True if the pipeline should overwrite audio files if they already exist",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=None,
        required=False,
        help="Target Sampling Rate. Default is to use audio's inherent sample rate",
    )
    parser.add_argument(
        "--transform-order",
        default=TRANSFORM_ORDER,
        help=f"""Transformation order to apply on extracted audio. Should be comma separated
        Default is {TRANSFORM_ORDER}.
        Valid transformation options are {VALID_TRANSFORM_OPTIONS}""",
    )
    args, _ = parser.parse_known_args()

    base_dir = args.base_dir
    output_dir = args.output_dir
    overwrite = args.overwrite
    sample_rate = args.sample_rate
    transform_order = args.transform_order

    # Postprocess sample_rate.
    if not sample_rate:
        sample_rate = None

    # Postprocess transform_order
    if transform_order is not None and transform_order.strip() != "":
        transform_order = transform_order.split(",")
        for idx, transform_name in enumerate(transform_order):
            cleaned_transform_name = transform_name.strip()
            if cleaned_transform_name not in VALID_TRANSFORM_OPTIONS:
                raise ValueError(
                    f"""Transform "{transform_name}" is not a valid transformation option.
                    Valid transformation options are {VALID_TRANSFORM_OPTIONS}"""
                )
            transform_order[idx] = cleaned_transform_name


def run(mini_batch: List[str]) -> List[str]:
    """Run step for ParallelRunStep

    Parameters
    ----------
    mini_batch : List[str]
        List of video filepaths to process.

    Returns
    -------
    mini_batch : List[str]
        Needs to return a list of equal length for correspondence with the input mini_batch
    """
    log.info(f"Mini Batch Processing: {mini_batch}")

    for video_filepath in mini_batch:
        log.info(f"Running ETL on {video_filepath}")

        audio_filename = remove_mini_batch_directory_from_path(video_filepath)
        audio_filename = f"{Path(audio_filename).stem}.wav"
        audio_filepath = join(base_dir, output_dir, audio_filename)

        log.info(f"Extracting Audio from {video_filepath} to {audio_filepath}")
        extract_audio_from_video(
            video_filepath, audio_filepath, sr=sample_rate, overwrite=overwrite
        )

        for transform_name in transform_order:
            if transform_name == "compress":
                log.info(f"Running Dynamic Range Compression on {audio_filename}")
                run_ffmpeg_dynaudnorm(
                    audio_filepath, audio_filepath, overwrite=overwrite
                )
            elif transform_name == "denoise":
                log.info(f"Running Denoising on {audio_filename}")
                denoise_audio(
                    audio_filepath,
                    audio_filepath,
                    overwrite=overwrite,
                )

    return mini_batch
