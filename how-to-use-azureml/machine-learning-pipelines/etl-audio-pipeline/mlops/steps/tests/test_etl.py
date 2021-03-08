"""
Test etl.py
"""
import sys
import tempfile
from os.path import join
from pathlib import Path

import pytest
from src.utils.aml import get_logger

from ..etl import init, run

log = get_logger(__name__)


@pytest.mark.parametrize(
    "sample_rate, transform_order",
    [
        (0, ""),
        (0, "compress"),
        (0, "denoise"),
        (0, "compress, denoise"),
        (0, "denoise,compress"),
        (16000, ""),
    ],
)
def test_etl(sample_rate: int, transform_order: str):
    """Test init() and run() for etl.py

    Due to the way ParallelRunStep, it is necessary to run `init` then
    `run` so the global variables are set from the argparse

    Parameters
    ----------
    sample_rate : int
        Parameter to test
    transform_order : str
        Parameter to test
    """
    with tempfile.TemporaryDirectory() as tempdir:
        input_dir = join("data", "audio")
        output_dir = str(tempdir)

        sys.argv[1:] = [
            "--base-dir",
            "",
            "--input-dir",
            input_dir,
            "--output-dir",
            output_dir,
            "--overwrite",
            str(True),
            "--sample-rate",
            str(sample_rate),
            "--transform-order",
            transform_order,
        ]

        init()

        audio_filenames = [
            join(input_dir, Path(f).name) for f in Path(join(input_dir)).glob("*.wav")
        ]
        run(audio_filenames)

        assert len(audio_filenames) > 0

        # Directory not empty
        for audio_filename in audio_filenames:
            audio_filename = Path(audio_filename).name
            log.info(join(output_dir, audio_filename))
            assert Path(join(output_dir, audio_filename)).exists()
