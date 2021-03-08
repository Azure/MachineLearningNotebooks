"""
Test aml.py
"""
import pytest

from ..aml import remove_mini_batch_directory_from_path


@pytest.mark.parametrize(
    "filepath, expected_filepath",
    [
        (
            (
                "/mnt/batch/tasks/shared/LS_root/jobs/sandbox/"
                "azureml/9f448dc4-19b5-4f63-82c4-55f47ff5d689/wd/tmpgnhd6dp4/"
                "testing_nested_directory/random_noise_1.wav"
            ),
            "testing_nested_directory/random_noise_1.wav",
        ),
        (
            (
                "/mnt/batch/tasks/shared/LS_root/jobs/sandbox/"
                "azureml/9f448dc4-19b5-4f63-82c4-55f47ff5d689/wd/tmpgnhd6dp4/"
                "random_noise_2.wav"
            ),
            "random_noise_2.wav",
        ),
    ],
)
def test_remove_mini_batch_directory_from_path(filepath: str, expected_filepath: str):
    """Test remove_mini_batch_directory_from_path

    Parameters
    ----------
    filepath : str
        Parameter to test
    expected_filepath : str
        Parameter for validation
    """
    filepath = remove_mini_batch_directory_from_path(filepath)

    assert filepath == expected_filepath
