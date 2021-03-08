"""
Test timestamp.py
"""
import pytest

from ..timestamp import format_seconds_to_timestamp, format_timestamp_to_seconds


@pytest.mark.parametrize(
    "seconds, expected_timestamp",
    [
        (60, "00:01:00.000"),
        (90, "00:01:30.000"),
        (90.5, "00:01:30.500"),
        (60 * 60 * 10 + 1, "10:00:01.000"),
    ],
)
def test_format_seconds_to_timestamp(seconds: int, expected_timestamp: str):
    """
    Test format_seconds_to_timestamp

    Parameters
    ----------
    seconds : int
        Parameter to test
    expected_timestamp : str
        Parameter for validation
    """
    timestamp = format_seconds_to_timestamp(seconds=seconds)
    assert timestamp == expected_timestamp


@pytest.mark.parametrize(
    "timestamp, expected_seconds",
    [
        ("0:01:00.000", 60),
        ("0:01:00.500", 60.5),
        ("0:01:00.500000", 60.5),
        ("00:01:00.000", 60),
        ("00:01:30.000", 90),
        ("00:01:30.500", 90.5),
        ("10:00:01.000", 60 * 60 * 10 + 1),
        ("10:00:01.600000", 60 * 60 * 10 + 1 + 0.6),
        ("10:00:01", 60 * 60 * 10 + 1),
        ("3:01", 3 * 60 + 1),
        ("3:01.500", 3 * 60 + 1 + 0.5),
        ("3:01.500000", 3 * 60 + 1 + 0.5),
        ("13:01", 13 * 60 + 1),
        ("13:01.500", 13 * 60 + 1 + 0.5),
        ("13:01.500000", 13 * 60 + 1 + 0.5),
    ],
)
def test_format_timestamp_to_seconds(timestamp: str, expected_seconds: float):
    """
    Test format_timestamp_to_seconds

    Parameters
    ----------
    timestamp : str
        Time formatted as HH:MM:SS.sss
    expected_seconds : float
        Parameter for validation.
    """
    seconds = format_timestamp_to_seconds(timestamp=timestamp)
    assert seconds == expected_seconds
