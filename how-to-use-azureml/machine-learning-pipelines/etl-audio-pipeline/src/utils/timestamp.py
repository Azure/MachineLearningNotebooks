"""
General timestamp preprocessing utilities
"""
import datetime
import re


def format_seconds_to_timestamp(seconds: float):
    """Format seconds to HH:MM:SS.sss (millisecond precision)

    Parameters
    ----------
    seconds : float
        Seconds count expressed as floating point number

    Returns
    -------
    timestamp : str
        Time formatted as HH:MM:SS.sss
    """
    timestamp = datetime.timedelta(seconds=seconds)

    timestamp = str(timestamp)

    # Timestamp was in HH:MM:SS format instead of HH:MM:SS.sss
    if "." not in timestamp:
        timestamp = timestamp + ".000"
    # Timestamp was in HH:MM:SS.ssssss (microseconds) instead of HH:MM:SS.sss
    else:
        timestamp = timestamp[:-3]

    # Timestamp was in H:MM:SS format instead of HH:MM:SS
    if int(timestamp[0]) == 0:
        timestamp = "0" + timestamp

    return timestamp


def format_timestamp_to_seconds(timestamp: str):
    """Format HH:MM:SS.sss (millisecond precision) to seconds

    Parameters
    ----------
    timestamp : str
        Time formatted as HH:MM:SS.sss

    Returns
    -------
    seconds : float
        Seconds count expressed as floating point number

    Raises
    ------
    ValueError
        If timestamp does not meet any of the expected formats
    """
    hours = minutes = seconds = milliseconds = 0

    # H:MM:SS or HH:MM:SS
    if re.match(r"^\d{1,2}:\d{2}:\d{2}$", timestamp):
        hours, minutes, seconds = timestamp.split(":")
    # M:SS or MM:SS
    elif re.match(r"^\d{1,2}:\d{2}$", timestamp):
        minutes, seconds = timestamp.split(":")
    # H:MM:SS.sss or HH:MM:SS.sss
    elif re.match(r"^\d{1,2}:\d{2}:\d{2}\.\d{3,}$", timestamp):
        hours, minutes, seconds = timestamp.split(":")
        seconds, milliseconds = seconds.split(".")
        if len(milliseconds) < 3:
            milliseconds = milliseconds + 0 * (3 - len(milliseconds))
        if len(milliseconds) > 3:
            milliseconds = milliseconds[:3]
    # M:SS.sss or MM:SS.sss
    elif re.match(r"^\d{1,2}:\d{2}\.\d{3,}$", timestamp):
        minutes, seconds = timestamp.split(":")
        seconds, milliseconds = seconds.split(".")
        if len(milliseconds) < 3:
            milliseconds = milliseconds + 0 * (3 - len(milliseconds))
        if len(milliseconds) > 3:
            milliseconds = milliseconds[:3]
    else:
        raise ValueError(
            'Parameter "timestamp" does not meet any of the expected formats'
        )

    return (
        int(hours) * 3600 + int(minutes) * 60 + int(seconds) + 0.001 * int(milliseconds)
    )


def timestamp_correction(timestamp: str):
    """Timstamp Correction

    Ensures that the timestamp formatted as M:SS or M:SS.sss is correctd and converts
    the timestamp to seconds

    Parameters
    ----------
    timestamp : str
        Timestamp formatted as M:SS or M:SS.sss

    Returns
    -------
    seconds : int
        Returns the corrected timestamp in seconds
    """
    timestamp = str(timestamp)
    timestamp = timestamp.strip()

    try:
        seconds = format_timestamp_to_seconds(timestamp)
    except ValueError:
        seconds = None

    return seconds
