"""
Common utilities between preprocessing steps
"""
from pathlib import Path
from typing import Callable

from .aml import get_logger

log = get_logger(__name__)


def reuse(preprocessing_func: Callable):
    """Decorator to skip processing of a function if in non-overwrite mode and the output_filepath already exists

    Intended to be used with functions that have the following signature

    def preprocessing_func(input_filepath: str, output_filepath: str, *args, overwrite: bool = False, **kwargs):
        ...


    Parameters
    ----------
    preprocessing_func : Callable
        Function to be decorated

    Returns
    -------
    preprocessing_func_wrapper : Callable
        Decorated function
    """

    def preprocessing_func_wrapper(
        input_filepath: str,
        output_filepath: str,
        *args,
        overwrite: bool = False,
        **kwargs
    ):
        """Wrapper function around a preprocessing function

        Parameters
        ----------
        input_filepath : str
            Input filepath to process
        output_filepath : str
            Output filepath to write the processed file
        overwrite : bool
            True if output_filepath should be overwritten
        args
            Argument list for preprocessing_func
        kwargs
            Keyword arguments for preprocessing_func
        """
        # pylint: disable=missing-param-doc
        # args and kwargs have problems with pylint param documentation
        Path(output_filepath).parent.mkdir(parents=True, exist_ok=True)
        if not Path(output_filepath).exists():
            log.info(
                "%s does not exist. Generating file for the first time", output_filepath
            )
            preprocessing_func(
                input_filepath, output_filepath, overwrite=overwrite, *args, **kwargs
            )
        elif overwrite:
            log.info("Overwriting %s", output_filepath)
            preprocessing_func(
                input_filepath, output_filepath, overwrite=overwrite, *args, **kwargs
            )
        else:
            log.info("%s exists in non-overwrite mode. Reusing file", output_filepath)

    return preprocessing_func_wrapper
