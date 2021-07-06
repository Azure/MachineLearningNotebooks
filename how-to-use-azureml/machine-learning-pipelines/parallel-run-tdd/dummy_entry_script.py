# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
"""This is a self-contained module to simulate azureml_user.parallel_run.EntryScript."""
import logging


class SingletonMeta(type):
    """This is a singleton metaclass."""

    _instances = {}

    def __call__(cls, *args, **kwargs):
        """Lookup and create a single instance for the class if not exists, and then return it."""
        if cls not in cls._instances:
            cls._instances[cls] = super(SingletonMeta, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class DummyEntryScript(metaclass=SingletonMeta):
    """This is a helper module for users to use in entry script."""

    def __init__(self, logging_level=None):
        """Initialize an instance."""
        self._logger = None
        self._logger_name = None
        self._log_dir = None
        self._output_dir = None
        self.logging_level = logging_level if logging_level else logging.INFO
        self._entry_script_dir = "."
        self._working_dir = "."

    def __reduce__(self):
        """Declare what to pickle."""
        return (self.__class__, ())

    def config_log(self):
        """Sink the log to console."""
        logger = logging.getLogger("ParallelRunStep")
        logger.setLevel(logging.DEBUG)
        hdl = logging.StreamHandler()
        hdl.setLevel(self.logging_level)
        formatter = logging.Formatter("%(asctime)s|%(name)s|%(funcName)s()|%(levelname)s|%(message)s")
        hdl.setFormatter(formatter)
        logger.addHandler(hdl)
        self._logger = logger

    @property
    def logger(self):
        """Return a logger to write logs to users/ folder.

        The folder will show up in run detail in azure portal.
        """
        if self._logger is None:
            self.config_log()

        return self._logger

    @property
    def log_dir(self):
        """Return the full path containing user logs."""
        return "logs"

    @property
    def working_dir(self):
        """Return the root directory for this run.

        This directory contains the entry script, driver, logs, etc.
        Each worker changes to its own folder under this folder to avoid model download conflict.
        """
        return "."

    @property
    def output_dir(self):
        """Return the full path of the directory containing generated temp results and final result for 'append_row'.

        Users should also use this folder to store the output of their entry script.
        Users don't need to create this directory.
        """
        return "output"

    @property
    def agent_name(self):
        """Return the agent name."""
        return "agent000"
