"""Definitions for logging values to a structured text file."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import numpy as np
from dltools.filelock import filelock


class FileLogWriter(object):
    """Let's you log arbitrary information to a log file."""

    def __init__(self, filename, flush_frequency=2):
        """Initializes a new instance of the FileLogWriter class.

        Args:
            filename: The filename of the log file.
            flush_frequency: The frequency at which the data shall be written
                to the log file.
        """
        self.filename = filename
        self.flush_frequency = flush_frequency
        self.cache = []

    def _flush_cache(self):
        # Make sure that no one is currently reading the log file
        with open(self.filename, "a") as file_handle:
            with filelock.FileLock(self.filename):
                for entry in self.cache:
                    file_handle.write(json.dumps(entry))
                    file_handle.write('\n')

        # Reset the cache
        self.cache = []

    def log(self, key, message):
        """Logs some information to the file

        Args:
            key: The key under which the information shall be stored.
            message: The value that shall be logged.
        """
        # If message is a numpy array, then convert it to a python list first
        if type(message).__module__ == np.__name__:
            message = message.tolist()

        self.cache.append({"key": key, "value": message})
        if len(self.cache) > self.flush_frequency:
            self._flush_cache()


class FileLogReader(object):
    """Allows you to retrieve the information that is stored in a log file."""

    def __init__(self, filename):
        """Initializes a new instance of the FileLogReader class.

        Args:
            filename: Name of the log file.
        """
        self._logs = {}
        self.filename = filename
        self.handle = None
        self.handle = open(self.filename, "r")

    def close(self):
        """Closes the log file."""
        self.handle.close()

    def logs(self, key):
        """Returns the value corresponding to `key` or an empty list.

        Args:
            key: The logging key.

        Returns:
        Value corresponding to the key or an empty list if the key doesn't
        exist.
        """
        if not key in self._logs:
            return []
        else:
            return self._logs[key]

    def update(self):
        """Reads new logs from the log file."""
        # Read the log messages
        with filelock.FileLock(self.filename):
            for row in self.handle:
                message = json.loads(row)

                # Does the key already exist?
                if message["key"] not in self._logs:
                    self._logs[message["key"]] = []

                self._logs[message["key"]].append(message["value"])
